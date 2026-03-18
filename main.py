"""
DeepRack Chatbot / Agent API — FastAPI server with optional web research.
Connects to an Ollama-powered LLM on a DeepRack GPU rack.
All configuration is via environment variables — no code changes needed.

When ENABLED_TOOLS is set (e.g. "web_search,url_reader"), the API acts as
a research agent that automatically searches the web before answering.
When ENABLED_TOOLS is empty, it's a normal chatbot with optional RAG.
"""

import os
import re
import json
import time
import hashlib
import logging
import asyncio
import urllib.parse
from typing import Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field

from rag import RAGEngine

# ═══════════════════════  LOGGING  ═══════════════════════

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("chatbot-api")

# ═══════════════════════  CONFIG  ═══════════════════════

CHATBOT_NAME = os.getenv("CHATBOT_NAME", "AI Assistant")
ORG_NAME = os.getenv("ORG_NAME", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"
RAG_DATA_DIR = os.getenv("RAG_DATA_DIR", "/workspace/data")
RAG_DB_DIR = os.getenv("RAG_DB_DIR", "/workspace/chromadb")
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "20"))
MAX_RAG_CHUNKS = int(os.getenv("MAX_RAG_CHUNKS", "5"))
PORT = int(os.getenv("PORT", "8000"))

# Agent tools — set ENABLED_TOOLS="web_search,url_reader" to activate agent mode
_tools_raw = os.getenv("ENABLED_TOOLS", "")
ENABLED_TOOLS = [t.strip() for t in _tools_raw.split(",") if t.strip()] if _tools_raw else []
IS_AGENT = len(ENABLED_TOOLS) > 0

# Email notification settings (set by deployer, can also be fetched at runtime)
CHATBOT_ID = os.getenv("CHATBOT_ID", "")
CHATBOT_BASE_NAME = os.getenv("CHATBOT_BASE_NAME", "")
CHATBOT_IDENTIFIER = CHATBOT_ID or CHATBOT_BASE_NAME  # used for notification config lookups
EMAIL_NOTIFICATIONS = os.getenv("EMAIL_NOTIFICATIONS", "false").lower() == "true"
NOTIFY_WHEN = os.getenv("NOTIFY_WHEN", "")
BACKEND_INTERNAL_URL = os.getenv("BACKEND_INTERNAL_URL", "http://host.docker.internal:5000")
INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET", "deeprack-internal-2024")

# Runtime-refreshable notification config
_notification_config = {
    "enabled": EMAIL_NOTIFICATIONS,
    "notify_when": NOTIFY_WHEN,
    "last_refresh": 0,
}

_identity_block = (
    f"You are an AI assistant{f' for {ORG_NAME}' if ORG_NAME else ''}. "
    f"Your name is {CHATBOT_NAME}. "
    "You were created and deployed using the DeepRack GPU Cloud Platform by Bavision LLP. "
    "You are NOT built by Meta AI, Google, OpenAI, or any other company. "
    "Never mention Meta, Llama, or any underlying model architecture. "
    + (f"You work exclusively for {ORG_NAME}. When users ask who you work for or who made you, "
       f"always say you are the AI assistant for {ORG_NAME}. "
       f"You represent {ORG_NAME} and should help their customers, visitors, and team members. "
       if ORG_NAME else
       "If anyone asks who made you or who built you, say you were built using DeepRack by Bavision LLP. "
    )
)

DEFAULT_BAVISION_PROMPT = (
    _identity_block + "\n\n"
    "STRICT GRAMMAR AND LANGUAGE RULES — you MUST follow every rule below in every response without exception.\n\n"
    "1. SUBJECT-VERB AGREEMENT:\n"
    "   WRONG: 'The data are ready.' CORRECT: 'The data is ready.'\n"
    "   WRONG: 'Each of the models have been evaluated.' CORRECT: 'Each of the models has been evaluated.'\n"
    "   WRONG: 'A number of issues has been found.' CORRECT: 'A number of issues have been found.'\n"
    "   WRONG: 'The team are working on it.' CORRECT: 'The team is working on it.'\n\n"
    "2. ARTICLES (a, an, the):\n"
    "   Use 'a' before consonant sounds, 'an' before vowel sounds.\n"
    "   WRONG: 'a API key' CORRECT: 'an API key'\n"
    "   WRONG: 'an unique feature' CORRECT: 'a unique feature'\n"
    "   WRONG: 'an hour ago' — this is actually CORRECT because 'hour' has a silent h.\n"
    "   Use 'the' for specific/previously mentioned items, 'a/an' for general first mentions.\n"
    "   WRONG: 'Create the new project.' (first mention) CORRECT: 'Create a new project.'\n\n"
    "3. PRONOUN CLARITY:\n"
    "   Never use 'it', 'this', 'that', or 'they' if the reference is ambiguous.\n"
    "   WRONG: 'The model was trained on the dataset. It achieved 95% accuracy.' — unclear what 'it' refers to.\n"
    "   CORRECT: 'The model was trained on the dataset. The model achieved 95% accuracy.'\n"
    "   WRONG: 'Upload the file and click the button. It will process it.' — two ambiguous 'it's.\n"
    "   CORRECT: 'Upload the file and click the button. The system will process the file.'\n\n"
    "4. TENSE CONSISTENCY:\n"
    "   WRONG: 'Click the button. The system processed your request and will return the result.' — mixed tenses.\n"
    "   CORRECT: 'Click the button. The system processes your request and returns the result.'\n"
    "   Use present tense for instructions, past tense for completed actions, present perfect for ongoing states.\n\n"
    "5. COMMON WORD CONFUSIONS — memorise these:\n"
    "   its = possessive ('The model improved its accuracy.'), it's = it is ('It's ready.')\n"
    "   their = possessive, there = place, they're = they are\n"
    "   your = possessive, you're = you are\n"
    "   affect = verb ('This will affect results.'), effect = noun ('The effect is significant.')\n"
    "   fewer = countable ('fewer errors'), less = uncountable ('less memory')\n"
    "   ensure = make certain, insure = insurance\n"
    "   that = restrictive clause, which = non-restrictive clause (preceded by comma)\n"
    "   comprise = consist of ('The system comprises three modules.'), NEVER say 'is comprised of'.\n\n"
    "6. SENTENCE STRUCTURE:\n"
    "   No run-on sentences. Every sentence must have exactly one main clause or two clauses joined by a conjunction or semicolon.\n"
    "   WRONG: 'The model finished training you can download the weights.' — run-on.\n"
    "   CORRECT: 'The model finished training. You can now download the weights.'\n"
    "   Always use parallel structure in lists.\n"
    "   WRONG: 'The pipeline includes collecting data, data cleaning, to train the model, and evaluation.'\n"
    "   CORRECT: 'The pipeline includes collecting data, cleaning the data, training the model, and evaluating results.'\n"
    "   No dangling modifiers.\n"
    "   WRONG: 'Using the GPU, the model trained faster.'\n"
    "   CORRECT: 'The model trained faster when using the GPU.'\n\n"
    "7. PROFESSIONAL TONE:\n"
    "   NEVER start with filler phrases: 'Certainly!', 'Of course!', 'Absolutely!', 'Great question!', 'Sure thing!' — answer directly.\n"
    "   NEVER hedge on facts: say 'The answer is X' not 'I think the answer might be X.'\n"
    "   NEVER repeat the user's question back: if they ask 'How do I install PyTorch?' do NOT say 'To install PyTorch, you need to install PyTorch by...'\n"
    "   Use active voice: 'The system processes your request' not 'Your request is processed by the system.'\n"
    "   Use contractions naturally: don't, can't, won't, isn't — overly formal text sounds robotic.\n"
    "   Be specific: '45 minutes with 94.2% accuracy' not 'some time with good results.'\n\n"
    "8. COMPLETENESS:\n"
    "   Every sentence must be grammatically complete — it must have a subject and a predicate.\n"
    "   WRONG: 'Which is why it works.' (fragment) CORRECT: 'This is why the system works.'\n"
    "   WRONG: 'Because of the high demand.' (fragment) CORRECT: 'The delay occurred because of the high demand.'\n"
    "   Always use the Oxford comma in lists of three or more.\n"
    "   CORRECT: 'Python, JavaScript, and C++' not 'Python, JavaScript and C++.'\n\n"
    "9. FORMATTING:\n"
    "   Use numbered lists for sequential steps, bullet points for non-sequential items.\n"
    "   Use proper capitalisation for proper nouns: Python, Docker, Linux, NVIDIA, PyTorch, TensorFlow, GitHub.\n"
    "   Spell out numbers below 10 in prose; use numerals for 10 and above, and always with units.\n\n"
    "10. REVIEW BEFORE RESPONDING:\n"
    "   Before outputting any response, mentally review it for subject-verb agreement, article correctness, pronoun clarity, "
    "tense consistency, run-on sentences, and completeness. Fix any errors. Your grammar must be flawless."
)

_user_system_prompt = os.getenv("SYSTEM_PROMPT", "You are a helpful AI assistant.")


def get_system_prompt():
    """Get the full system prompt = Bavision branding + user prompt."""
    return DEFAULT_BAVISION_PROMPT.strip() + "\n\n" + _user_system_prompt.strip()


SYSTEM_PROMPT = get_system_prompt()

# ═══════════════════  TOOL IMPLEMENTATIONS  ═══════════════════
# These are only used when ENABLED_TOOLS includes the tool.


async def web_search(query: str, max_results: int = 8) -> list[dict]:
    """Search the web using DuckDuckGo HTML. Returns list of {title, url, snippet}."""
    logger.info(f"[web_search] Searching: {query}")
    try:
        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://duckduckgo.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        form_data = {"q": query, "b": "", "kl": ""}

        html = ""
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            for attempt in range(3):
                try:
                    resp = await client.post(url, data=form_data, headers=headers)
                    html = resp.text
                    if "result__a" in html:
                        break
                    # GET fallback
                    encoded_q = urllib.parse.quote_plus(query)
                    resp = await client.get(f"{url}?q={encoded_q}", headers=headers)
                    html = resp.text
                    if "result__a" in html:
                        break
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.warning(f"[web_search] Attempt {attempt + 1} error: {e}")
                    await asyncio.sleep(1)

        if not html or "result__a" not in html:
            logger.warning(f"[web_search] No results for: {query}")
            return []

        results = []

        # Try structured result_blocks first (more accurate extraction)
        result_blocks = re.findall(
            r'<div class="links_main links_deep result__body">(.*?)</div>\s*</div>',
            html,
            re.DOTALL,
        )
        if not result_blocks:
            result_blocks = re.findall(
                r'class="result__body">(.*?)</div>\s*</div>',
                html,
                re.DOTALL,
            )

        if result_blocks:
            for block in result_blocks[:max_results]:
                title_match = re.search(
                    r'class="result__a"[^>]*>(.*?)</a>', block, re.DOTALL
                )
                href_match = re.search(
                    r'class="result__a"\s+href="(.*?)"', block
                )
                snippet_match = re.search(
                    r'class="result__snippet">(.*?)</(?:a|span|div)',
                    block,
                    re.DOTALL,
                )
                title = (
                    re.sub(r"<[^>]+>", "", title_match.group(1)).strip()
                    if title_match
                    else ""
                )
                link = href_match.group(1) if href_match else ""
                if "uddg=" in link:
                    m = re.search(r"uddg=([^&]+)", link)
                    if m:
                        link = urllib.parse.unquote(m.group(1))
                snippet = (
                    re.sub(r"<[^>]+>", "", snippet_match.group(1)).strip()
                    if snippet_match
                    else ""
                )
                if title and link and link.startswith("http") and "duckduckgo.com/y.js" not in link:
                    results.append({"title": title, "url": link, "snippet": snippet})
        else:
            # Fallback: extract individual fields from the whole page
            titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL)
            hrefs = re.findall(r'class="result__a"\s+href="(.*?)"', html)
            snippets = re.findall(
                r'class="result__snippet">(.*?)</(?:a|span|div)', html, re.DOTALL
            )

            for i in range(min(len(titles), len(hrefs), max_results)):
                title = re.sub(r"<[^>]+>", "", titles[i]).strip()
                link = hrefs[i]
                if "uddg=" in link:
                    m = re.search(r"uddg=([^&]+)", link)
                    if m:
                        link = urllib.parse.unquote(m.group(1))
                snippet = re.sub(
                    r"<[^>]+>", "", snippets[i] if i < len(snippets) else ""
                ).strip()
                # Skip ads / non-http links
                if title and link and link.startswith("http") and "duckduckgo.com/y.js" not in link:
                    results.append({"title": title, "url": link, "snippet": snippet})

        logger.info(f"[web_search] Found {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"[web_search] Error: {e}")
        return []


async def url_reader(url: str) -> str:
    """Fetch and extract readable text from a URL."""
    logger.info(f"[url_reader] Reading: {url}")
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            html = resp.text

        # Remove non-content
        for tag in ["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe"]:
            html = re.sub(
                rf"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.DOTALL | re.IGNORECASE
            )
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

        # Try to find main content area
        main = ""
        for sel in [
            r"<article[^>]*>(.*?)</article>",
            r"<main[^>]*>(.*?)</main>",
            r'class="content"[^>]*>(.*?)</div>',
            r'class="article-body"[^>]*>(.*?)</div>',
            r"<body[^>]*>(.*?)</body>",
        ]:
            m = re.search(sel, html, re.DOTALL | re.IGNORECASE)
            if m:
                main = m.group(1)
                break
        if not main:
            main = html

        # Block elements → newlines
        for tag in ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "br", "hr"]:
            main = re.sub(rf"</?{tag}[^>]*>", "\n", main, flags=re.IGNORECASE)

        # Strip remaining tags
        text = re.sub(r"<[^>]+>", " ", main)
        # Decode HTML entities
        text = (
            text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
            .replace("&nbsp;", " ")
        )
        text = re.sub(r"&#?\w+;", " ", text)

        # Clean whitespace, skip short lines (nav items etc.)
        lines = []
        for line in text.split("\n"):
            line = re.sub(r"\s+", " ", line).strip()
            if line and len(line) > 15:
                lines.append(line)
        text = "\n".join(lines)

        if len(text) > 6000:
            text = text[:6000] + "\n\n[Content truncated]"
        if not text.strip():
            return ""
        logger.info(f"[url_reader] Extracted {len(text)} chars")
        return text
    except Exception as e:
        logger.warning(f"[url_reader] Failed {url}: {e}")
        return ""


# ═══════════════════  RESEARCH PIPELINE  ═══════════════════


async def do_research(user_query: str) -> str:
    """
    Automatic web research pipeline for agent mode.
    Searches the web, reads relevant pages, and returns a context block.
    """
    if "web_search" not in ENABLED_TOOLS:
        return ""

    logger.info(f"[research] Starting for: {user_query}")
    start = time.time()

    # Generate search queries (the original + a variant)
    queries = [user_query]
    words = user_query.lower()
    if "in " in words or "about " in words:
        queries.append(user_query + " companies list")
    if len(user_query.split()) > 5:
        key = [
            w
            for w in user_query.split()
            if len(w) > 3
            and w.lower()
            not in {
                "about", "research", "comprehensive", "detailed",
                "please", "write", "find", "india", "the",
            }
        ]
        if len(key) >= 2:
            queries.append(" ".join(key[:5]))

    # Run searches (max 2 to avoid rate-limiting)
    all_results = []
    seen_urls: set[str] = set()
    for q in queries[:2]:
        results = await web_search(q)
        for r in results:
            if r["url"] not in seen_urls:
                all_results.append(r)
                seen_urls.add(r["url"])
        if results:
            await asyncio.sleep(0.5)

    if not all_results:
        logger.warning("[research] No search results found")
        return ""

    # Read top 3-4 pages concurrently
    page_contents: list[dict] = []
    if "url_reader" in ENABLED_TOOLS:
        urls = [r["url"] for r in all_results[:4]]
        tasks = [url_reader(u) for u in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for u, content in zip(urls, results):
            if isinstance(content, str) and content.strip():
                page_contents.append({"url": u, "content": content})

    elapsed = time.time() - start
    logger.info(
        f"[research] Done in {elapsed:.1f}s — "
        f"{len(all_results)} search results, {len(page_contents)} pages read"
    )

    # Build context block
    ctx = "## WEB RESEARCH RESULTS\n\n"
    ctx += "The following information was gathered from real web searches.\n"
    ctx += "Use ONLY this information to answer. Do NOT add facts not found below.\n\n"

    ctx += "### Search Results:\n\n"
    for i, r in enumerate(all_results[:8], 1):
        ctx += f"{i}. **{r['title']}**\n"
        ctx += f"   URL: {r['url']}\n"
        if r.get("snippet"):
            ctx += f"   {r['snippet']}\n"
        ctx += "\n"

    if page_contents:
        ctx += "\n### Detailed Page Contents:\n\n"
        for pc in page_contents:
            ctx += f"--- Content from {pc['url']} ---\n"
            ctx += pc["content"] + "\n\n"

    # Keep context under 20k chars to fit in LLM context window
    if len(ctx) > 20000:
        ctx = ctx[:20000] + "\n\n[Research data truncated]"

    return ctx


# ═══════════════════  RAG ENGINE  ═══════════════════

rag_engine: Optional[RAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    if RAG_ENABLED:
        try:
            rag_engine = RAGEngine(data_dir=RAG_DATA_DIR, db_dir=RAG_DB_DIR)
            doc_count = rag_engine.document_count()
            logger.info(f"RAG engine initialized — {doc_count} document chunks indexed")
        except Exception as e:
            logger.warning(f"RAG engine initialization skipped: {e}")
            rag_engine = None
    else:
        logger.info("RAG disabled via RAG_ENABLED=false")

    mode = "agent" if IS_AGENT else "chatbot"
    logger.info(
        f"Chatbot '{CHATBOT_NAME}' started — "
        f"mode={mode}, Ollama at {OLLAMA_URL}, model {OLLAMA_MODEL}"
        + (f", tools: {ENABLED_TOOLS}" if ENABLED_TOOLS else "")
    )
    yield
    logger.info("Chatbot API shutting down")


# ═══════════════════  FASTAPI APP  ═══════════════════

app = FastAPI(
    title=f"{CHATBOT_NAME} API",
    description="AI Chatbot / Agent API powered by DeepRack",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=10000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=50)
    stream: bool = True
    session_id: str = ""  # optional, client can pass a session identifier


class IngestRequest(BaseModel):
    directory: str = RAG_DATA_DIR


class UpdateConfigRequest(BaseModel):
    system_prompt: str = Field(..., min_length=1, max_length=10000)


# ── Helpers ──


def _build_system_prompt(user_query: str, conversation_messages: list = None) -> str:
    """Build the full system prompt, optionally with RAG context and notification instructions."""
    base = SYSTEM_PROMPT

    # Inject notification-aware behaviour when email notifications are enabled
    if _notification_config["enabled"] and _notification_config["notify_when"]:
        notify_when = _notification_config["notify_when"]
        base += (
            "\n\n## ENQUIRY COLLECTION INSTRUCTIONS\n"
            "The owner of this chatbot has enabled email notifications for enquiries. "
            f"An enquiry is defined as: {notify_when}\n\n"
            "When you detect that a conversation is heading toward an enquiry situation:\n"
            "1. First, help the user with their question as best you can.\n"
            "2. When the topic requires human follow-up (pricing quotes, complaints, custom requests, "
            "demo scheduling, partnership proposals, or anything you cannot fully resolve), "
            "naturally ask for the visitor\'s **name**, **email address**, and **company name** (if relevant).\n"
            "3. Be conversational — don\'t dump all questions at once. Ask naturally as the conversation flows.\n"
            "4. Once you have the contact details, confirm them back to the user and let them know "
            "the team/owner will follow up.\n"
            "5. NEVER say you are sending an email or triggering a notification. "
            "Instead say something like \'I\'ll pass this along to our team\' or "
            "\'Our team will reach out to you at <email> shortly.\'"
            "\n"
        )

    if rag_engine and RAG_ENABLED:
        try:
            chunks = rag_engine.search(user_query, top_k=MAX_RAG_CHUNKS)
            if chunks:
                ctx = "\n\n---\n\n".join(c["text"] for c in chunks)
                base += (
                    "\n\n## Relevant Context from Knowledge Base\n"
                    "Use the following information to answer the user's question. "
                    "If the information is not relevant, rely on your general knowledge.\n\n"
                    + ctx
                )
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")

    return base


# ═══════════════════  ENDPOINTS  ═══════════════════


@app.get("/")
def root():
    return {
        "service": CHATBOT_NAME,
        "status": "running",
        "mode": "agent" if IS_AGENT else "chatbot",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "ingest": "/ingest",
            "upload": "/upload",
            "knowledge": "/knowledge",
            "config": "/config",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "chatbot_name": CHATBOT_NAME,
        "model": OLLAMA_MODEL,
        "rag_enabled": RAG_ENABLED,
        "rag_documents": rag_engine.document_count() if rag_engine else 0,
    }


async def _refresh_notification_config():
    """Refresh notification config from backend every 60 seconds."""
    now = time.time()
    if CHATBOT_IDENTIFIER and now - _notification_config["last_refresh"] > 60:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{BACKEND_INTERNAL_URL}/api/internal/chatbot-notification-config/{CHATBOT_IDENTIFIER}",
                    headers={"X-Internal-Secret": INTERNAL_API_SECRET},
                )
                if resp.status_code == 200:
                    cfg = resp.json()
                    _notification_config["enabled"] = cfg.get("email_notifications", False)
                    _notification_config["notify_when"] = cfg.get("notify_when", "")
                    _notification_config["last_refresh"] = now
        except Exception as e:
            logger.warning(f"[enquiry] Config refresh failed: {e}")


async def _check_enquiry_and_notify(conversation: list[dict]):
    """Background task: Analyse the full conversation for enquiry + contact details, then notify owner."""
    await _refresh_notification_config()

    if not _notification_config["enabled"] or not CHATBOT_IDENTIFIER or not _notification_config["notify_when"]:
        return
    try:
        notify_when = _notification_config["notify_when"]

        # Build a conversation transcript for the LLM to analyse
        transcript_lines = []
        for msg in conversation[-20:]:  # last 20 messages
            role = msg.get("role", "user")
            if role in ("user", "assistant"):
                transcript_lines.append(f"{role.upper()}: {msg.get('content', '')}")
        transcript = "\n".join(transcript_lines)

        # Step 1: Quick classification — is this an enquiry?
        classify_prompt = (
            "You are classifying chatbot conversations. The chatbot owner wants to be notified when: "
            f"{notify_when}\n\n"
            f"CONVERSATION:\n{transcript}\n\n"
            "Does this conversation match the notification criteria above? Reply with ONLY 'YES' or 'NO'."
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": classify_prompt, "stream": False, "options": {"temperature": 0, "num_predict": 10}},
            )
            answer = resp.json().get("response", "").strip().upper()

        if "YES" not in answer:
            return

        logger.info(f"[enquiry] Detected enquiry for chatbot {CHATBOT_IDENTIFIER}, generating email body")

        # Step 2: Generate a natural email body written as the GPT assistant
        bot_name = CHATBOT_NAME or "Your AI Assistant"
        email_prompt = (
            "You are writing an email to a business owner on behalf of their AI chatbot. "
            "The chatbot just had a conversation with a potential customer/visitor and detected an enquiry.\n\n"
            f"The chatbot's name is: {bot_name}\n"
            f"The owner wants to be notified when: {notify_when}\n\n"
            f"CONVERSATION:\n{transcript}\n\n"
            "Write an email body (plain text, not HTML) that the chatbot sends to its owner reporting this enquiry. "
            "Rules:\n"
            "- Start with 'Hi Boss,' (keep it short and professional)\n"
            "- Briefly describe what the customer wanted (1-2 sentences)\n"
            "- List ALL details the customer provided as bullet points using '•' (Name, Email, Company, Phone, Team size, Interest, etc.)\n"
            "- If details like name or email were NOT provided by the customer, write 'Not provided' — NEVER invent details\n"
            "- Add a short note about what follow-up is needed\n"
            f"- End with 'Yours faithfully,\n{bot_name}'\n"
            "- Keep it concise and professional — no fluff, no markdown formatting, no subject line\n"
            "- Do NOT wrap in quotes or add any preamble — output ONLY the email body"
        )

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": email_prompt, "stream": False, "options": {"temperature": 0.3, "num_predict": 600}},
            )
            email_body = resp.json().get("response", "").strip()

        if not email_body or len(email_body) < 20:
            logger.warning("[enquiry] LLM generated empty or too-short email body")
            return

        # Extract customer name and email from conversation for subject line
        email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', transcript)
        customer_email = email_match.group() if email_match else ""
        name_match = re.search(r"(?:I'm|I am|name is|Name:)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)", transcript)
        customer_name = name_match.group(1) if name_match else ""

        async with httpx.AsyncClient(timeout=15.0) as client:
            await client.post(
                f"{BACKEND_INTERNAL_URL}/api/internal/chatbot-enquiry-email",
                json={
                    "chatbot_id": CHATBOT_ID,
                    "base_name": CHATBOT_BASE_NAME,
                    "email_body": email_body,
                    "customer_name": customer_name,
                    "customer_email": customer_email,
                    "bot_name": bot_name,
                    "secret": INTERNAL_API_SECRET,
                },
            )
    except Exception as e:
        logger.warning(f"[enquiry] Notification check failed: {e}")


async def _save_chat_messages(session_id: str, user_msg: str, assistant_msg: str):
    """Fire-and-forget: send the user + assistant message pair to the platform backend."""
    if not BACKEND_INTERNAL_URL or not (CHATBOT_ID or CHATBOT_BASE_NAME):
        return
    try:
        payload = {
            "secret": INTERNAL_API_SECRET,
            "chatbot_id": CHATBOT_ID,
            "base_name": CHATBOT_BASE_NAME,
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": user_msg[:10000]},
                {"role": "assistant", "content": assistant_msg[:10000]},
            ],
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{BACKEND_INTERNAL_URL}/api/internal/chatbot-messages",
                json=payload,
            )
            if resp.status_code != 200:
                logger.warning(f"[chat-save] Backend returned {resp.status_code}")
    except Exception as e:
        logger.warning(f"[chat-save] Failed to save messages: {e}")


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Chat endpoint — works in two modes:
    • **Chatbot mode** (default): sends messages straight to LLM (+ RAG).
    • **Agent mode** (ENABLED_TOOLS set): automatically searches the web,
      reads pages, and injects the findings into the system prompt before
      the LLM generates a response.
    """
    user_messages = [m for m in req.messages if m.role == "user"]
    latest_query = user_messages[-1].content if user_messages else ""
    logger.info(f"[chat] User: {latest_query[:120]}...")

    # ── Session ID for message grouping ──
    if req.session_id:
        session_id = req.session_id[:100]
    else:
        # Derive from first user message so the same conversation always maps to one session
        first_user = user_messages[0].content if user_messages else ""
        session_id = hashlib.sha256(first_user.encode()).hexdigest()[:16]

    # ── Fire-and-forget enquiry detection ──
    if CHATBOT_IDENTIFIER and latest_query:
        conv_snapshot = [{"role": m.role, "content": m.content} for m in req.messages[-MAX_CONTEXT_MESSAGES:]]
        asyncio.create_task(_check_enquiry_and_notify(conv_snapshot))

    # ── Agent mode: auto-research ──
    research_context = ""
    if IS_AGENT:
        try:
            research_context = await do_research(latest_query)
        except Exception as e:
            logger.error(f"[chat] Research failed: {e}", exc_info=True)

    # ── Build final system prompt ──
    system_prompt = _build_system_prompt(latest_query, [{"role": m.role, "content": m.content} for m in req.messages])

    if research_context:
        system_prompt += "\n\n" + research_context
        system_prompt += (
            "\n\n## INSTRUCTIONS\n"
            "Write a comprehensive, well-structured response based on the web research above.\n"
            "- Use ONLY facts from the research data. Do NOT make up companies, statistics, or URLs.\n"
            "- Structure your answer with clear headings and sections.\n"
            "- Cite sources by including their real URLs.\n"
            "- If you found specific companies, list them with details from the pages.\n"
            "- Be thorough but factual."
        )

    logger.info(
        f"[chat] System prompt: {len(system_prompt)} chars"
        + (f", research: {len(research_context)} chars" if research_context else "")
    )

    messages = [{"role": "system", "content": system_prompt}]
    for m in req.messages[-MAX_CONTEXT_MESSAGES:]:
        messages.append({"role": m.role, "content": m.content})

    if req.stream:
        collected_response = []

        async def generate():
            try:
                async with httpx.AsyncClient(timeout=180.0) as client:
                    async with client.stream(
                        "POST",
                        f"{OLLAMA_URL}/api/chat",
                        json={
                            "model": OLLAMA_MODEL,
                            "messages": messages,
                            "stream": True,
                            "options": {"temperature": 0.3, "num_predict": 4096},
                        },
                    ) as response:
                        async for line in response.aiter_lines():
                            if line.strip():
                                # Collect assistant content for saving
                                try:
                                    chunk = json.loads(line)
                                    content = chunk.get("message", {}).get("content", "")
                                    if content:
                                        collected_response.append(content)
                                except Exception:
                                    pass
                                yield line + "\n"
            except httpx.ConnectError:
                yield (
                    '{"error": "AI model is starting up. Please try again in a moment."}\n'
                )
            except Exception as e:
                logger.error(f"Chat stream error: {e}")
                yield '{"error": "Something went wrong. Please try again."}\n'

        async def _save_after_stream():
            full_response = "".join(collected_response)
            if latest_query and full_response:
                await _save_chat_messages(session_id, latest_query, full_response)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            background=BackgroundTask(_save_after_stream),
        )
    else:
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": 0.3, "num_predict": 4096},
                    },
                )
                data = resp.json()
                # Save the user + assistant message pair
                assistant_content = data.get("message", {}).get("content", "")
                if latest_query and assistant_content:
                    asyncio.create_task(_save_chat_messages(session_id, latest_query, assistant_content))
                return data
        except httpx.ConnectError:
            raise HTTPException(503, "AI model is currently unavailable.")
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(500, "Chat failed")


# ═══════════════════  RAG ENDPOINTS  ═══════════════════


@app.post("/ingest")
async def ingest_documents(req: IngestRequest):
    global rag_engine
    if not RAG_ENABLED:
        raise HTTPException(400, "RAG is disabled. Set RAG_ENABLED=true to enable.")
    try:
        if rag_engine is None:
            rag_engine = RAGEngine(data_dir=req.directory, db_dir=RAG_DB_DIR)
        count = rag_engine.ingest(req.directory)
        return {
            "status": "success",
            "documents_ingested": count,
            "total_chunks": rag_engine.document_count(),
        }
    except FileNotFoundError:
        raise HTTPException(404, f"Directory not found: {req.directory}")
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(500, f"Ingestion failed: {str(e)}")


@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    global rag_engine
    if not RAG_ENABLED:
        raise HTTPException(400, "RAG is disabled. Set RAG_ENABLED=true to enable.")

    SUPPORTED = {".txt", ".md", ".csv", ".json", ".pdf", ".jsonl"}
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
    data_dir = RAG_DATA_DIR
    os.makedirs(data_dir, exist_ok=True)

    saved_files = []
    errors = []
    for ufile in files:
        ext = os.path.splitext(ufile.filename or "")[1].lower()
        if ext not in SUPPORTED:
            errors.append({"file": ufile.filename, "error": f"Unsupported format: {ext}"})
            continue
        try:
            file_bytes = await ufile.read()
        except Exception as e:
            errors.append({"file": ufile.filename, "error": f"Read failed: {str(e)}"})
            continue
        if len(file_bytes) > MAX_FILE_SIZE:
            errors.append({
                "file": ufile.filename,
                "error": f"File too large ({len(file_bytes) / 1024 / 1024:.1f} MB). Max: 20 MB",
            })
            continue
        if len(file_bytes) == 0:
            errors.append({"file": ufile.filename, "error": "Empty file"})
            continue

        safe_name = ufile.filename.replace("/", "_").replace("\\", "_")
        dest = os.path.join(data_dir, safe_name)
        try:
            with open(dest, "wb") as f:
                f.write(file_bytes)
            saved_files.append(safe_name)
            logger.info(f"Saved uploaded file: {safe_name} ({len(file_bytes)} bytes)")
        except Exception as e:
            errors.append({"file": ufile.filename, "error": f"Save failed: {str(e)}"})

    ingested = 0
    total_chunks = 0
    if saved_files:
        try:
            if rag_engine is None:
                rag_engine = RAGEngine(data_dir=data_dir, db_dir=RAG_DB_DIR)
            ingested = rag_engine.ingest(data_dir)
            total_chunks = rag_engine.document_count()
        except Exception as e:
            logger.error(f"Ingestion after upload failed: {e}")
            errors.append({"file": "ingestion", "error": str(e)})

    return {
        "status": "success" if saved_files else "failed",
        "files_saved": saved_files,
        "files_ingested": ingested,
        "total_chunks": total_chunks,
        "errors": errors,
    }


@app.get("/config")
async def get_config():
    return {
        "name": CHATBOT_NAME,
        "rag_enabled": RAG_ENABLED,
        "rag_documents": rag_engine.document_count() if rag_engine else 0,
        "system_prompt": _user_system_prompt,
    }


@app.put("/config")
async def update_config(req: UpdateConfigRequest):
    global _user_system_prompt, SYSTEM_PROMPT
    _user_system_prompt = req.system_prompt
    SYSTEM_PROMPT = get_system_prompt()
    logger.info(f"System prompt updated ({len(req.system_prompt)} chars)")
    return {"status": "updated", "system_prompt": _user_system_prompt}


@app.get("/knowledge")
async def get_knowledge_status():
    SUPPORTED = {".txt", ".md", ".csv", ".json", ".pdf", ".jsonl"}
    doc_files = []
    if os.path.exists(RAG_DATA_DIR):
        for fname in sorted(os.listdir(RAG_DATA_DIR)):
            fpath = os.path.join(RAG_DATA_DIR, fname)
            if os.path.isfile(fpath):
                ext = os.path.splitext(fname)[1].lower()
                if ext in SUPPORTED:
                    doc_files.append({
                        "name": fname,
                        "size": os.path.getsize(fpath),
                        "type": ext.lstrip(".").upper(),
                    })
    return {
        "rag_enabled": RAG_ENABLED,
        "total_chunks": rag_engine.document_count() if rag_engine else 0,
        "files": doc_files,
    }


@app.delete("/knowledge")
async def clear_knowledge():
    global rag_engine
    if not RAG_ENABLED:
        raise HTTPException(400, "RAG is disabled.")
    if rag_engine:
        rag_engine.clear()
    deleted = []
    if os.path.exists(RAG_DATA_DIR):
        for fname in os.listdir(RAG_DATA_DIR):
            fpath = os.path.join(RAG_DATA_DIR, fname)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                    deleted.append(fname)
                except Exception as e:
                    logger.warning(f"Could not delete {fname}: {e}")
    return {"status": "success", "files_deleted": len(deleted), "total_chunks": 0}


@app.delete("/knowledge/{filename}")
async def delete_knowledge_file(filename: str):
    global rag_engine
    if not RAG_ENABLED:
        raise HTTPException(400, "RAG is disabled.")
    fpath = os.path.join(RAG_DATA_DIR, filename)
    if not os.path.exists(fpath) or not os.path.isfile(fpath):
        raise HTTPException(404, f"File not found: {filename}")
    os.remove(fpath)
    logger.info(f"Deleted knowledge file: {filename}")
    if rag_engine:
        rag_engine.clear()
        rag_engine.ingest(RAG_DATA_DIR)
    return {
        "status": "success",
        "file_deleted": filename,
        "total_chunks": rag_engine.document_count() if rag_engine else 0,
    }


# ── Entry point ──

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
