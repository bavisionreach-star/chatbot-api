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
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")   # "gemini" or "ollama"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
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
    "owner_email": "",
    "last_refresh": 0,
}

# Track sessions that already have a ticket — avoid re-classifying on every message
_session_tickets: dict[str, str] = {}  # session_id -> ticket_number

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
    + "\n\n"
    "## GREETING INSTRUCTIONS\n"
    "At the very beginning of every conversation (when the first user message arrives), "
    "greet the user warmly and ask for their name. Example: "
    f"'Hi there! Welcome! I'm {CHATBOT_NAME}"
    + (f", {ORG_NAME}'s AI assistant" if ORG_NAME else "")
    + ". Before we get started, may I know your name?' "
    "Once they provide their name, use it naturally throughout the conversation. "
    "If they skip or decline, continue without pressing — never insist."
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
    # For Gemini, use a concise prompt with the no-code restriction embedded
    # right after the identity/role.
    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
        return (
            _identity_block.strip() + "\n\n"
            + _user_system_prompt.strip() + "\n\n"
            "CRITICAL RULES:\n"
            "- You are ONLY a support assistant. You handle enquiries, collect information, and pass requests to the team.\n"
            "- NEVER write code, provide code snippets, give technical implementations, or debug errors. This is strictly forbidden.\n"
            "- If someone asks you to write code or build something, say: \"I'm a support assistant and I can't write code. "
            "Let me collect your requirements and pass them to our technical team.\"\n"
            "- Collect the user's name, email, and details of their request, then confirm you will forward it to the team.\n"
        )
    return DEFAULT_BAVISION_PROMPT.strip() + "\n\n" + _user_system_prompt.strip()


SYSTEM_PROMPT = get_system_prompt()


# ═══════════════════  LLM ABSTRACTION LAYER  ═══════════════════
# Wraps Ollama and Gemini APIs behind a unified interface.

def _use_gemini():
    """Check if Gemini provider should be used."""
    return LLM_PROVIDER == "gemini" and GEMINI_API_KEY


def _extract_llm_api_key() -> str:
    """Extract the LLM API key from the OLLAMA_URL (metered proxy URL)."""
    # OLLAMA_URL looks like https://api.deeprack.in/api/llm/{api_key}
    parts = OLLAMA_URL.rstrip("/").split("/api/llm/")
    return parts[1] if len(parts) == 2 else ""


async def _report_gemini_usage(input_tokens: int, output_tokens: int):
    """Report Gemini token usage to the backend metered proxy for billing."""
    api_key = _extract_llm_api_key()
    if not api_key or (input_tokens == 0 and output_tokens == 0):
        return
    try:
        base_url = OLLAMA_URL.split("/api/llm/")[0]
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{base_url}/api/llm/{api_key}/api/usage",
                json={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "provider": "gemini",
                    "model": GEMINI_MODEL,
                },
            )
    except Exception as e:
        logger.warning(f"[billing] Failed to report Gemini usage: {e}")


async def _llm_generate(prompt: str, temperature: float = 0, max_tokens: int = 10) -> str:
    """Non-streaming text generation. Returns the generated text."""
    if _use_gemini():
        messages = [{"role": "user", "content": prompt}]
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                GEMINI_API_URL,
                headers={
                    "Authorization": f"Bearer {GEMINI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GEMINI_MODEL,
                    "messages": messages,
                    "stream": False,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            data = resp.json()
            # Gemini error responses return a JSON array, not object
            if isinstance(data, list) or resp.status_code >= 400:
                err = data[0] if isinstance(data, list) else data
                err_msg = err.get("error", {}).get("message", "") if isinstance(err, dict) else str(err)
                logger.warning(f"[gemini] API error {resp.status_code}: {err_msg[:200]}")
                return ""
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            asyncio.create_task(_report_gemini_usage(
                usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
            ))
            return (content or "").strip()
    else:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            )
            return resp.json().get("response", "").strip()


async def _llm_chat_nonstream(messages: list[dict], temperature: float = 0.3, max_tokens: int = 4096) -> dict:
    """Non-streaming chat. Returns Ollama-format response dict."""
    if _use_gemini():
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                GEMINI_API_URL,
                headers={
                    "Authorization": f"Bearer {GEMINI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GEMINI_MODEL,
                    "messages": messages,
                    "stream": False,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            data = resp.json()
            # Gemini error responses return a JSON array, not object
            if isinstance(data, list) or resp.status_code >= 400:
                err = data[0] if isinstance(data, list) else data
                err_msg = err.get("error", {}).get("message", "") if isinstance(err, dict) else str(err)
                logger.warning(f"[gemini] API error {resp.status_code}: {err_msg[:200]}")
                if resp.status_code == 429:
                    raise Exception("AI model is temporarily busy. Please try again in a moment.")
                raise Exception("AI service encountered an error. Please try again.")
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            asyncio.create_task(_report_gemini_usage(
                usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
            ))
            # Convert to Ollama format for UI compatibility
            return {
                "message": {"role": "assistant", "content": content or ""},
                "done": True,
            }
    else:
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            )
            return resp.json()


async def _llm_chat_stream(messages: list[dict], temperature: float = 0.3, max_tokens: int = 4096):
    """Streaming chat. Yields Ollama-format NDJSON lines."""
    if _use_gemini():
        stream_usage = {"input": 0, "output": 0}
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream(
                "POST",
                GEMINI_API_URL,
                headers={
                    "Authorization": f"Bearer {GEMINI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GEMINI_MODEL,
                    "messages": messages,
                    "stream": True,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            ) as response:
                # Check for error status before streaming
                if response.status_code >= 400:
                    body = await response.aread()
                    logger.warning(f"[gemini] Stream error {response.status_code}: {body[:300]}")
                    if response.status_code == 429:
                        err_chunk = json.dumps({"message": {"role": "assistant", "content": "I'm temporarily busy due to high demand. Please try again in a moment."}, "done": True})
                    else:
                        err_chunk = json.dumps({"message": {"role": "assistant", "content": "I encountered an error. Please try again."}, "done": True})
                    yield err_chunk
                    return
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        chunk = json.loads(line)
                        # Capture usage from the final chunk if present
                        if "usage" in chunk:
                            stream_usage["input"] = chunk["usage"].get("prompt_tokens", 0)
                            stream_usage["output"] = chunk["usage"].get("completion_tokens", 0)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content is None:
                            continue
                        finish = chunk.get("choices", [{}])[0].get("finish_reason")
                        ollama_chunk = {
                            "message": {"role": "assistant", "content": content},
                            "done": finish == "stop",
                        }
                        yield json.dumps(ollama_chunk)
                    except Exception:
                        pass
        # Ensure a final done=true
        yield json.dumps({"message": {"role": "assistant", "content": ""}, "done": True})
        # Report usage for billing
        asyncio.create_task(_report_gemini_usage(stream_usage["input"], stream_usage["output"]))
    else:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            ) as response:
                async for line in response.aiter_lines():
                    if line.strip():
                        yield line


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
    provider_info = f"Gemini ({GEMINI_MODEL})" if _use_gemini() else f"Ollama at {OLLAMA_URL}, model {OLLAMA_MODEL}"
    logger.info(
        f"Chatbot '{CHATBOT_NAME}' started — "
        f"mode={mode}, LLM: {provider_info}"
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
            "1. Answer general questions about the company, products, or services using your knowledge. "
            "Do NOT write code, provide technical implementations, or debug errors — that is outside your scope.\n"
            "2. When the topic requires human follow-up (pricing quotes, complaints, custom requests, "
            "demo scheduling, partnership proposals, or anything you cannot fully resolve), "
            "naturally ask for the visitor\'s **name** (if not already known), **email address**, and **company name** (if relevant).\n"
            "3. Be conversational — don\'t dump all questions at once. Ask naturally as the conversation flows.\n"
            "4. Once you have enough details, proactively tell the user something like:\n"
            "   \'I\'ve summarized our conversation and will be notifying the respective team regarding this. "
            "Is there anything else you\'d like me to include or let them know?\'\n"
            "5. Wait for the user to confirm or add anything. After their confirmation, say something like:\n"
            "   \'Great, I\'ll make sure the team is notified with all the details. "
            "Someone will reach out to you shortly!\'\n"
            "6. Do NOT promise to send an email yourself — the notification system handles that automatically in the background. "
            "Use natural language like \'notifying the team\' or \'passing this along\'.\n"
            "7. If the user mentions additional details after you have offered to notify, include those too "
            "and confirm again before concluding.\n"
        )
    else:
        # No email notifications — still collect requests but don't promise notifications
        base += (
            "\n\n## HANDLING REQUESTS BEYOND YOUR SCOPE\n"
            "If a user makes a request that requires human assistance or falls outside your capabilities, "
            "acknowledge their request, let them know you've taken note of it, and reassure them that "
            "someone from the team will follow up. Collect their name and email if appropriate.\n"
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

    # Scope restriction at the end so it takes highest priority
    base += (
        "\n\n## CRITICAL BEHAVIOURAL RULES (MUST FOLLOW)\n"
        "- You are a SUPPORT ASSISTANT only. You MUST strictly follow the role described at the top of this prompt.\n"
        "- NEVER write code, provide code snippets, give implementation details, debug errors, or act as a developer/consultant.\n"
        "- If a user asks you to write code, build something, or provide technical implementations, "
        "politely decline and explain that you are a support assistant. Offer to collect their request "
        "and pass it to the appropriate technical team.\n"
        "- Keep responses concise, helpful, and focused on support/enquiry handling.\n"
    )

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
        "model": GEMINI_MODEL if _use_gemini() else OLLAMA_MODEL,
        "llm_provider": LLM_PROVIDER,
        "rag_enabled": RAG_ENABLED,
        "rag_documents": rag_engine.document_count() if rag_engine else 0,
    }


# ── Cached default questions (fetched from backend DB) ──
_suggestions_cache = {"questions": [], "last_refresh": 0}


@app.get("/suggestions")
async def get_suggestions():
    """Return starter questions for the chatbot UI. Fetched from backend DB, cached 5 min."""
    now = time.time()
    if CHATBOT_IDENTIFIER and now - _suggestions_cache["last_refresh"] > 300:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{BACKEND_INTERNAL_URL}/api/internal/chatbot-suggestions/{CHATBOT_IDENTIFIER}",
                    headers={"X-Internal-Secret": INTERNAL_API_SECRET},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    _suggestions_cache["questions"] = data.get("default_questions", [])
                    _suggestions_cache["last_refresh"] = now
        except Exception as e:
            logger.warning(f"Failed to fetch suggestions: {e}")

    return {"suggestions": _suggestions_cache["questions"]}


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
                    _notification_config["owner_email"] = cfg.get("owner_email", "")
                    _notification_config["ticket_prefix"] = cfg.get("ticket_prefix", "")
                    _notification_config["last_refresh"] = now
        except Exception as e:
            logger.warning(f"[enquiry] Config refresh failed: {e}")


async def _fetch_ticket_status(session_id: str = "", ticket_number: str = "") -> list[dict]:
    """Query the backend for service tickets. Supports lookup by session_id or ticket_number."""
    if not CHATBOT_IDENTIFIER or not BACKEND_INTERNAL_URL:
        return []
    try:
        params = {}
        if ticket_number:
            params["ticket_number"] = ticket_number
        elif session_id:
            params["session_id"] = session_id
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{BACKEND_INTERNAL_URL}/api/internal/chatbot-ticket-status/{CHATBOT_IDENTIFIER}",
                headers={"X-Internal-Secret": INTERNAL_API_SECRET},
                params=params,
            )
            if resp.status_code == 200:
                return resp.json().get("tickets", [])
    except Exception as e:
        logger.warning(f"[tickets] Failed to fetch ticket status: {e}")
    return []


_TICKET_NUMBER_RE = re.compile(r"[A-Za-z0-9]{1,10}-\d{1,6}", re.IGNORECASE)


_TICKET_QUERY_KEYWORDS = re.compile(
    r"(ticket|status|follow[- ]?up|request|enquiry|inquiry|update|progress|[A-Za-z0-9]{1,10}-\d{4,})",
    re.IGNORECASE,
)


async def _check_enquiry_and_notify(conversation: list[dict], session_id: str = "") -> dict:
    """Analyse the full conversation for enquiry + contact details, then notify owner.
    Uses service tickets to prevent duplicate emails. Returns ticket info so the caller can inform the user."""
    await _refresh_notification_config()

    if not _notification_config["enabled"] or not CHATBOT_IDENTIFIER or not _notification_config["notify_when"]:
        return {"notified": False, "reason": "disabled"}

    try:
        # Require at least 3 user messages before checking
        user_msg_count = sum(1 for m in conversation if m.get("role") == "user")
        if user_msg_count < 3:
            return {"notified": False, "reason": "not_enough_conversation"}

        # If this session already has a ticket, check for genuine follow-up info
        has_existing_ticket = session_id and session_id in _session_tickets
        existing_ticket_number = _session_tickets.get(session_id, "") if has_existing_ticket else ""

        notify_when = _notification_config["notify_when"]

        # Build conversation transcript for the LLM
        transcript_lines = []
        for msg in conversation[-20:]:
            role = msg.get("role", "user")
            if role in ("user", "assistant"):
                transcript_lines.append(f"{role.upper()}: {msg.get('content', '')}")
        transcript = "\n".join(transcript_lines)

        if has_existing_ticket:
            # ── Follow-up check: does the latest message contain genuinely NEW info? ──
            last_user_msg = ""
            for msg in reversed(conversation):
                if msg.get("role") == "user":
                    last_user_msg = msg.get("content", "")
                    break

            followup_prompt = (
                "You are analysing a chatbot conversation. A service ticket has already been created "
                f"(ticket {existing_ticket_number}) for this customer's enquiry.\n\n"
                f"The customer's LATEST message is:\n\"{last_user_msg}\"\n\n"
                "Does this latest message contain SUBSTANTIAL NEW information that should be forwarded "
                "to the team? Examples of NEW info: additional requirements, changed specifications, "
                "new contact details, urgent updates, corrections.\n\n"
                "Examples that are NOT new info: 'thanks', 'ok', 'great', 'sure', 'no problem', "
                "'that's all', 'bye', general pleasantries, or questions to the assistant.\n\n"
                "Reply with ONLY 'YES' or 'NO'."
            )
            answer = (await _llm_generate(followup_prompt, temperature=0, max_tokens=10)).upper()
            logger.info(f"[enquiry] Follow-up check for {existing_ticket_number}: {answer!r}")

            if "YES" not in answer:
                return {"notified": False, "reason": "no_new_info", "ticket_number": existing_ticket_number}

            # Has new info — send as follow-up
            logger.info(f"[enquiry] Sending follow-up for ticket {existing_ticket_number}")

            # Extract contact details
            email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', transcript)
            customer_email = email_match.group() if email_match else ""
            name_match = re.search(r"(?:I'm|I am|name is|Name:|name:|My name is)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)", transcript)
            customer_name = name_match.group(1) if name_match else ""
            phone_match = re.search(r'[\+]?[\d\s\-]{7,15}', transcript)
            customer_phone = phone_match.group().strip() if phone_match else ""

            bot_name = CHATBOT_NAME or "Your AI Assistant"
            user_messages = [m.get("content", "") for m in conversation if m.get("role") == "user"]
            followup_body = (
                f"Hi,\n\n"
                f"The customer provided additional information via {bot_name}.\n\n"
                f"Customer: {customer_name or 'Unknown'}\n"
                f"Email: {customer_email or 'Not provided'}\n\n"
                f"New information:\n  {user_messages[-1] if user_messages else ''}\n\n"
                f"Yours faithfully,\n{bot_name}"
            )

            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{BACKEND_INTERNAL_URL}/api/internal/chatbot-enquiry-email",
                    json={
                        "chatbot_id": CHATBOT_ID,
                        "base_name": CHATBOT_BASE_NAME,
                        "email_body": followup_body,
                        "customer_name": customer_name,
                        "customer_email": customer_email,
                        "customer_phone": customer_phone,
                        "bot_name": bot_name,
                        "session_id": session_id,
                        "is_followup": True,
                        "secret": INTERNAL_API_SECRET,
                    },
                )
                result = resp.json()
                ticket_num = result.get("ticket_number", existing_ticket_number)
                if result.get("sent"):
                    logger.info(f"[enquiry] Follow-up email sent for {ticket_num}")
                    return {"notified": True, "reason": "followup_sent", "ticket_number": ticket_num}
                else:
                    logger.warning(f"[enquiry] Follow-up not sent: {result.get('reason')}")
                    return {"notified": False, "reason": result.get("reason", "backend_rejected"), "ticket_number": ticket_num}

        # ── First-time classification: is this an enquiry? ──
        classify_prompt = (
            "You are classifying chatbot conversations. The chatbot owner wants to be notified when: "
            f"{notify_when}\n\n"
            f"CONVERSATION:\n{transcript}\n\n"
            "Answer YES ONLY if ALL of these conditions are met:\n"
            "1. The conversation matches the notification criteria above\n"
            "2. The customer has shared their contact details (name AND email or phone number)\n"
            "3. The chatbot has gathered enough information about what the customer needs\n\n"
            "If the customer has NOT yet shared their name and contact info, answer NO.\n"
            "Reply with ONLY 'YES' or 'NO'."
        )

        answer = (await _llm_generate(classify_prompt, temperature=0, max_tokens=10)).upper()
        logger.info(f"[enquiry] Classification answer: {answer!r}")

        if "YES" not in answer:
            return {"notified": False, "reason": "not_enquiry"}

        logger.info(f"[enquiry] Detected enquiry for chatbot {CHATBOT_IDENTIFIER}, sending notification")

        # Extract customer details from conversation
        bot_name = CHATBOT_NAME or "Your AI Assistant"
        email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', transcript)
        customer_email = email_match.group() if email_match else ""
        name_match = re.search(r"(?:I'm|I am|name is|Name:|name:|My name is)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)", transcript)
        customer_name = name_match.group(1) if name_match else ""
        phone_match = re.search(r'[\+]?[\d\s\-]{7,15}', transcript)
        customer_phone = phone_match.group().strip() if phone_match else ""

        user_messages = [m.get("content", "") for m in conversation if m.get("role") == "user"]
        user_requirement = " | ".join(user_messages[-3:])

        email_body = (
            f"Hi,\n\n"
            f"A new enquiry was received via {bot_name}.\n\n"
            f"Customer Details:\n"
            f"  • Name: {customer_name or 'Not provided'}\n"
            f"  • Email: {customer_email or 'Not provided'}\n"
            f"  • Phone: {customer_phone or 'Not provided'}\n\n"
            f"What they're looking for:\n"
            f"  {user_requirement}\n\n"
            f"You can view the full conversation on the DeepRack platform.\n\n"
            f"Yours faithfully,\n{bot_name}"
        )

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{BACKEND_INTERNAL_URL}/api/internal/chatbot-enquiry-email",
                json={
                    "chatbot_id": CHATBOT_ID,
                    "base_name": CHATBOT_BASE_NAME,
                    "email_body": email_body,
                    "customer_name": customer_name,
                    "customer_email": customer_email,
                    "customer_phone": customer_phone,
                    "bot_name": bot_name,
                    "session_id": session_id,
                    "secret": INTERNAL_API_SECRET,
                },
            )
            result = resp.json()
            ticket_number = result.get("ticket_number", "")

            if result.get("sent"):
                # Cache the ticket number for this session
                if session_id and ticket_number:
                    _session_tickets[session_id] = ticket_number
                logger.info(f"[enquiry] Email sent, ticket {ticket_number}")
                return {"notified": True, "reason": "sent", "ticket_number": ticket_number}
            elif result.get("reason") == "ticket_exists":
                # Backend says ticket already exists — cache it locally too
                ticket_number = result.get("ticket_number", "")
                if session_id and ticket_number:
                    _session_tickets[session_id] = ticket_number
                return {"notified": False, "reason": "ticket_exists", "ticket_number": ticket_number}
            else:
                logger.warning(f"[enquiry] Email not sent: {result.get('reason')}")
                return {"notified": False, "reason": result.get("reason", "backend_rejected")}
    except Exception as e:
        logger.warning(f"[enquiry] Notification check failed: {e}")
        return {"notified": False, "reason": "error"}


def _build_notification_followup(result: dict) -> str:
    """Build a short follow-up message based on email notification result."""
    owner_email = _notification_config.get("owner_email", "")
    bot_name = CHATBOT_NAME or "Your AI Assistant"
    org = ORG_NAME or "the team"
    ticket_number = result.get("ticket_number", "")

    if result.get("notified"):
        reason = result.get("reason", "sent")
        if reason == "followup_sent" and ticket_number:
            return (
                f"📩 Your additional details have been forwarded to {org} "
                f"under ticket **{ticket_number}**. The team will review the update shortly!"
            )
        elif ticket_number:
            return (
                f"✅ Your service ticket **{ticket_number}** has been created! "
                f"I've notified {org} — someone from the team will get back to you shortly!"
            )
        else:
            return (
                f"✅ I've notified {org} about your enquiry — "
                f"someone from the team will get back to you shortly!"
            )
    else:
        reason = result.get("reason", "")
        if reason == "ticket_exists" and ticket_number:
            return (
                f"Your enquiry is already being tracked under ticket **{ticket_number}**. "
                f"Someone from {org} will reach out to you shortly!"
            )
        elif reason == "no_new_info" and ticket_number:
            # User said something like "thanks" — no need to show anything
            return ""
        # Notification failed — give the user a direct way to reach out
        if owner_email:
            return (
                f"⚠️ I wasn't able to reach {org} right now, "
                f"but you can contact them directly at **{owner_email}**."
            )
        else:
            return (
                f"⚠️ I wasn't able to notify {org} at the moment. "
                f"Please try again later or reach out to them directly."
            )


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

    # ── Prepare enquiry snapshot (will be checked after response) ──
    conv_snapshot = None
    if CHATBOT_IDENTIFIER and latest_query:
        conv_snapshot = [{"role": m.role, "content": m.content} for m in req.messages[-MAX_CONTEXT_MESSAGES:]]

    # ── Agent mode: auto-research ──
    research_context = ""
    if IS_AGENT:
        try:
            research_context = await do_research(latest_query)
        except Exception as e:
            logger.error(f"[chat] Research failed: {e}", exc_info=True)

    # ── Build final system prompt ──
    system_prompt = _build_system_prompt(latest_query, [{"role": m.role, "content": m.content} for m in req.messages])

    # ── Inject ticket status if user asks about their request/ticket ──
    if _TICKET_QUERY_KEYWORDS.search(latest_query):
        try:
            # Extract explicit ticket number (BV-XXXX) from the query
            tn_match = _TICKET_NUMBER_RE.search(latest_query)
            explicit_ticket = tn_match.group().upper() if tn_match else ""
            tickets = await _fetch_ticket_status(
                session_id=session_id,
                ticket_number=explicit_ticket,
            )
            if tickets:
                ticket_info = "\n".join(
                    f"- Ticket **{t['ticket_number']}**: status = {t['status']}, "
                    f"customer = \"{t.get('customer_name', 'N/A')}\", "
                    f"created = {t['created_at'][:10] if t.get('created_at') else 'N/A'}, "
                    f"updates sent = {t.get('email_count', 1)}"
                    for t in tickets
                )
                system_prompt += (
                    "\n\n## SERVICE TICKET INFO (HIGHEST PRIORITY)\n"
                    "The user is asking about a specific service ticket. "
                    "You MUST answer their question about the ticket IMMEDIATELY — "
                    "do NOT ask for their name or details first. "
                    "Here are the matching tickets:\n"
                    + ticket_info + "\n\n"
                    "Respond with the ticket status right away. Be conversational and reassuring. "
                    "If the status is 'open', let them know the team has received their request and is working on it. "
                    "If 'closed', let them know it has been resolved. "
                    "Always mention the ticket number so they can reference it."
                )
            elif session_id in _session_tickets:
                # We have a cached ticket number but couldn't fetch details
                ticket_num = _session_tickets[session_id]
                system_prompt += (
                    f"\n\n## SERVICE TICKET INFO\n"
                    f"This user has an open ticket: **{ticket_num}**. "
                    f"Let them know their request is being tracked under this ticket number "
                    f"and the team will follow up."
                )
        except Exception as e:
            logger.warning(f"[tickets] Failed to inject ticket context: {e}")

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
                async for line in _llm_chat_stream(messages, temperature=0.3, max_tokens=4096):
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

            # ── Post-response: enquiry detection + email notification ──
            if conv_snapshot:
                try:
                    result = await _check_enquiry_and_notify(conv_snapshot, session_id=session_id)
                    if result.get("reason") not in ("not_enquiry", "disabled", "not_enough_conversation", "no_new_info"):
                        followup = _build_notification_followup(result)
                        if followup:
                            followup_chunk = json.dumps({
                                "message": {"role": "assistant", "content": "\n\n" + followup},
                                "done": False,
                            })
                            yield followup_chunk + "\n"
                            collected_response.append("\n\n" + followup)
                except Exception as e:
                    logger.warning(f"[enquiry] Post-stream notification failed: {e}")

        async def _save_after_stream():
            full_response = "".join(collected_response)
            if latest_query and full_response:
                await _save_chat_messages(session_id, latest_query, full_response)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
            background=BackgroundTask(_save_after_stream),
        )
    else:
        try:
            data = await _llm_chat_nonstream(messages, temperature=0.3, max_tokens=4096)
            assistant_content = data.get("message", {}).get("content", "")

            # Post-response: enquiry detection + email notification
            if conv_snapshot:
                try:
                    result = await _check_enquiry_and_notify(conv_snapshot, session_id=session_id)
                    if result.get("reason") not in ("not_enquiry", "disabled", "not_enough_conversation", "no_new_info"):
                        followup = _build_notification_followup(result)
                        if followup:
                            assistant_content += "\n\n" + followup
                            data["message"]["content"] = assistant_content
                except Exception as e:
                    logger.warning(f"[enquiry] Non-stream notification failed: {e}")

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
