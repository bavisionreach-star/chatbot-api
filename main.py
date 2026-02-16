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
import logging
import asyncio
import urllib.parse
from typing import Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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

DEFAULT_BAVISION_PROMPT = (
    "You are an AI assistant built by Bavision LLP. "
    "You are NOT built by Meta AI, Google, OpenAI, or any other company. "
    "You were created and deployed by Bavision LLP using their DeepRack GPU Cloud Platform. "
    "If anyone asks who made you or who built you, always say you were built by Bavision LLP. "
    "Never mention Meta, Llama, or any underlying model architecture. "
    "You are a Bavision AI product."
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

        if len(text) > 4000:
            text = text[:4000] + "\n[Content truncated]"
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


class IngestRequest(BaseModel):
    directory: str = RAG_DATA_DIR


class UpdateConfigRequest(BaseModel):
    system_prompt: str = Field(..., min_length=1, max_length=10000)


# ── Helpers ──


def _build_system_prompt(user_query: str) -> str:
    """Build the full system prompt, optionally with RAG context."""
    base = SYSTEM_PROMPT

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
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            models = resp.json().get("models", [])
            ollama_ok = any(
                m.get("name", "").startswith(OLLAMA_MODEL.split(":")[0])
                for m in models
            )
    except Exception:
        pass
    return {
        "status": "healthy" if ollama_ok else "degraded",
        "chatbot_name": CHATBOT_NAME,
        "model": OLLAMA_MODEL,
        "ollama_connected": ollama_ok,
        "rag_enabled": RAG_ENABLED,
        "rag_documents": rag_engine.document_count() if rag_engine else 0,
    }


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

    # ── Agent mode: auto-research ──
    research_context = ""
    if IS_AGENT:
        try:
            research_context = await do_research(latest_query)
        except Exception as e:
            logger.error(f"[chat] Research failed: {e}", exc_info=True)

    # ── Build final system prompt ──
    system_prompt = _build_system_prompt(latest_query)

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
                                yield line + "\n"
            except httpx.ConnectError:
                yield (
                    '{"error": "AI model is starting up. Please try again in a moment."}\n'
                )
            except Exception as e:
                logger.error(f"Chat stream error: {e}")
                yield '{"error": "Something went wrong. Please try again."}\n'

        return StreamingResponse(generate(), media_type="text/event-stream")
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
                return resp.json()
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
