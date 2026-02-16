"""
DeepRack Agent API v4 — Research Agent with pipeline visualization + history.

When ENABLED_TOOLS is set, this acts as a research agent with:
  - SSE pipeline progress events for animated UI
  - SQLite database for research history
  - Step-by-step visibility into the agentic pipeline

When ENABLED_TOOLS is empty, it falls back to a normal chatbot.
"""

import os
import re
import json
import time
import uuid
import logging
import asyncio
import sqlite3
import urllib.parse
from datetime import datetime, timezone
from typing import Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
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
logger = logging.getLogger("agent-api")

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
DB_PATH = os.getenv("DB_PATH", "/workspace/agent.db")

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
    return DEFAULT_BAVISION_PROMPT.strip() + "\n\n" + _user_system_prompt.strip()


SYSTEM_PROMPT = get_system_prompt()


# ═══════════════════════  DATABASE  ═══════════════════════


def _init_db():
    """Initialize SQLite database for research history."""
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            description TEXT DEFAULT '',
            status TEXT DEFAULT 'pending',
            search_queries TEXT DEFAULT '[]',
            search_results TEXT DEFAULT '[]',
            pages_read TEXT DEFAULT '[]',
            llm_response TEXT DEFAULT '',
            sources TEXT DEFAULT '[]',
            pipeline_log TEXT DEFAULT '[]',
            created_at TEXT NOT NULL,
            completed_at TEXT,
            duration_seconds REAL,
            error TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


def _db():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _research_to_dict(row) -> dict:
    """Convert a database row to a dict."""
    d = dict(row)
    for key in ("search_queries", "search_results", "pages_read", "sources", "pipeline_log"):
        if d.get(key):
            try:
                d[key] = json.loads(d[key])
            except Exception:
                d[key] = []
    return d


# ═══════════════════  TOOL IMPLEMENTATIONS  ═══════════════════


async def web_search(query: str, max_results: int = 8) -> list[dict]:
    """Search the web using DuckDuckGo HTML."""
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
            snippet = re.sub(r"<[^>]+>", "", snippets[i] if i < len(snippets) else "").strip()
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

        for tag in ["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe"]:
            html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

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

        for tag in ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "br", "hr"]:
            main = re.sub(rf"</?{tag}[^>]*>", "\n", main, flags=re.IGNORECASE)

        text = re.sub(r"<[^>]+>", " ", main)
        text = (
            text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            .replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
        )
        text = re.sub(r"&#?\w+;", " ", text)

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


# ═══════════════════  RESEARCH PIPELINE WITH SSE  ═══════════════════


async def run_research_pipeline(research_id: str, topic: str, description: str):
    """
    Execute the full research pipeline and store results in DB.
    This runs in the background after the SSE stream is set up.
    """
    conn = _db()
    start_time = time.time()
    pipeline_log = []
    all_results = []
    page_contents = []
    search_queries_used = []

    def _log(step: str, status: str, detail: str = "", data: dict = None):
        entry = {
            "step": step,
            "status": status,
            "detail": detail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if data:
            entry["data"] = data
        pipeline_log.append(entry)
        conn.execute(
            "UPDATE research SET pipeline_log = ?, status = ? WHERE id = ?",
            (json.dumps(pipeline_log), "running", research_id),
        )
        conn.commit()

    try:
        # ── Step 1: Understanding the research topic ──
        _log("understanding", "running", f"Analyzing research topic: {topic}")
        await asyncio.sleep(0.5)  # Brief pause for UI animation

        full_query = topic
        if description:
            full_query = f"{topic}: {description}"

        _log("understanding", "done", "Research topic analyzed successfully")

        # ── Step 2: Planning search strategy ──
        _log("planning", "running", "Preparing search queries for sub-agents...")

        queries = [full_query]
        words = full_query.lower()
        if "in " in words or "about " in words:
            queries.append(full_query + " latest data 2025")
        key_words = [
            w for w in full_query.split()
            if len(w) > 3 and w.lower() not in {
                "about", "research", "comprehensive", "detailed",
                "please", "write", "find", "the", "what", "how",
            }
        ]
        if len(key_words) >= 2:
            queries.append(" ".join(key_words[:5]) + " detailed analysis")
        if len(queries) < 3:
            queries.append(f"{topic} overview statistics facts")

        search_queries_used = queries[:3]
        _log("planning", "done", f"Prepared {len(search_queries_used)} search queries", {
            "queries": search_queries_used
        })

        conn.execute(
            "UPDATE research SET search_queries = ? WHERE id = ?",
            (json.dumps(search_queries_used), research_id),
        )
        conn.commit()

        # ── Step 3: Sub-agents searching the web ──
        _log("searching", "running", "Sub-agents are searching the web...")

        seen_urls: set[str] = set()
        for i, q in enumerate(search_queries_used):
            _log("searching", "running", f"Agent {i + 1}/{len(search_queries_used)}: Searching \"{q}\"")
            results = await web_search(q)
            for r in results:
                if r["url"] not in seen_urls:
                    all_results.append(r)
                    seen_urls.add(r["url"])
            if results:
                await asyncio.sleep(0.5)

        _log("searching", "done", f"Found {len(all_results)} search results across {len(search_queries_used)} queries", {
            "result_count": len(all_results),
            "results": [{"title": r["title"], "url": r["url"]} for r in all_results[:12]],
        })

        conn.execute(
            "UPDATE research SET search_results = ? WHERE id = ?",
            (json.dumps(all_results[:12]), research_id),
        )
        conn.commit()

        if not all_results:
            _log("searching", "error", "No search results found. Try a different topic.")
            conn.execute(
                "UPDATE research SET status = 'failed', error = 'No search results found' WHERE id = ?",
                (research_id,),
            )
            conn.commit()
            conn.close()
            return

        # ── Step 4: Agents collecting detailed data ──
        _log("collecting", "running", "Reading and extracting data from top sources...")

        if "url_reader" in ENABLED_TOOLS:
            urls_to_read = [r["url"] for r in all_results[:5]]
            tasks = [url_reader(u) for u in urls_to_read]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for u, content in zip(urls_to_read, results):
                if isinstance(content, str) and content.strip():
                    page_contents.append({"url": u, "content": content})
                    _log("collecting", "running", f"Extracted data from {u[:60]}...")

        _log("collecting", "done", f"Successfully read {len(page_contents)} pages", {
            "pages_read": len(page_contents),
            "sources": [pc["url"] for pc in page_contents],
        })

        conn.execute(
            "UPDATE research SET pages_read = ? WHERE id = ?",
            (json.dumps([{"url": pc["url"], "chars": len(pc["content"])} for pc in page_contents]), research_id),
        )
        conn.commit()

        # ── Step 5: LLM analyzing collected data ──
        _log("analyzing", "running", "AI is analyzing all collected data...")

        research_context = "## WEB RESEARCH RESULTS\n\n"
        research_context += "The following information was gathered from real web searches.\n"
        research_context += "Use ONLY this information to answer. Do NOT add facts not found below.\n\n"

        research_context += "### Search Results:\n\n"
        for i, r in enumerate(all_results[:10], 1):
            research_context += f"{i}. **{r['title']}**\n"
            research_context += f"   URL: {r['url']}\n"
            if r.get("snippet"):
                research_context += f"   {r['snippet']}\n"
            research_context += "\n"

        if page_contents:
            research_context += "\n### Detailed Page Contents:\n\n"
            for pc in page_contents:
                research_context += f"--- Content from {pc['url']} ---\n"
                research_context += pc["content"] + "\n\n"

        if len(research_context) > 20000:
            research_context = research_context[:20000] + "\n\n[Research data truncated]"

        _log("analyzing", "done", f"Processed {len(research_context)} characters of research data")

        # ── Step 6: LLM generating research report ──
        _log("generating", "running", "AI is writing the research report...")

        system_prompt = SYSTEM_PROMPT + "\n\n" + research_context
        system_prompt += (
            "\n\n## INSTRUCTIONS\n"
            "Write a comprehensive, well-structured research report based on the web research above.\n"
            "- Use ONLY facts from the research data. Do NOT make up companies, statistics, or URLs.\n"
            "- Structure with clear headings: Executive Summary, Key Findings, Detailed Analysis, Sources.\n"
            "- Cite sources by including their real URLs.\n"
            "- If you found specific companies/data, list them with details from the pages.\n"
            "- Be thorough, factual, and professional.\n"
            "- Use markdown formatting for readability."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research Topic: {topic}\n\nDescription: {description or 'Provide a comprehensive analysis'}"},
        ]

        llm_response = ""
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
                llm_response = data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            llm_response = f"Error generating report: {str(e)}"

        _log("generating", "done", f"Research report generated ({len(llm_response)} chars)")

        # ── Step 7: Finalizing ──
        _log("complete", "done", "Research ready for preview")

        elapsed = time.time() - start_time
        sources = [{"title": r["title"], "url": r["url"]} for r in all_results[:12]]

        conn.execute(
            """UPDATE research SET
                status = 'completed',
                llm_response = ?,
                sources = ?,
                pipeline_log = ?,
                completed_at = ?,
                duration_seconds = ?
            WHERE id = ?""",
            (
                llm_response,
                json.dumps(sources),
                json.dumps(pipeline_log),
                datetime.now(timezone.utc).isoformat(),
                round(elapsed, 1),
                research_id,
            ),
        )
        conn.commit()
        logger.info(f"[research] {research_id} completed in {elapsed:.1f}s")

    except Exception as e:
        logger.error(f"[research] Pipeline error: {e}", exc_info=True)
        _log("error", "error", str(e))
        conn.execute(
            "UPDATE research SET status = 'failed', error = ?, pipeline_log = ? WHERE id = ?",
            (str(e), json.dumps(pipeline_log), research_id),
        )
        conn.commit()
    finally:
        conn.close()


# ═══════════════════  RAG ENGINE  ═══════════════════

rag_engine: Optional[RAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    _init_db()

    if RAG_ENABLED:
        try:
            rag_engine = RAGEngine(data_dir=RAG_DATA_DIR, db_dir=RAG_DB_DIR)
            doc_count = rag_engine.document_count()
            logger.info(f"RAG engine initialized — {doc_count} document chunks indexed")
        except Exception as e:
            logger.warning(f"RAG engine initialization skipped: {e}")
            rag_engine = None

    mode = "agent" if IS_AGENT else "chatbot"
    logger.info(
        f"Agent '{CHATBOT_NAME}' started — "
        f"mode={mode}, Ollama={OLLAMA_URL}, model={OLLAMA_MODEL}"
        + (f", tools: {ENABLED_TOOLS}" if ENABLED_TOOLS else "")
    )
    yield
    logger.info("Agent API shutting down")


# ═══════════════════  FASTAPI APP  ═══════════════════

app = FastAPI(
    title=f"{CHATBOT_NAME} API",
    description="AI Agent API with research pipeline — powered by DeepRack",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──

class StartResearchRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    description: str = Field("", max_length=2000)


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=10000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=50)
    stream: bool = True


class UpdateConfigRequest(BaseModel):
    system_prompt: str = Field(..., min_length=1, max_length=10000)


class IngestRequest(BaseModel):
    directory: str = RAG_DATA_DIR


# ── Helpers ──

def _build_system_prompt(user_query: str) -> str:
    base = SYSTEM_PROMPT
    if rag_engine and RAG_ENABLED:
        try:
            chunks = rag_engine.search(user_query, top_k=MAX_RAG_CHUNKS)
            if chunks:
                ctx = "\n\n---\n\n".join(c["text"] for c in chunks)
                base += (
                    "\n\n## Relevant Context from Knowledge Base\n"
                    "Use the following information to answer the user's question.\n\n"
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
        "version": "4.0.0",
        "endpoints": {
            "health": "/health",
            "research": "/research",
            "research_stream": "/research/{id}/stream",
            "chat": "/chat",
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
                m.get("name", "").startswith(OLLAMA_MODEL.split(":")[0]) for m in models
            )
    except Exception:
        pass

    conn = _db()
    research_count = conn.execute("SELECT COUNT(*) FROM research").fetchone()[0]
    conn.close()

    return {
        "status": "healthy" if ollama_ok else "degraded",
        "chatbot_name": CHATBOT_NAME,
        "model": OLLAMA_MODEL,
        "ollama_connected": ollama_ok,
        "is_agent": IS_AGENT,
        "tools": ENABLED_TOOLS,
        "research_count": research_count,
    }


# ═══════════════════  RESEARCH ENDPOINTS  ═══════════════════


@app.post("/research")
async def start_research(req: StartResearchRequest):
    """Start a new research task. Returns research ID for tracking."""
    research_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()

    conn = _db()
    conn.execute(
        "INSERT INTO research (id, topic, description, status, created_at) VALUES (?, ?, ?, 'running', ?)",
        (research_id, req.topic, req.description, now),
    )
    conn.commit()
    conn.close()

    # Start the pipeline in the background
    asyncio.create_task(run_research_pipeline(research_id, req.topic, req.description))

    logger.info(f"[research] Started {research_id}: {req.topic}")
    return {"research_id": research_id, "status": "running"}


@app.get("/research/{research_id}/stream")
async def stream_research(research_id: str):
    """
    SSE stream of pipeline progress for a research task.
    The client connects to this after POST /research to watch progress.
    """
    conn = _db()
    row = conn.execute("SELECT id FROM research WHERE id = ?", (research_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Research not found")

    async def event_stream():
        last_log_len = 0
        while True:
            conn = _db()
            row = conn.execute(
                "SELECT status, pipeline_log, llm_response, sources, search_results, pages_read, duration_seconds FROM research WHERE id = ?",
                (research_id,),
            ).fetchone()
            conn.close()

            if not row:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Research not found'})}\n\n"
                break

            status = row["status"]
            try:
                log = json.loads(row["pipeline_log"] or "[]")
            except Exception:
                log = []

            # Send any new pipeline steps
            if len(log) > last_log_len:
                for entry in log[last_log_len:]:
                    yield f"data: {json.dumps({'type': 'step', **entry})}\n\n"
                last_log_len = len(log)

            if status in ("completed", "failed"):
                # Send final result
                try:
                    sources = json.loads(row["sources"] or "[]")
                except Exception:
                    sources = []
                try:
                    search_results = json.loads(row["search_results"] or "[]")
                except Exception:
                    search_results = []
                try:
                    pages_read = json.loads(row["pages_read"] or "[]")
                except Exception:
                    pages_read = []

                final = {
                    "type": "complete" if status == "completed" else "error",
                    "status": status,
                    "response": row["llm_response"] or "",
                    "sources": sources,
                    "search_results": search_results,
                    "pages_read": pages_read,
                    "duration_seconds": row["duration_seconds"],
                }
                yield f"data: {json.dumps(final)}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/research/{research_id}")
async def get_research(research_id: str):
    """Get a specific research result."""
    conn = _db()
    row = conn.execute("SELECT * FROM research WHERE id = ?", (research_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Research not found")
    return _research_to_dict(row)


@app.get("/research")
async def list_research(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List research history, newest first."""
    conn = _db()
    rows = conn.execute(
        "SELECT id, topic, description, status, created_at, completed_at, duration_seconds FROM research ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM research").fetchone()[0]
    conn.close()
    return {
        "items": [dict(r) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.delete("/research/{research_id}")
async def delete_research(research_id: str):
    """Delete a research entry."""
    conn = _db()
    row = conn.execute("SELECT id FROM research WHERE id = ?", (research_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Research not found")
    conn.execute("DELETE FROM research WHERE id = ?", (research_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "id": research_id}


# ═══════════════════  CHAT ENDPOINT (fallback for chatbot mode)  ═══════════════════


@app.post("/chat")
async def chat(req: ChatRequest):
    """Chat endpoint — standard chatbot mode (RAG-enabled)."""
    user_messages = [m for m in req.messages if m.role == "user"]
    latest_query = user_messages[-1].content if user_messages else ""
    system_prompt = _build_system_prompt(latest_query)

    messages = [{"role": "system", "content": system_prompt}]
    for m in req.messages[-MAX_CONTEXT_MESSAGES:]:
        messages.append({"role": m.role, "content": m.content})

    if req.stream:
        async def generate():
            try:
                async with httpx.AsyncClient(timeout=180.0) as client:
                    async with client.stream(
                        "POST", f"{OLLAMA_URL}/api/chat",
                        json={"model": OLLAMA_MODEL, "messages": messages, "stream": True,
                              "options": {"temperature": 0.3, "num_predict": 4096}},
                    ) as response:
                        async for line in response.aiter_lines():
                            if line.strip():
                                yield line + "\n"
            except httpx.ConnectError:
                yield '{"error": "AI model is starting up. Please try again."}\n'
            except Exception as e:
                logger.error(f"Chat stream error: {e}")
                yield '{"error": "Something went wrong."}\n'

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={"model": OLLAMA_MODEL, "messages": messages, "stream": False,
                          "options": {"temperature": 0.3, "num_predict": 4096}},
                )
                return resp.json()
        except httpx.ConnectError:
            raise HTTPException(503, "AI model is currently unavailable.")
        except Exception as e:
            raise HTTPException(500, "Chat failed")


# ═══════════════════  RAG ENDPOINTS  ═══════════════════


@app.post("/ingest")
async def ingest_documents(req: IngestRequest):
    global rag_engine
    if not RAG_ENABLED:
        raise HTTPException(400, "RAG is disabled.")
    try:
        if rag_engine is None:
            rag_engine = RAGEngine(data_dir=req.directory, db_dir=RAG_DB_DIR)
        count = rag_engine.ingest(req.directory)
        return {"status": "success", "documents_ingested": count, "total_chunks": rag_engine.document_count()}
    except FileNotFoundError:
        raise HTTPException(404, f"Directory not found: {req.directory}")
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {str(e)}")


@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    global rag_engine
    if not RAG_ENABLED:
        raise HTTPException(400, "RAG is disabled.")

    SUPPORTED = {".txt", ".md", ".csv", ".json", ".pdf", ".jsonl"}
    MAX_FILE_SIZE = 20 * 1024 * 1024
    os.makedirs(RAG_DATA_DIR, exist_ok=True)

    saved, errors = [], []
    for f in files:
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in SUPPORTED:
            errors.append({"file": f.filename, "error": f"Unsupported: {ext}"})
            continue
        data = await f.read()
        if len(data) > MAX_FILE_SIZE:
            errors.append({"file": f.filename, "error": "Too large"})
            continue
        if not data:
            errors.append({"file": f.filename, "error": "Empty"})
            continue
        dest = os.path.join(RAG_DATA_DIR, f.filename.replace("/", "_").replace("\\", "_"))
        with open(dest, "wb") as out:
            out.write(data)
        saved.append(f.filename)

    ingested = total = 0
    if saved:
        try:
            if rag_engine is None:
                rag_engine = RAGEngine(data_dir=RAG_DATA_DIR, db_dir=RAG_DB_DIR)
            ingested = rag_engine.ingest(RAG_DATA_DIR)
            total = rag_engine.document_count()
        except Exception as e:
            errors.append({"file": "ingestion", "error": str(e)})

    return {"status": "success" if saved else "failed", "files_saved": saved,
            "files_ingested": ingested, "total_chunks": total, "errors": errors}


@app.get("/config")
async def get_config():
    return {"name": CHATBOT_NAME, "rag_enabled": RAG_ENABLED,
            "rag_documents": rag_engine.document_count() if rag_engine else 0,
            "system_prompt": _user_system_prompt}


@app.put("/config")
async def update_config(req: UpdateConfigRequest):
    global _user_system_prompt, SYSTEM_PROMPT
    _user_system_prompt = req.system_prompt
    SYSTEM_PROMPT = get_system_prompt()
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
                    doc_files.append({"name": fname, "size": os.path.getsize(fpath), "type": ext.lstrip(".").upper()})
    return {"rag_enabled": RAG_ENABLED, "total_chunks": rag_engine.document_count() if rag_engine else 0, "files": doc_files}


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
                except Exception:
                    pass
    return {"status": "success", "files_deleted": len(deleted)}


@app.delete("/knowledge/{filename}")
async def delete_knowledge_file(filename: str):
    global rag_engine
    if not RAG_ENABLED:
        raise HTTPException(400, "RAG is disabled.")
    fpath = os.path.join(RAG_DATA_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(404, f"Not found: {filename}")
    os.remove(fpath)
    if rag_engine:
        rag_engine.clear()
        rag_engine.ingest(RAG_DATA_DIR)
    return {"status": "success", "file_deleted": filename}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
