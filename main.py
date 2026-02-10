"""
DeepRack Chatbot API — FastAPI server for AI chatbot with RAG.
Connects to an Ollama-powered LLM on a DeepRack GPU rack.
All configuration is via environment variables — no code changes needed.
"""

import os
import time
import uuid
import logging
from typing import Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag import RAGEngine

# ── Configuration (all from env vars) ──

CHATBOT_NAME = os.getenv("CHATBOT_NAME", "AI Assistant")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful AI assistant.")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"
RAG_DATA_DIR = os.getenv("RAG_DATA_DIR", "/workspace/data")
RAG_DB_DIR = os.getenv("RAG_DB_DIR", "/workspace/chromadb")
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "20"))
MAX_RAG_CHUNKS = int(os.getenv("MAX_RAG_CHUNKS", "5"))
PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chatbot-api")

# ── RAG Engine ──

rag_engine: Optional[RAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
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

    logger.info(f"Chatbot '{CHATBOT_NAME}' started — Ollama at {OLLAMA_URL}, model {OLLAMA_MODEL}")
    yield
    logger.info("Chatbot API shutting down")


# ── FastAPI App ──

app = FastAPI(
    title=f"{CHATBOT_NAME} API",
    description="AI Chatbot API powered by DeepRack",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response Models ──


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=10000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=50)
    stream: bool = True


class IngestRequest(BaseModel):
    directory: str = RAG_DATA_DIR


# ── Helpers ──


def _build_system_prompt(user_query: str) -> str:
    """Build the full system prompt with optional RAG context."""
    base_prompt = SYSTEM_PROMPT

    if rag_engine and RAG_ENABLED:
        try:
            chunks = rag_engine.search(user_query, top_k=MAX_RAG_CHUNKS)
            if chunks:
                context_text = "\n\n---\n\n".join([c["text"] for c in chunks])
                base_prompt += (
                    f"\n\n## Relevant Context from Knowledge Base\n"
                    f"Use the following information to answer the user's question. "
                    f"If the information is not relevant, rely on your general knowledge.\n\n"
                    f"{context_text}"
                )
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")

    return base_prompt


# ── Endpoints ──


@app.get("/health")
async def health():
    """Health check endpoint."""
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            models = resp.json().get("models", [])
            ollama_ok = any(m.get("name", "").startswith(OLLAMA_MODEL.split(":")[0]) for m in models)
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
    """Chat endpoint — streams responses from the LLM."""
    # Get the latest user message for RAG search
    user_messages = [m for m in req.messages if m.role == "user"]
    latest_query = user_messages[-1].content if user_messages else ""

    # Build messages with system prompt + RAG context
    system_prompt = _build_system_prompt(latest_query)
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (limited)
    for m in req.messages[-MAX_CONTEXT_MESSAGES:]:
        messages.append({"role": m.role, "content": m.content})

    if req.stream:
        async def generate():
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{OLLAMA_URL}/api/chat",
                        json={"model": OLLAMA_MODEL, "messages": messages, "stream": True},
                    ) as response:
                        async for line in response.aiter_lines():
                            if line.strip():
                                yield line + "\n"
            except httpx.ConnectError:
                yield '{"error": "AI model is starting up. Please try again in a moment."}\n'
            except Exception as e:
                logger.error(f"Chat stream error: {e}")
                yield '{"error": "Something went wrong. Please try again."}\n'

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
                )
                return resp.json()
        except httpx.ConnectError:
            raise HTTPException(503, "AI model is currently unavailable.")
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(500, "Chat failed")


@app.post("/ingest")
async def ingest_documents(req: IngestRequest):
    """Trigger document ingestion for RAG."""
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


@app.get("/config")
async def get_config():
    """Return public chatbot configuration (for the frontend)."""
    return {
        "name": CHATBOT_NAME,
        "rag_enabled": RAG_ENABLED,
        "rag_documents": rag_engine.document_count() if rag_engine else 0,
    }


# ── Entry point ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
