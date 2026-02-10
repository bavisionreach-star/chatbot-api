# 🤖 DeepRack Chatbot API

A production-ready AI chatbot backend powered by **Ollama** and **ChromaDB RAG**, deployed on DeepRack.

## Features

- 🧠 LLM-powered chat via Ollama (Llama 3.1 8B)
- 📚 RAG (Retrieval-Augmented Generation) with ChromaDB
- 📄 Supports PDF, TXT, CSV, JSON, Markdown documents
- 🔄 Streaming responses
- ⚙️ Fully configurable via environment variables

## Configuration

All configuration is done through environment variables:

| Variable | Default | Description |
|---|---|---|
| `CHATBOT_NAME` | `AI Assistant` | Display name of the chatbot |
| `SYSTEM_PROMPT` | `You are a helpful AI assistant.` | The persona/behavior instructions |
| `OLLAMA_URL` | `http://localhost:11434` | URL of the Ollama API (your GPU rack) |
| `OLLAMA_MODEL` | `llama3.1:8b` | Which model to use |
| `RAG_ENABLED` | `true` | Enable/disable RAG document search |
| `RAG_DATA_DIR` | `/workspace/data` | Directory for source documents |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check + status |
| `POST` | `/chat` | Chat with the AI (supports streaming) |
| `POST` | `/ingest` | Trigger document ingestion |
| `GET` | `/config` | Public chatbot configuration |

## Adding Knowledge (RAG)

1. Place your documents in the data directory (PDF, TXT, CSV, JSON, MD)
2. Call `POST /ingest` to process them
3. The chatbot will now use your documents to answer questions

## Deployed by [DeepRack](https://deeprack.bavision.in)
