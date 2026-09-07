# Agentic PDF Research Assistant

A small agentic RAG (Retrieval-Augmented Generation) app that lets you upload a PDF and ask it questions in a chat interface. Built to learn LangChain, LangGraph, FAISS, embeddings, and basic agentic workflows from the ground up — the graph is hand-built with explicit nodes, state, and conditional edges rather than relying on high-level agent abstractions.

**Live demo:** _add your Streamlit Cloud link here once deployed_

## Pipeline

```
PDF → Chunking → Embeddings → FAISS → Classify → Retrieve/Summarize → Analyze → Generate
```

## How it works

The app is a **LangGraph state machine**, not a fixed linear pipeline. Every question flows through these nodes:

1. **classify** — decides if the question is `broad` (asks about the whole document — "what is this paper about?") or `specific` (asks about one concept/fact). Uses recent chat history so follow-ups like "why is *it* useful?" are classified correctly.
2. **retrieve** *(specific path only)* — rewrites the question into a standalone query using chat history (so pronouns like "it"/"that" resolve correctly), then runs a similarity search against the FAISS index to pull the top-k relevant chunks.
3. **check_relevance** — asks the LLM to judge whether the retrieved chunks can actually answer the question, before generating anything. This is what prevents the assistant from confidently hallucinating an answer from irrelevant chunks.
4. **generate** *(if relevant)* — answers strictly from the retrieved context.
5. **no_context_response** *(if not relevant)* — returns a clear "couldn't find relevant information" fallback instead of guessing.
6. **summarize** *(broad path)* — instead of similarity search, feeds a larger slice of the document directly to the LLM for an overview-style answer (similarity search doesn't work well for "summarize the whole thing" style questions).

```
classify ──broad──► summarize ──► END
   │
   └──specific──► retrieve ──► check_relevance ──relevant──► generate ──► END
                                       │
                                       └──not relevant──► no_context_response ──► END
```

## Project structure

```
agentic-pdf-assistant/
├── app.py            # Streamlit UI: upload, chat, session state
├── graph.py           # AgentState, all LangGraph nodes, build_agent()
├── ingestion.py         # PDF loading, chunking, FAISS index building
├── requirements.txt
├── .env
└── .gitignore
```

## Tech stack

- **LangChain** — document loading, text splitting, prompt templates
- **LangGraph** — the agentic state machine (nodes, state, conditional edges)
- **FAISS** — local vector similarity search
- **HuggingFace sentence-transformers** (`all-MiniLM-L6-v2`) — embeddings, runs locally, no API key needed
- **Groq** (`openai/gpt-oss-20b`) — fast, free-tier LLM for classification, retrieval-query rewriting, relevance checking, and generation
- **Streamlit** — chat UI

## Run it locally

1. Clone the repo and set up a virtual environment:
   ```bash
   git clone <your-repo-url>
   cd agentic-pdf-assistant
   python -m venv venv
   venv\\Scripts\\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Get a free API key from [console.groq.com](https://console.groq.com), then create a `.env` file (copy `.env.example`):
   ```bash
   cp .env.example .env
   # then edit .env and paste your real key
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

5. Upload a PDF in the sidebar and start asking questions.

## Known limitations

- **Summarization is truncated**: `summarize_node` only feeds the first ~40 chunks to the LLM to stay under Groq's free-tier token-per-minute limit, so summaries of very long documents may miss later sections. A proper fix would use map-reduce summarization (summarize chunks in groups, then summarize the summaries).
- **Single PDF at a time**: the app currently supports one uploaded document per session.
- **Relevance/classification are LLM judgment calls**: since `check_relevance` and `classify` both rely on the LLM's own judgment (rather than a fixed rule), they can occasionally misclassify ambiguous questions.

## Possible next steps

- Multi-PDF support with per-document source tagging
- Map-reduce summarization for full-document coverage
- A structured-extraction node for list-style questions (e.g. "list the authors")
- Swap FAISS for a hosted vector DB (Chroma/Pinecone) for persistence across sessions