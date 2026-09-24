# Advanced RAG Patterns
Five production-ready Retrieval-Augmented Generation patterns implemented with LangGraph, FastAPI, and modern LLMs — covering Self-Correcting RAG, Corrective RAG (CRAG), Graph RAG, Multimodal RAG, and Agentic RAG. Each pattern solves a specific limitation of basic RAG and includes a FastAPI web interface.

---

## What This Project Does

Basic RAG has a simple problem — it retrieves chunks and generates answers without checking if the retrieval was good or if the answer is grounded. These 5 patterns each solve that problem differently.

```
Basic RAG Problem:
Query → Retrieve → Generate → Answer (no quality checks)

These 5 Patterns:
Query → Retrieve → [Evaluate / Correct / Route / Fuse] → Generate → [Verify] → Answer
```

---

## Patterns Overview

| # | Pattern | Key Idea | Best For |
|---|---------|----------|----------|
| 1 | [Self-Correcting RAG](#pattern-1--self-correcting-rag) | Agent critiques and rewrites its own answer | Factual accuracy |
| 2 | [Corrective RAG (CRAG)](#pattern-2--corrective-rag-crag) | Grades retrieved docs and falls back to web search | Low-quality retrieval |
| 3 | [Graph RAG](#pattern-3--graph-rag) | Builds knowledge graph of entities and relationships | Connected reasoning |
| 4 | [Multimodal RAG](#pattern-4--multimodal-rag) | Retrieves and reasons over text + images together | Document + visual content |
| 5 | [Agentic RAG](#pattern-5--agentic-rag) | Agent decides which tools and retrieval strategies to use | Complex multi-step queries |

---

## Pattern 1 — Self-Correcting RAG

The agent generates an initial answer, then critiques it against the retrieved context and rewrites if the answer is not grounded — a reflection loop that improves output quality automatically.

### How It Works
```
Query
  ↓
Retrieve relevant chunks
  ↓
Generate initial answer
  ↓
Critique agent — is the answer grounded in the retrieved context?
  ├── YES → return final answer
  └── NO  → rewrite answer using critique feedback → check again
```

### Architecture
```
[Retrieve Node] → [Generate Node] → [Critique Node]
                        ↑                   |
                        └── rewrite loop ───┘
                                            |
                                     [Final Answer]
```

### Key Concepts
- **Self-reflection** — LLM evaluates its own output against source documents
- **Grounding check** — verifies every claim is supported by retrieved context
- **Rewrite loop** — critique feedback injected as context for improved generation
- **LangGraph state machine** — conditional edges route between generate and rewrite nodes

### FastAPI Endpoint
```
POST /api/run
{"topic": "What is an AI agent?"}
→ Returns: initial answer, critique, rewritten answer, iteration count
```

---

## Pattern 2 — Corrective RAG (CRAG)

Before generating, a grader evaluates whether retrieved documents are actually relevant to the query. If they score below threshold — the system automatically falls back to real-time web search instead.

### How It Works
```
Query
  ↓
Retrieve from vector store
  ↓
Relevance Grader — score each retrieved document
  ├── ALL relevant → generate answer from documents
  ├── SOME relevant → filter + supplement with web search
  └── NONE relevant → discard all → web search only → generate
```

### Architecture
```
[Retrieve Node]
      ↓
[Grade Documents Node]
      ├── relevant → [Generate Node] → Answer
      └── not relevant → [Web Search Node] → [Generate Node] → Answer
```

### Key Concepts
- **Document grading** — LLM scores each retrieved chunk for relevance (relevant / not relevant)
- **Corrective fallback** — poor retrieval automatically triggers Tavily web search
- **Selective filtering** — only relevant chunks passed to generation
- **No hallucination from bad retrieval** — irrelevant docs never reach the LLM context

---

## Pattern 3 — Graph RAG

Instead of treating documents as flat chunks, Graph RAG extracts entities and relationships and builds a knowledge graph. Queries traverse the graph to find connected information that chunk-based retrieval misses.

### How It Works
```
Documents
  ↓
Entity extraction — people, places, concepts, events
  ↓
Relationship extraction — how entities connect
  ↓
Knowledge graph built (nodes + edges)
  ↓
Query → graph traversal → retrieve connected subgraph
  ↓
Generate answer from graph context
```

### Architecture
```
[Document Ingestion]
      ↓
[Entity + Relationship Extraction] — LLM extracts structured triples
      ↓
[Knowledge Graph] — nodes (entities) + edges (relationships)
      ↓
[Graph Query] — traverse from query entity outward
      ↓
[Generate Node] → Answer with relationship-aware context
```

### Key Concepts
- **Knowledge graph construction** — structured (subject, relation, object) triples from documents
- **Graph traversal retrieval** — finding connected entities beyond keyword/vector similarity
- **Relationship-aware context** — LLM receives entity connections, not just text chunks
- **Multi-hop reasoning** — answers questions that require connecting information across documents

---

## Pattern 4 — Multimodal RAG

Extends RAG beyond text to handle images, charts, and diagrams alongside text content. Images are embedded and retrieved using vision models — the LLM receives both text and visual context together.

### How It Works
```
Documents (PDF, images, slides)
  ↓
Extract text chunks AND image regions
  ↓
Text embedding → text vector store
Image embedding → image vector store (CLIP / vision model)
  ↓
Query
  ├── Text retrieval → relevant text chunks
  └── Image retrieval → relevant images/diagrams
  ↓
Multimodal context → Vision LLM generates answer
```

### Architecture
```
[Document Loader]
      ↓
  ┌───┴───┐
  ▼       ▼
Text    Images
Index   Index
  └───┬───┘
      ↓
[Multimodal Retriever]
      ↓
[Vision LLM] → Answer with text + image context
```

### Key Concepts
- **Dual indexing** — separate vector stores for text and image embeddings
- **Vision model embeddings** — CLIP-style models for semantic image search
- **Multimodal context assembly** — combining text chunks and images into single LLM prompt
- **Visual question answering** — answering questions about charts, diagrams, and figures

---

## Pattern 5 — Agentic RAG

The most advanced pattern. An LLM agent with access to multiple retrieval tools decides autonomously which strategy to use, in what order, and whether to combine results — handling complex multi-step queries that no single retrieval strategy can answer alone.

### How It Works
```
Complex Query
      ↓
Agent reasons about which tools to use
      ↓
      ├── vector_search(query)     — semantic similarity retrieval
      ├── keyword_search(query)    — BM25 exact term matching
      ├── web_search(query)        — real-time Tavily search
      ├── graph_query(entity)      — knowledge graph traversal
      └── summarize_document(doc)  — full document summarization
      ↓
Agent combines results from multiple tools
      ↓
Generate comprehensive answer
      ↓
Agent checks if answer fully addresses query
  ├── YES → return final answer
  └── NO  → retrieve more information → try again
```

### Architecture
```
User Query
    ↓
[Agent Node] — LLM with bound tools
    ↓
Which tool(s) needed?
    ├── [Vector Search Tool]
    ├── [Keyword Search Tool]
    ├── [Web Search Tool]
    ├── [Graph Query Tool]
    └── [Summarizer Tool]
    ↓
[Combine Results Node]
    ↓
[Generate Node]
    ↓
[Self-Check Node] — is query fully answered?
    ├── YES → Final Answer
    └── NO  → back to Agent Node
```

### Key Concepts
- **Tool-calling agent** — LLM selects tools dynamically based on query needs
- **Multi-strategy retrieval** — combining vector, keyword, graph, and web search
- **Autonomous planning** — agent decides retrieval order without manual orchestration
- **Self-verification** — agent checks its own answer completeness before returning

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-orange?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-RAG-teal?style=flat-square)
![Tavily](https://img.shields.io/badge/Tavily-Web%20Search-purple?style=flat-square)

- **LangGraph** — stateful workflow orchestration for all 5 patterns
- **LangChain** — LLM integration, document loading, vector stores
- **FastAPI** — unified web API with Jinja2 templates and static files
- **Tavily** — real-time web search for CRAG fallback and Agentic RAG
- **OpenAI / Gemini** — LLM backbone for generation and grading
- **ChromaDB / FAISS** — vector stores for semantic retrieval
- **Python 3.10+** — core language

---

## Project Structure

```
Advanced-RAG-Patterns/
│
├── pattern1_self_correcting/
│   ├── app.py                    # FastAPI app
│   ├── backend.py                # Self-correcting RAG workflow
│   ├── templates/index.html
│   └── static/
│
├── pattern2_crag/
│   ├── app.py                    # FastAPI app
│   ├── backend.py                # CRAG workflow with document grader
│   ├── templates/index.html
│   └── static/
│
├── pattern3_graph_rag/
│   ├── app.py                    # FastAPI app
│   ├── backend.py                # Graph construction + traversal
│   ├── templates/index.html
│   └── static/
│
├── pattern4_multimodal/
│   ├── app.py                    # FastAPI app
│   ├── backend.py                # Multimodal retrieval pipeline
│   ├── templates/index.html
│   └── static/
│
├── pattern5_agentic/
│   ├── app.py                    # FastAPI app
│   ├── backend.py                # Agentic RAG with tool selection
│   ├── templates/index.html
│   └── static/
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Saqib00712/Advanced-RAG-Patterns.git
cd Advanced-RAG-Patterns
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LANGSMITH_API_KEY=your_langsmith_key_here    # optional
```

### 4. Run any pattern
```bash
# Pattern 1 — Self-Correcting RAG
cd pattern1_self_correcting
uvicorn app:app --reload --port 8001

# Pattern 2 — CRAG
cd pattern2_crag
uvicorn app:app --reload --port 8002

# Pattern 3 — Graph RAG
cd pattern3_graph_rag
uvicorn app:app --reload --port 8003

# Pattern 4 — Multimodal RAG
cd pattern4_multimodal
uvicorn app:app --reload --port 8004

# Pattern 5 — Agentic RAG
cd pattern5_agentic
uvicorn app:app --reload --port 8005
```

### 5. Open in browser
```
http://localhost:8001   # Self-Correcting RAG
http://localhost:8002   # CRAG
http://localhost:8003   # Graph RAG
http://localhost:8004   # Multimodal RAG
http://localhost:8005   # Agentic RAG
```

---

## API Reference

All 5 patterns share the same API interface:

### GET /
Web interface for the pattern

### GET /api/config
Returns runtime info — model name, pattern type, version. Never returns secrets.

### POST /api/run
```json
{ "topic": "Your question here" }
```
Returns pattern-specific response including retrieved context, intermediate steps, and final answer.

---

## Pattern Comparison

| Pattern | Retrieval | Self-Check | Web Search | Graph | Vision |
|---------|-----------|-----------|------------|-------|--------|
| Self-Correcting | Vector | ✅ | ❌ | ❌ | ❌ |
| CRAG | Vector + Web | ❌ | ✅ | ❌ | ❌ |
| Graph RAG | Graph | ❌ | ❌ | ✅ | ❌ |
| Multimodal | Vector + Image | ❌ | ❌ | ❌ | ✅ |
| Agentic | All strategies | ✅ | ✅ | ✅ | ❌ |

**When to use which:**
- **Self-Correcting** — when answer accuracy and grounding matter most
- **CRAG** — when your vector store may have incomplete or outdated information
- **Graph RAG** — when questions require connecting multiple facts or entities
- **Multimodal** — when documents contain charts, figures, or images
- **Agentic** — when queries are complex and no single strategy is enough

---

## Key Concepts Covered

- **LangGraph StateGraph** — defining nodes, edges, and conditional routing for each pattern
- **Document grading** — LLM scoring retrieved chunks for relevance before generation
- **Corrective fallback** — automatic web search when vector retrieval quality is low
- **Knowledge graph construction** — entity and relationship extraction from documents
- **Graph traversal retrieval** — finding connected subgraphs for multi-hop reasoning
- **Dual-modal indexing** — separate embeddings for text and images
- **Tool-calling agent** — LLM dynamically selecting retrieval strategies per query
- **Self-reflection loop** — agent critiquing and rewriting its own answers

---

## Related Certifications

Built applying skills from the IBM **RAG for Generative AI Applications Specialization**, **Advanced RAG with Vector Databases and Retrievers**, and **Building AI Agents and Agentic Workflows Specialization** on Coursera.

[![IBM Badge](https://img.shields.io/badge/IBM-RAG%20Specialization-blue?style=flat-square)](https://www.credly.com/users/muhammad-saqib.361f9b8c)
[![IBM Badge](https://img.shields.io/badge/IBM-AI%20Agents%20Specialization-blue?style=flat-square)](https://www.credly.com/users/muhammad-saqib.361f9b8c)
[![IBM Badge](https://img.shields.io/badge/IBM-Advanced%20RAG-blue?style=flat-square)](https://www.credly.com/users/muhammad-saqib.361f9b8c)

---

## Author

**Muhammad Saqib**
- GitHub: [@Saqib00712](https://github.com/Saqib00712)
- LinkedIn: [muhammad-saqib](https://www.linkedin.com/in/muhammad-saqib-68b9b3374/)
- Email: saqibkhosa649@gmail.com
- Credly: [15x IBM Certified](https://www.credly.com/users/muhammad-saqib.361f9b8c)
