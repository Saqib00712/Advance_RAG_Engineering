#!/usr/bin/env python
# coding: utf-8

"""
Industry-Style Agentic RAG with LangGraph

Why this is industry-style:

A simple RAG system searches only the private knowledge base.

An industry-style Agentic RAG system should do this:

Question
  ↓
Search private knowledge base first
  ↓
Grade the private evidence
  ↓
If private evidence is good → answer from KB
  ↓
If private evidence is weak → search the web
  ↓
Grade web evidence
  ↓
Generate a grounded answer with source type

This design is useful because private documents may be incomplete or outdated.
"""

# ============================================================
# 1. Install Dependencies
# ============================================================
# pip install -U langgraph langchain langchain-groq langchain-tavily langchain-huggingface langchain-pinecone pinecone-client langchain-community langchain-text-splitters beautifulsoup4 requests python-dotenv pydantic typing_extensions

# ============================================================
# 2. Configure API Keys
# ============================================================

import os
from getpass import getpass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass("Enter GROQ_API_KEY: ")

if not os.getenv("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass("Enter TAVILY_API_KEY: ")

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass("Enter PINECONE_API_KEY: ")

print("GROQ_API_KEY configured:", bool(os.getenv("GROQ_API_KEY")))
print("TAVILY_API_KEY configured:", bool(os.getenv("TAVILY_API_KEY")))
print("PINECONE_API_KEY configured:", bool(os.getenv("PINECONE_API_KEY")))


# ============================================================
# 3. Load Real Documents
# ============================================================
# We will create our private knowledge base from a real web document:
# https://docs.langchain.com/oss/python/langgraph/agentic-rag
#
# This becomes our internal/private KB for the demo.
# The web fallback is separate. That means:
# - First source: indexed LangGraph documentation page
# - Fallback source: live web search through Tavily

from langchain_community.document_loaders import WebBaseLoader

SOURCE_URL = "https://docs.langchain.com/oss/python/langgraph/agentic-rag"

loader = WebBaseLoader(
    web_paths=(SOURCE_URL,),
    requests_kwargs={
        "headers": {
            "User-Agent": "Mozilla/5.0 Agentic-RAG-Industry-Demo"
        }
    },
)

raw_docs = loader.load()

print("Loaded documents:", len(raw_docs))
print("Source:", raw_docs[0].metadata.get("source"))
print("\nPreview:\n")
print(raw_docs[0].page_content[:1500])


# ============================================================
# 4. Split the Documents into Chunks
# ============================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    add_start_index=True,
)

chunks = splitter.split_documents(raw_docs)

print("Total chunks:", len(chunks))
print("\nFirst chunk preview:\n")
print(chunks[0].page_content[:900])


# ============================================================
# 5. Create Free Local Embeddings
# ============================================================

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

sample_vector = embeddings.embed_query("What is Agentic RAG?")
print("Embedding dimensions:", len(sample_vector))


# ============================================================
# 6. Create the Pinecone Vector Database
# ============================================================
# The embedding model used in this notebook produces 384-dimensional vectors,
# so the Pinecone index must use dimension 384.

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import time

INDEX_NAME = "industry-agentic-rag-kb"
NAMESPACE = "langgraph-agentic-rag"

# Connect to Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Create the index only if it does not already exist.
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,       # all-MiniLM-L6-v2 embedding dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

    # Wait until Pinecone reports the new index as ready.
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

print("Pinecone index ready:", INDEX_NAME)

# Upload the document chunks and create the LangChain vector store.
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME,
    namespace=NAMESPACE,
)

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "namespace": NAMESPACE,
    }
)

print("Pinecone vector database and retriever are ready.")


# ============================================================
# Load Existing Index (Optional)
# ============================================================

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

INDEX_NAME = "industry-agentic-rag-kb"
NAMESPACE = "langgraph-agentic-rag"

# Connect to Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Load existing Pinecone index
index = pc.Index(INDEX_NAME)

# Connect existing index with LangChain
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace=NAMESPACE,
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "namespace": NAMESPACE,
    }
)

print("Existing Pinecone index loaded:", INDEX_NAME)


# ============================================================
# 7. Test Private KB Retrieval
# ============================================================

test_question = "What happens if retrieved documents are not relevant in Agentic RAG?"

kb_docs = retriever.invoke(test_question)

for i, doc in enumerate(kb_docs, 1):
    print(f"\n--- KB RESULT {i} ---")
    print("Source:", doc.metadata.get("source"))
    print(doc.page_content[:700])


# ============================================================
# 8. Initialize Groq LLM
# ============================================================

from langchain_groq import ChatGroq

# Replace this model if your Groq dashboard shows a different available model.
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

print(llm.invoke("Explain Agentic RAG in one sentence.").content)


# ============================================================
# 9. Initialize Tavily Web Search
# ============================================================
# Tavily will be used only when the private knowledge base is insufficient.
# This is important because we do not want to send every private/company question
# to the public web.

from langchain_tavily import TavilySearch

web_search = TavilySearch(
    max_results=5,
    topic="general",
    include_answer=True,
    include_raw_content=False,
)

print("Tavily search tool ready.")


# ============================================================
# 10. Define Structured Decisions
# ============================================================
# We need reliable yes/no or route outputs from the LLM.
# Structured outputs help avoid parsing messy natural language.

from typing import List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class RouteDecision(BaseModel):
    route: Literal["kb", "direct"] = Field(
        description="Use kb for questions needing Agentic RAG docs; direct for greetings/simple chat."
    )


class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(
        description="good means evidence can answer the question; weak means not enough evidence."
    )


class AgentState(TypedDict):
    question: str
    current_query: str
    kb_docs: List[Document]
    web_results: str
    kb_grade: str
    web_grade: str
    answer: str
    source_used: str
    retry_count: int


# ============================================================
# 11. Node 1 — Route the Question
# ============================================================
# The router decides:
# - kb: use private knowledge base
# - direct: answer without retrieval
#
# In a real company assistant, this step may also route between many sources:
# HR docs, finance docs, product docs, SQL database, web search, etc.

router_llm = llm.with_structured_output(RouteDecision, method="json_mode")

def route_question(state: AgentState):
    question = state["question"]

    decision = router_llm.invoke(f'''
You are a router for an Agentic RAG assistant.

Route to "kb" if the user asks about:
- Agentic RAG
- LangGraph Agentic RAG workflow
- retrieval grading
- query rewriting
- RAG architecture
- retriever tools
- web fallback in RAG

Route to "direct" only for greetings, thanks, or very simple conversation.

Question:
{question}

Return your response as valid JSON.
Example:
{{"route": "kb"}}
''')

    print("[Router]", decision.route)

    return {
        "current_query": question,
        "source_used": decision.route,
    }


def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    if state["source_used"] == "kb":
        return "retrieve_kb"
    return "direct_answer"


# ============================================================
# 12. Node 2 — Retrieve from the Private KB
# ============================================================

def retrieve_kb(state: AgentState):
    query = state["current_query"]
    docs = retriever.invoke(query)

    print(f"[KB Retriever] Query: {query}")
    print(f"[KB Retriever] Retrieved: {len(docs)} chunks")

    return {"kb_docs": docs}


# ============================================================
# 13. Node 3 — Grade Private KB Evidence
# ============================================================
# This is the key step that decides whether we can answer from the private KB
# or need web fallback.

kb_grader_llm = llm.with_structured_output(EvidenceGrade, method="json_mode")

def grade_kb_evidence(state: AgentState):
    question = state["question"]

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source')}\n{doc.page_content}"
        for doc in state["kb_docs"]
    )

    grade = kb_grader_llm.invoke(f'''
You are an evidence grader.

Question:
{question}

Private KB evidence:
{context}

Can this private KB evidence answer the question?
Return "good" if it can answer.
Return "weak" if it cannot answer or is incomplete.

Return your response as valid JSON.
Example:
{{"grade": "good"}}
''')

    print("[KB Grader]", grade.grade)

    return {"kb_grade": grade.grade}


def decide_after_kb_grade(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    if state["kb_grade"] == "good":
        return "generate_from_kb"
    return "search_web"


# ============================================================
# 14. Node 4 — Tavily Web Search Fallback
# ============================================================
# This node runs only when the private KB evidence is weak.
# We search the web using the original question, or the current rewritten query
# if a rewrite already happened.

def search_web(state: AgentState):
    query = state["current_query"]

    print(f"[Tavily Search] Query: {query}")

    result = web_search.invoke({"query": query})

    # Tavily can return dict/list structures depending on package version.
    # Convert it into readable text for grading and generation.
    if isinstance(result, dict):
        answer = result.get("answer", "")
        results = result.get("results", [])
        lines = []
        if answer:
            lines.append(f"Tavily answer: {answer}")

        for item in results:
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")

        web_text = "\n\n".join(lines) if lines else str(result)
    else:
        web_text = str(result)

    print("[Tavily Search] Result characters:", len(web_text))

    return {
        "web_results": web_text,
        "source_used": "web",
    }


# ============================================================
# 15. Node 5 — Grade Web Evidence
# ============================================================

web_grader_llm = llm.with_structured_output(EvidenceGrade, method="json_mode")

def grade_web_evidence(state: AgentState):
    question = state["question"]
    web_results = state["web_results"]

    grade = web_grader_llm.invoke(f'''
You are an evidence grader.

Question:
{question}

Web search evidence:
{web_results}

Can this web evidence answer the question?
Return "good" if it can answer.
Return "weak" if it cannot answer or is incomplete.

Return your response as valid JSON.
Example:
{{"grade": "good"}}
''')

    print("[Web Grader]", grade.grade)

    return {"web_grade": grade.grade}


MAX_RETRIES = 1

def decide_after_web_grade(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "answer_insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"

    if state["retry_count"] < MAX_RETRIES:
        return "rewrite_query"

    return "answer_insufficient"


# ============================================================
# 16. Node 6 — Rewrite the Query
# ============================================================
# If both KB and web evidence are weak, we rewrite once and then try the KB again.
#
# Why back to KB first?
# Because the rewritten query may help retrieve better private evidence before
# relying on web search again.

from langchain_core.messages import HumanMessage

def rewrite_query(state: AgentState):
    question = state["question"]
    retry_count = state["retry_count"] + 1

    rewritten = llm.invoke(f'''
Rewrite the question for better retrieval and web search.

Rules:
- Preserve original intent.
- Make it specific and search-friendly.
- Do not answer.
- Return only the rewritten query.

Original question:
{question}
''').content.strip()

    print("[Rewriter]", rewritten)

    return {
        "current_query": rewritten,
        "retry_count": retry_count,
    }


# ============================================================
# 17. Node 7 — Generate from Private KB
# ============================================================

def generate_from_kb(state: AgentState):
    question = state["question"]

    context = "\n\n".join(
        f"[KB Source: {doc.metadata.get('source')}]\n{doc.page_content}"
        for doc in state["kb_docs"]
    )

    answer = llm.invoke(f'''
You are a technical instructor.

Answer using ONLY the private KB context.

Rules:
- Beginner-friendly explanation.
- Do not invent unsupported details.
- Mention that the answer is based on the private KB.
- Include source type: Private KB.

Question:
{question}

Private KB context:
{context}
''').content

    return {
        "answer": answer,
        "source_used": "private_kb",
    }


# ============================================================
# 18. Node 8 — Generate from Web Search
# ============================================================

def generate_from_web(state: AgentState):
    question = state["question"]
    web_context = state["web_results"]

    answer = llm.invoke(f'''
You are a technical instructor.

The private KB was insufficient, so web search was used.

Answer using ONLY the web search context.

Rules:
- Beginner-friendly explanation.
- Do not invent unsupported details.
- Mention that the answer is based on Tavily web search.
- Include source type: Web Search.
- If URLs are present in context, include the most useful URLs.

Question:
{question}

Web search context:
{web_context}
''').content

    return {
        "answer": answer,
        "source_used": "web_search",
    }


# ============================================================
# 19. Node 9 — Direct Answer
# ============================================================

def direct_answer(state: AgentState):
    question = state["question"]

    answer = llm.invoke(f'''
Respond briefly and naturally.

Message:
{question}
''').content

    return {
        "answer": answer,
        "source_used": "direct",
    }


# ============================================================
# 20. Node 10 — Insufficient Evidence Answer
# ============================================================

def answer_insufficient(state: AgentState):
    answer = (
        "I could not find enough reliable evidence in the private knowledge base "
        "or the web search results to answer this confidently. "
        "Please provide more specific documents or rephrase the question."
    )

    return {
        "answer": answer,
        "source_used": "insufficient_evidence",
    }


# ============================================================
# 21. Build the LangGraph Workflow
# ============================================================

from langgraph.graph import StateGraph, START, END

workflow = StateGraph(AgentState)

workflow.add_node("route_question", route_question)
workflow.add_node("retrieve_kb", retrieve_kb)
workflow.add_node("grade_kb_evidence", grade_kb_evidence)
workflow.add_node("search_web", search_web)
workflow.add_node("grade_web_evidence", grade_web_evidence)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("generate_from_kb", generate_from_kb)
workflow.add_node("generate_from_web", generate_from_web)
workflow.add_node("direct_answer", direct_answer)
workflow.add_node("answer_insufficient", answer_insufficient)

workflow.add_edge(START, "route_question")

workflow.add_conditional_edges(
    "route_question",
    route_after_router,
    {
        "retrieve_kb": "retrieve_kb",
        "direct_answer": "direct_answer",
    },
)

workflow.add_edge("retrieve_kb", "grade_kb_evidence")

workflow.add_conditional_edges(
    "grade_kb_evidence",
    decide_after_kb_grade,
    {
        "generate_from_kb": "generate_from_kb",
        "search_web": "search_web",
    },
)

workflow.add_edge("search_web", "grade_web_evidence")

workflow.add_conditional_edges(
    "grade_web_evidence",
    decide_after_web_grade,
    {
        "generate_from_web": "generate_from_web",
        "rewrite_query": "rewrite_query",
        "answer_insufficient": "answer_insufficient",
    },
)

workflow.add_edge("rewrite_query", "retrieve_kb")
workflow.add_edge("generate_from_kb", END)
workflow.add_edge("generate_from_web", END)
workflow.add_edge("direct_answer", END)
workflow.add_edge("answer_insufficient", END)

app = workflow.compile()

print("Industry-style Agentic RAG graph compiled.")


# ============================================================
# 22. Visualize the Graph
# ============================================================

try:
    from IPython.display import Image, display
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception as e:
    print("Graph visualization is optional. The graph can still run.")
    print("Reason:", e)


# ============================================================
# 23. Helper Function
# ============================================================

def ask_agent(question: str):
    initial_state: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
    }

    result = app.invoke(initial_state)

    print("\n" + "=" * 90)
    print("QUESTION:")
    print(question)
    print("\nSOURCE USED:")
    print(result["source_used"])
    print("\nFINAL ANSWER:")
    print(result["answer"])
    print("=" * 90)

    return result


# ============================================================
# 24. Demo 1 — Answer from Private KB
# ============================================================

result = ask_agent(
    "In Agentic RAG, what happens when retrieved documents are not relevant?"
)


# ============================================================
# 25. Demo 2 — Web Search Fallback
# ============================================================

result = ask_agent(
    "What is Tavily Search and why is it useful for AI agents and RAG workflows?"
)


# ============================================================
# 26. Demo 3 — Direct Answer
# ============================================================

result = ask_agent("Hello, how are you?")


# ============================================================
# 27. Demo 4 — Ask a Current / External Question
# ============================================================

result = ask_agent(
    "What is the current LangChain Tavily package used for Python web search integration?"
)


# ============================================================
# 28. Why This Is More Industry-Ready
# ============================================================
# This implementation has:
#
# 1. Private KB first — trusted source first.
# 2. Evidence grading — do not trust weak retrieval.
# 3. Web fallback — use Tavily when KB is insufficient.
# 4. Query rewriting — recover from weak search.
# 5. Retry guard — avoid infinite loops.
# 6. Source tracking — know whether answer came from KB, web or direct path.
# 7. LangGraph control flow — stateful, visible and debuggable workflow.