"""
DocuWeb RAG - Agentic Retrieval-Augmented Generation Pipeline

- Ingests:   PDF files + arbitrary website URLs (with optional deep crawl)
- Embeds:    sentence-transformers/all-MiniLM-L6-v2 (local, free, no API key)
- Stores:    Qdrant in-memory vector store (no external server required)
- Generates: Groq llama3-70b-8192 (free-tier API key required)
- Fallback:  DuckDuckGo web search when local context is insufficient
- Metrics:   retrieval & generation evaluation panel
"""

import time
import json
import math
import requests
import streamlit as st
import PyPDF2
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from ddgs import DDGS
from groq import Groq

# Page config
st.set_page_config(
    page_title="DocuWeb RAG",
    layout="wide",
    menu_items={},          
)

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    .metric-label { color: #94a3b8; font-size: 0.78rem; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; }
    .metric-value { color: #e2e8f0; font-size: 1.8rem; font-weight: 700; margin: 0.2rem 0; }
    .metric-sub   { color: #64748b; font-size: 0.72rem; }
    .source-badge {
        display: inline-block;
        background: rgba(96, 165, 250, 0.15);
        border: 1px solid rgba(96, 165, 250, 0.35);
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.72rem;
        color: #93c5fd;
        margin-right: 4px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55rem 1.4rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Hide Streamlit deploy button and top-right toolbar */
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stToolbarActions"],
    header button[title="Deploy"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state bootstrap
DEFAULTS: Dict[str, Any] = {
    "qdrant_client": None,
    "collection_name": None,
    "chunks_count": 0,
    "eval_history": [],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

COLLECTION      = "docuweb_rag"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM       = 384
GROQ_MODEL      = "openai/gpt-oss-20b"  

# Embedding helpers
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = load_embed_model()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]


# Web crawling helpers
def get_all_urls(base_url: str) -> set:
    urls: set = set()
    try:
        resp = requests.get(base_url, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            base_netloc = urlparse(base_url).netloc
            for link in soup.find_all("a", href=True):
                full = urljoin(base_url, link["href"])
                p = urlparse(full)
                if p.netloc == base_netloc:
                    urls.add(p.scheme + "://" + p.netloc + p.path)
    except Exception as e:
        st.warning(f"Crawl error for {base_url}: {e}")
    return urls


def extract_text_from_url(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            st.warning(f"HTTP {resp.status_code} for {url}")
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        raw = soup.get_text(separator=" ")
        clean = " ".join(raw.split())
        return clean if len(clean) > 100 else None
    except Exception as e:
        st.warning(f"Error extracting {url}: {e}")
        return None


# PDF helpers
def extract_text_from_pdfs(pdf_files) -> List[Dict[str, str]]:
    docs = []
    for f in pdf_files:
        try:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
            docs.append({"content": text, "filename": f.name})
        except Exception as e:
            st.error(f"PDF error ({f.name}): {e}")
    return docs


# Process & index
def process_and_index(pdf_files, site_links_raw: str, chunk_size: int, deep_crawl: bool):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=30,
        length_function=len,
    )
    text_batches: List[str] = []
    meta_batches: List[Dict] = []

    if pdf_files:
        with st.spinner("Extracting PDF text..."):
            for doc in extract_text_from_pdfs(pdf_files):
                chunks = splitter.split_text(doc["content"])
                text_batches.extend(chunks)
                meta_batches.extend(
                    {"source": "pdf", "filename": doc["filename"]} for _ in chunks
                )

    if site_links_raw.strip():
        urls = [u.strip() for u in site_links_raw.replace("\n", ",").split(",") if u.strip()]

        if deep_crawl:
            all_urls: set = set()
            bar = st.progress(0, text="Crawling websites...")
            for i, u in enumerate(urls):
                all_urls.update(get_all_urls(u))
                bar.progress((i + 1) / len(urls))
            urls = list(all_urls)
            bar.empty()
            st.info(f"Deep crawl found **{len(urls)}** unique pages.")

        bar2 = st.progress(0, text="Fetching URL content...")
        for i, url in enumerate(urls):
            bar2.progress((i + 1) / len(urls), text=f"Fetching {url[:60]}...")
            content = extract_text_from_url(url)
            if content:
                chunks = splitter.split_text(content)
                text_batches.extend(chunks)
                meta_batches.extend(
                    {"source": "web", "url": url} for _ in chunks
                )
            time.sleep(0.3)
        bar2.empty()

    if not text_batches:
        st.error("No content extracted. Provide valid PDFs or URLs.")
        return

    with st.spinner(f"Embedding {len(text_batches)} chunks..."):
        t0 = time.time()
        vectors = embed_texts(text_batches)
        embed_time = time.time() - t0

    with st.spinner("Building vector index..."):
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=i, vector=vec, payload={"content": text, "metadata": meta})
            for i, (vec, text, meta) in enumerate(zip(vectors, text_batches, meta_batches))
        ]
        client.upsert(collection_name=COLLECTION, points=points)

    st.session_state.qdrant_client = client
    st.session_state.collection_name = COLLECTION
    st.session_state.chunks_count = len(text_batches)

    unique_sources = len(set(m.get("filename", m.get("url", "")) for m in meta_batches))
    st.success(
        f"Indexed **{len(text_batches)} chunks** from **{unique_sources} source(s)** "
        f"in {embed_time:.1f}s."
    )


# Groq LLM call
def groq_chat(
    messages: List[Dict],
    api_key: str,
    max_tokens: int = 1024,
) -> str:
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# Retrieval
def retrieve(question: str, top_k: int = 5) -> List[Any]:
    client = st.session_state.qdrant_client
    coll   = st.session_state.collection_name
    qvec   = embed_query(question)
    return client.search(
        collection_name=coll,
        query_vector=qvec,
        limit=top_k,
        with_payload=True,
    )


def format_context(hits) -> str:
    parts = []
    for h in hits:
        meta = h.payload["metadata"]
        src = (
            f"[PDF: {meta['filename']}]"
            if meta["source"] == "pdf"
            else f"[Web: {meta['url']}]"
        )
        parts.append(f"{src}\n{h.payload['content']}")
    return "\n\n---\n\n".join(parts)


# Evaluation helpers
def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def compute_retrieval_metrics(hits) -> Dict[str, Any]:
    if not hits:
        return {"mean_retrieval_score": 0.0, "top1_score": 0.0, "num_chunks_retrieved": 0}
    scores = [h.score for h in hits]
    return {
        "mean_retrieval_score": round(sum(scores) / len(scores), 4),
        "top1_score": round(scores[0], 4),
        "num_chunks_retrieved": len(scores),
    }


def compute_answer_relevance(question: str, answer: str) -> float:
    if not answer.strip():
        return 0.0
    return round(cosine_sim(embed_query(question), embed_query(answer[:512])), 4)


def compute_answer_faithfulness(context: str, answer: str) -> float:
    if not context or not answer:
        return 0.0
    return round(cosine_sim(embed_query(context[:512]), embed_query(answer[:512])), 4)


# Answer pipeline
EVAL_PROMPT = """Decide if the CONTEXT below is sufficient to answer the QUESTION.
Reply with exactly 1 (yes) or 0 (no). No other text.

QUESTION: {question}
CONTEXT (first 600 chars): {context}
"""

MAIN_PROMPT = """You are a precise, expert assistant. Answer the QUESTION using ONLY the CONTEXT provided.

Rules:
1. If the answer is not in the CONTEXT, reply: "I don't know based on the provided documents."
2. Be concise and accurate.
3. Format your answer in Markdown.

CONTEXT:
{context}
"""


def answer_question(question: str, api_key: str, top_k: int = 5):
    t_start = time.time()

    hits = retrieve(question, top_k=top_k)
    context = format_context(hits)
    retrieval_metrics = compute_retrieval_metrics(hits)
    retrieval_time = time.time() - t_start

    route_answer = groq_chat(
        messages=[{"role": "user", "content": EVAL_PROMPT.format(question=question, context=context[:600])}],
        api_key=api_key,
        max_tokens=5,
    )
    use_local    = route_answer.strip().startswith("1")
    source_label = "Local documents"

    if use_local:
        st.info("Answering from indexed documents.")
        final_context = context
    else:
        st.info("Local context insufficient - searching the web...")
        web_results = None
        last_err = None
        for attempt in range(3):
            try:
                web_results = DDGS().text(question, max_results=5)
                if web_results:
                    break
            except Exception as e:
                last_err = e
                time.sleep(2)

        if web_results:
            final_context = "\n\n".join(r["body"] for r in web_results)
            source_label  = "Web search (DuckDuckGo)"
        else:
            final_context = context
            if last_err:
                st.warning(f"Web search failed after 3 attempts: {type(last_err).__name__}: {last_err}")
                source_label = "Local documents (web search rate-limited)"
            else:
                source_label = "Local documents (web search returned nothing)"

    t_gen = time.time()
    answer = groq_chat(
        messages=[
            {"role": "system", "content": MAIN_PROMPT.format(context=final_context)},
            {"role": "user",   "content": f"Question: {question}\nAnswer:"},
        ],
        api_key=api_key,
        max_tokens=1024,
    )
    generation_time = time.time() - t_gen
    total_time      = time.time() - t_start

    relevance    = compute_answer_relevance(question, answer)
    faithfulness = compute_answer_faithfulness(final_context, answer)
    word_count   = len(answer.split())

    eval_record = {
        "question":             question,
        "answer":               answer,
        "source":               source_label,
        "retrieval_time_s":     round(retrieval_time, 2),
        "generation_time_s":    round(generation_time, 2),
        "total_time_s":         round(total_time, 2),
        "answer_relevance":     relevance,
        "faithfulness":         faithfulness,
        "word_count":           word_count,
        **retrieval_metrics,
    }
    st.session_state.eval_history.append(eval_record)
    return answer, source_label, eval_record, hits



#  UI
st.markdown('<div class="main-title">DocuWeb RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Agentic RAG over PDFs & websites · Groq LLM · Qdrant in-memory vector store</div>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("Configuration")
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get a free key at console.groq.com",
        placeholder="gsk_...",
    )

    st.divider()
    st.subheader("Knowledge Sources")
    pdf_files  = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
    site_links = st.text_area(
        "Website URLs (one per line or comma-separated)",
        placeholder="https://example.com\nhttps://docs.example.com",
    )
    deep_crawl = st.checkbox(
        "Deep crawl (follow all internal links)",
        help="Scrapes every internal link found on the provided pages.",
    )
    chunk_size = st.slider("Chunk size (chars)", 200, 1000, 400, 50)
    top_k      = st.slider("Chunks to retrieve (top-k)", 1, 10, 5)

    st.divider()
    if st.button("Process & Index Documents", use_container_width=True):
        if not groq_key:
            st.error("Please enter your Groq API key first.")
        elif not pdf_files and not site_links.strip():
            st.error("Provide at least one PDF or URL.")
        else:
            process_and_index(pdf_files, site_links, chunk_size, deep_crawl)

    if st.session_state.chunks_count:
        st.success(f"Index ready: **{st.session_state.chunks_count}** chunks indexed.")

# Tabs 
tab_qa, tab_metrics = st.tabs(["Ask a Question", "Evaluation Metrics"])

# Q&A tab
with tab_qa:
    if not st.session_state.qdrant_client:
        st.info(
            "Enter your Groq API key, upload PDFs or URLs, then click "
            "**Process & Index Documents** to get started."
        )
    else:
        question = st.text_input(
            "Ask anything about your documents:",
            placeholder="What is the main topic of the uploaded documents?",
        )
        ask_col, _ = st.columns([1, 5])
        with ask_col:
            ask_btn = st.button("Get Answer", use_container_width=True)

        if ask_btn:
            if not question.strip():
                st.warning("Please type a question.")
            elif not groq_key:
                st.error("Please enter your Groq API key in the sidebar.")
            else:
                with st.spinner("Thinking..."):
                    answer, source_label, metrics, hits = answer_question(
                        question, groq_key, top_k=top_k
                    )

                st.markdown(f"**Source:** {source_label}")
                st.markdown("### Answer")
                st.markdown(answer)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total time",        f"{metrics['total_time_s']}s")
                c2.metric("Answer Relevance",  f"{metrics['answer_relevance']:.2f}")
                c3.metric("Faithfulness",       f"{metrics['faithfulness']:.2f}")
                c4.metric("Retrieval Score",   f"{metrics['mean_retrieval_score']:.2f}")

                with st.expander("Retrieved chunks"):
                    for i, h in enumerate(hits, 1):
                        meta  = h.payload["metadata"]
                        badge = (
                            f"PDF - {meta['filename']}"
                            if meta["source"] == "pdf"
                            else f"Web - {meta['url']}"
                        )
                        st.markdown(
                            f"**Chunk {i}** - score `{h.score:.4f}` "
                            f"<span class='source-badge'>{badge}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"> {h.payload['content'][:400]}...")

# Evaluation Metrics tab
with tab_metrics:
    st.subheader("RAG Evaluation Dashboard")
    st.caption(
        "Metrics computed locally via embedding cosine similarity. "
        "Answer Relevance = sim(question, answer). "
        "Faithfulness = sim(context, answer). "
        "Retrieval Score = mean cosine similarity of retrieved chunks."
    )

    history = st.session_state.eval_history
    if not history:
        st.info("No queries yet. Ask a question in the Ask a Question tab first.")
    else:
        avg_rel   = sum(r["answer_relevance"]      for r in history) / len(history)
        avg_faith = sum(r["faithfulness"]           for r in history) / len(history)
        avg_ret   = sum(r["mean_retrieval_score"]   for r in history) / len(history)
        avg_time  = sum(r["total_time_s"]           for r in history) / len(history)

        st.markdown("#### Aggregate Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Avg Answer Relevance</div>'
            f'<div class="metric-value">{avg_rel:.2f}</div>'
            f'<div class="metric-sub">0 = off-topic, 1 = on-target</div></div>',
            unsafe_allow_html=True,
        )
        col2.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Avg Faithfulness</div>'
            f'<div class="metric-value">{avg_faith:.2f}</div>'
            f'<div class="metric-sub">grounded in context</div></div>',
            unsafe_allow_html=True,
        )
        col3.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Avg Retrieval Score</div>'
            f'<div class="metric-value">{avg_ret:.2f}</div>'
            f'<div class="metric-sub">cosine similarity</div></div>',
            unsafe_allow_html=True,
        )
        col4.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Avg Latency</div>'
            f'<div class="metric-value">{avg_time:.1f}s</div>'
            f'<div class="metric-sub">end-to-end per query</div></div>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("#### Per-Query Breakdown")
        for i, r in enumerate(reversed(history), 1):
            idx = len(history) - i + 1
            with st.expander(f"Query {idx}: {r['question'][:80]}"):
                q1, q2, q3, q4, q5 = st.columns(5)
                q1.metric("Relevance",       r["answer_relevance"])
                q2.metric("Faithfulness",    r["faithfulness"])
                q3.metric("Retrieval Score", r["mean_retrieval_score"])
                q4.metric("Total Time",      f"{r['total_time_s']}s")
                q5.metric("Words",           r["word_count"])
                st.caption(
                    f"**Source:** {r['source']}  |  "
                    f"Retrieve: {r['retrieval_time_s']}s  |  "
                    f"Generate: {r['generation_time_s']}s  |  "
                    f"Chunks retrieved: {r.get('num_chunks_retrieved', '-')}"
                )
                st.markdown(f"**Answer preview:** {r['answer'][:300]}...")

        st.divider()
        st.download_button(
            label="Export metrics as JSON",
            data=json.dumps(history, indent=2),
            file_name="rag_metrics.json",
            mime="application/json",
        )