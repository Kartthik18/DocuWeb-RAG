# DocuWeb RAG

An agentic Retrieval-Augmented Generation (RAG) pipeline that answers questions from uploaded PDF documents and website content. When the local knowledge base cannot answer a question, the system falls back to a live DuckDuckGo web search.

---

## Features

- **PDF ingestion**: Upload one or more PDF files and extract their text for indexing.
- **Website ingestion**: Provide URLs to scrape and index web page content.
- **Deep crawl**: Optionally follow all internal links on a given website to extract content from every page.
- **Local embeddings**: Text is embedded using `all-MiniLM-L6-v2` from sentence-transformers, running fully on your machine with no external API required.
- **In-memory vector store**: Qdrant runs in-memory, so no Docker or external server setup is needed.
- **LLM generation**: Answers are generated using the Groq API (free tier).
- **Intelligent routing**: Before generating an answer, the system checks whether the retrieved context is sufficient. If not, it falls back to DuckDuckGo web search.
- **Evaluation metrics**: Every query is scored across four dimensions and results can be exported as JSON.

---

## Evaluation Metrics

All metrics are computed locally without any external evaluation API.

| Metric | Description |
| --- | --- |
| Answer Relevance | Cosine similarity between the question embedding and the answer embedding. Measures how on-topic the answer is. |
| Faithfulness | Cosine similarity between the retrieved context embedding and the answer embedding. Measures how grounded the answer is in the source material. |
| Retrieval Score | Mean cosine similarity of the top-k retrieved chunks to the query. Measures retrieval quality. |
| Latency | End-to-end time broken down into retrieval time and generation time. |
| Word Count | Number of words in the generated answer. |

An aggregate statistics dashboard shows averages across all queries in a session. Individual query results can be exported as a JSON file.

---

## Technology Stack

| Component | Technology |
| --- | --- |
| UI | Streamlit |
| PDF parsing | PyPDF2 |
| Web scraping | BeautifulSoup4, Requests |
| Text chunking | LangChain Text Splitters |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store | Qdrant (in-memory) |
| LLM | Groq API |
| Web search fallback | DuckDuckGo (ddgs) |

---

## Requirements

- Python 3.9 or higher
- A free Groq API key from [console.groq.com](https://console.groq.com)

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd DocuWeb-RAG
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

The sentence-transformer model (`all-MiniLM-L6-v2`) will be downloaded automatically on first run.

---

## Running the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. **Enter your Groq API key** in the sidebar. Get one for free at [console.groq.com](https://console.groq.com).

2. **Add knowledge sources** in the sidebar:
   - Upload one or more PDF files, or
   - Paste website URLs (one per line or comma-separated), or
   - Use both at the same time.

3. **Configure settings** (optional):
   - Enable "Deep crawl" to follow all internal links on provided URLs.
   - Adjust chunk size to control how text is split for indexing.
   - Adjust the top-k slider to control how many chunks are retrieved per query.

4. **Click "Process and Index Documents"** to extract, embed, and store all content.

5. **Ask a question** in the "Ask a Question" tab. The system will:
   - Retrieve the most relevant chunks from the index.
   - Decide whether the context is sufficient to answer.
   - If yes, generate an answer from the indexed content.
   - If no, fall back to a live DuckDuckGo search and generate an answer from the web results.

6. **View metrics** in the "Evaluation Metrics" tab after running queries.

---

## Notes

- The vector index is in-memory and resets each time the app restarts. Re-index your documents after each restart.
- DuckDuckGo search is rate-limited. If the fallback fails, the app retries up to three times with a short delay and will display a warning if all attempts fail.
- The Groq free tier is rate-limited at the organization level. If you hit limits, wait a moment and retry.
- To find which models are available on your Groq account, run:

```bash
curl -X GET "https://api.groq.com/openai/v1/models" \
     -H "Authorization: Bearer YOUR_GROQ_API_KEY"
```

Update the `GROQ_MODEL` constant in `app.py` to match a model available on your account.

---

## Dependencies

```
streamlit
PyPDF2
langchain-text-splitters
sentence-transformers
qdrant-client
ddgs
beautifulsoup4
groq
requests
```
