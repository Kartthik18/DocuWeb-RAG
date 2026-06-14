# Agentic RAG Pipeline with PDF & Web Scraping Support

This repository provides an **Agentic Retrieval-Augmented Generation (RAG)** solution. It allows you to query information from both uploaded PDF documents and provided website links. A built-in intelligent agent evaluates your question to decide whether it can be answered using the provided context, or if an external online search is necessary.

## Highlights

1. **PDF Processing**: Upload your PDF documents and extract their text for direct question answering.
2. **Web Content Extraction**: Input website URLs to fetch content and utilize it as knowledge base material.
3. **Hybrid Retrieval**: Combine both PDF and web resources into a unified vector database to retrieve comprehensive answers.
4. **Smart Agent Routing**: 
    - First, attempts to find the answer within the processed PDF context.
    - If missing, checks the extracted web content.
    - If the context does not contain the answer, it determines that the question is out-of-scope for the RAG database.
5. **Web Search Fallback**: Automatically falls back to DuckDuckGo search if the database cannot provide an answer.

## Technology Stack

- **Streamlit**: For the interactive web interface.
- **PyPDF2**: For PDF text extraction.
- **BeautifulSoup**: For parsing and scraping HTML web content.
- **OpenAI API**: For computing vector embeddings and generating responses.
- **Qdrant**: As the local vector database for semantic similarity searches.
- **DuckDuckGo Search**: For online web search capabilities when local context is insufficient.

## Quick Start

### 1. Clone the Project
```bash
git clone <your-github-repo-url>
cd Agentic_RAG
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Initialize Qdrant
- Ensure you have [Qdrant](https://qdrant.tech/documentation/quick_start/) installed.
- Run Qdrant locally so it is accessible at `http://localhost:6333`.

## How to Use

1. **Launch the App:**
    ```bash
    streamlit run app.py
    ```

2. **Set API Key:**
   - Enter your OpenAI API Key into the application's sidebar/input field.

3. **Provide Knowledge Sources:**
   - Upload one or more PDF files.
   - Enter website URLs separated by commas.
   - Check the crawling option if you want to extract text from all interlinked pages.

4. **Index the Data:**
   - Hit "Process and Index Documents". This will split the text, create embeddings, and load them into Qdrant.

5. **Query the System:**
   - Type in your question.
   - The agent will evaluate the sources:
       - First checks the uploaded PDFs.
       - Then checks the provided web URLs.
       - Finally, performs a web search if the information is completely missing from your uploaded context.

## Configuration Details

- **OpenAI API Key**: Required for the `gpt-4o-mini` language model and `text-embedding-3-small` embeddings.
- **Qdrant**: By default, the app expects a local Qdrant instance running on port 6333. Update `app.py` if you use a cloud instance.

## Dependencies

Make sure you are using Python 3.8+ (tested on 3.12). Below are the core packages used:
- `streamlit`
- `PyPDF2`
- `beautifulsoup4`
- `qdrant-client`
- `litellm`
- `duckduckgo_search`
- `langchain_text_splitters`

Install them all simultaneously via `pip install -r requirements.txt`.

## FAQ

**Q: Can I use both PDFs and URLs at the same time?**
Yes. The system indexes both sources and will query across the unified dataset, prioritizing the highest similarity matches.

**Q: Does the agent answer random questions not in the text?**
No. If the answer is absent from the provided context, the agent will clearly state that it cannot answer using the internal database, before potentially relying on the web search fallback.

**Q: What if the app fails to connect to Qdrant?**
Ensure that the Qdrant server is successfully running in a separate terminal or Docker container on `localhost:6333` prior to indexing your documents.
