import streamlit as st
import PyPDF2
import json
import requests
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from litellm import completion
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

if "client" not in st.session_state:
    st.session_state.client = None
if "db_collection" not in st.session_state:
    st.session_state.db_collection = None


def get_all_urls(base_url):
    urls = set()
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.find_all("a", href=True):
                url = link["href"]
                full_url = urljoin(base_url, url)
                parsed_url = urlparse(full_url)
                if parsed_url.netloc == urlparse(base_url).netloc:
                    urls.add(
                        parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path
                    )
    except Exception as e:
        st.error(f"An error occurred while crawling {base_url}: {e}")
    return urls


def extract_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()

            lines = (line.strip() for line in text.splitlines())

            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

            text = " ".join(chunk for chunk in chunks if chunk)

            return text
        else:
            st.warning(
                f"Failed to fetch content from {url}: Status code {response.status_code}"
            )
            return None
    except Exception as e:
        st.warning(f"Error extracting text from {url}: {e}")
        return None


def fetch_url_content(url: str) -> Optional[str]:
    try:
        return extract_text_from_url(url)
    except Exception as e:
        st.error(f"Error: Failed to fetch URL {url}. Exception: {e}")
        return None


def get_embeddings(texts, model="text-embedding-3-small", openai_key=None):
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_key}"}
    data = {"input": texts, "model": model}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["data"]
    else:
        st.error(f"Error {response.status_code}: {response.text}")
        return None


def process_uploaded_pdfs(pdf_docs):
    extracted_pdfs = []
    for uploaded_file in pdf_docs:
        content = ""
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                content += page.extract_text()
            extracted_pdfs.append({"content": content, "filename": uploaded_file.name})
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    return extracted_pdfs


def process_and_index_documents(
    pdf_docs, site_links=None, chunk_size=150, deep_crawl=False
):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o-mini",
        chunk_size=chunk_size,
        chunk_overlap=0,
    )

    text_batches = []
    meta_info = []

    if pdf_docs:
        parsed_docs = process_uploaded_pdfs(pdf_docs)
        for doc in parsed_docs:
            chunks = text_splitter.split_text(doc["content"])
            text_batches.extend(chunks)
            for _ in chunks:
                meta_info.append(
                    {"filename": doc["filename"], "source": "pdf_dataset"}
                )

    if site_links:
        urls = [url.strip() for url in site_links.split(",")]

        if deep_crawl:
            all_urls = set()
            status_bar = st.progress(0)
            status_text = st.empty()

            for i, base_url in enumerate(urls):
                status_text.text(f"Crawling website: {base_url}")
                site_urls = get_all_urls(base_url)
                all_urls.update(site_urls)
                status_bar.progress((i + 1) / len(urls))

            urls = list(all_urls)
            status_text.text(f"Found {len(urls)} unique URLs")

        status_bar = st.progress(0)
        status_text = st.empty()

        for i, url in enumerate(urls):
            status_text.text(f"Processing URL {i+1}/{len(urls)}: {url}")
            content = fetch_url_content(url)

            if content is not None:
                chunks = text_splitter.split_text(content)
                text_batches.extend(chunks)
                for _ in chunks:
                    meta_info.append({"url": url, "source": "web_content"})

            status_bar.progress((i + 1) / len(urls))
            time.sleep(0.5)

        status_text.empty()
        status_bar.empty()

    if not text_batches:
        st.error("No content to process. Please provide valid PDFs or web URLs.")
        return None, None

    openai_key = st.session_state.openai_api_key

    with st.spinner("Generating embeddings..."):
        embed_objs = get_embeddings(text_batches, openai_key=openai_key)
        if not embed_objs:
            return None, None
        embeddings = [obj["embedding"] for obj in embed_objs]

    client = QdrantClient("http://localhost:6333")
    db_collection = "agent_rag_index"
    EMBEDDING_DIM = 1536

    with st.spinner("Creating vector database..."):
        client.delete_collection(db_collection)
        client.create_collection(
            db_collection=db_collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        ids = list(range(len(text_batches)))
        payload = [
            {"content": chunk, "metadata": metadata}
            for chunk, metadata in zip(text_batches, meta_info)
        ]

        client.upload_collection(
            db_collection=db_collection,
            vectors=embeddings,
            payload=payload,
            ids=ids,
            batch_size=256,
        )

    st.success(
        f"Indexed {len(text_batches)} chunks from {len(set(m['source'] for m in meta_info))} different sources"
    )
    return client, db_collection


def answer_question(question, client, db_collection, top_k=3):
    if not question.strip():
        st.warning("Please enter a question.")
        return

    def search(text: str):
        search_vector = get_embeddings(text, openai_key=st.session_state.openai_api_key)[
            0
        ]["embedding"]
        return client.search(
            db_collection=db_collection, query_vector=search_vector, limit=top_k
        )

    def format_docs(docs):
        formatted_segments = []
        for doc in docs:
            source_info = ""
            if doc.payload["metadata"]["source"] == "pdf_dataset":
                source_info = (
                    f"\nSource: PDF file {doc.payload['metadata']['filename']}"
                )
            else:
                source_info = f"\nSource: Web article {doc.payload['metadata']['url']}"
            formatted_segments.append(doc.payload["content"] + source_info)
        return "\n\n".join(formatted_segments)

    eval_sys_prompt = """Your job is decide if a given question can be answered with a given context. 
    If context can answer the question return 1.
    If not return 0.
    Context: {context}
    """

    main_sys_prompt = """You are an expert in answering questions. Provide answers based **exclusively** on the given context. 

        **Rules:**
        1. If the question cannot be answered using the context, respond only with: "I don't know."
        2. Do **not** infer, assume, or add information not explicitly provided in the context.
        3. Your answers must be:
        - **Concise**: Avoid unnecessary details.
        - **Informative**: Focus on actionable and precise responses.
        4. Format your response in **Markdown**.

        **Context:** {context}

    """

    main_user_prompt = """
    Question: {question}
    Answer:"""

    with st.spinner("Searching for relevant information..."):
        results = search(question)
        context = format_docs(results)

        response = completion(
            model="gpt-4o-mini",
            messages=[
                {
                    "content": eval_sys_prompt.format(context=context),
                    "role": "system",
                },
                {"content": main_user_prompt.format(question=question), "role": "user"},
            ],
            max_tokens=50,
            openai_key=st.session_state.openai_api_key,
        )
        has_answer = response.choices[0].message.content

        if has_answer == "1":
            st.info("Found relevant information in the indexed content")
            response = completion(
                model="gpt-4o-mini",
                messages=[
                    {
                        "content": main_sys_prompt.format(context=context),
                        "role": "system",
                    },
                    {"content": main_user_prompt.format(question=question), "role": "user"},
                ],
                max_tokens=1000,
                openai_key=st.session_state.openai_api_key,
            )
            st.markdown(response.choices[0].message.content)
        else:
            st.info("No relevant information found. Searching online...")
            results = DDGS().text(question, max_results=5)
            context = "\n\n".join(doc["body"] for doc in results)
            st.info("Found online sources. Generating the response...")
            response = completion(
                model="gpt-4o-mini",
                messages=[
                    {
                        "content": main_sys_prompt.format(context=context),
                        "role": "system",
                    },
                    {"content": main_user_prompt.format(question=question), "role": "user"},
                ],
                max_tokens=1000,
                openai_key=st.session_state.openai_api_key,
            )
            st.markdown(response.choices[0].message.content)


st.title("RAG System with PDF and Website Crawling Support")

openai_key = st.text_input("Enter your OpenAI API Key:", type="password")
if openai_key:
    st.session_state.openai_api_key = openai_key

pdf_docs = st.file_uploader(
    "Upload PDF files:", accept_multiple_files=True, type=["pdf"]
)

st.subheader("Website Input")
site_links = st.text_input(
    "Enter website URLs (comma-separated):", placeholder="https://example.com"
)
deep_crawl = st.checkbox(
    "Crawl entire website(s)",
    help="Enable this to extract content from all pages of the specified website(s)",
)

if st.button("Process and Index Documents"):
    if not st.session_state.get("openai_api_key"):
        st.error("Please enter your OpenAI API key first.")
    else:
        st.session_state.client, st.session_state.db_collection = (
            process_and_index_documents(
                pdf_docs, site_links, deep_crawl=deep_crawl
            )
        )

if st.session_state.client and st.session_state.db_collection:
    question = st.text_input("Ask a question about the documents:")
    if st.button("Get Answer"):
        answer_question(
            question, st.session_state.client, st.session_state.db_collection
        )
elif pdf_docs or site_links:
    st.warning("Please process and index the documents first.")
else:
    st.info("Upload PDFs or provide web URLs to get started.")
