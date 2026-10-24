from hashlib import sha256
from io import BytesIO

import chromadb
from google import genai
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


MODEL_NAME = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_HOPS = 3
TOP_K = 3


st.set_page_config(page_title="Multi-Hop RAG", page_icon="🔎")
st.title("Multi-Hop RAG Question Answering")
st.write(
    "Upload text or PDF documents, then ask a question that may require "
    "connecting information across them."
)


@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="project_documents")


def extract_pages(uploaded_file):
    """Return a list of (text, page_number) pairs from a PDF or TXT file."""
    file_name = uploaded_file.name
    file_bytes = uploaded_file.getvalue()

    if file_name.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        return [
            (page.extract_text() or "", page_number)
            for page_number, page in enumerate(reader.pages, start=1)
        ]

    text = file_bytes.decode("utf-8", errors="ignore")
    return [(text, None)]


def split_into_chunks(text, max_words=180, overlap=30):
    """Split text into overlapping word chunks."""
    words = text.split()
    if not words:
        return []

    step = max_words - overlap
    chunks = []

    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + max_words])
        if chunk:
            chunks.append(chunk)

    return chunks


def index_files(uploaded_files):
    """Read, chunk, embed, and save the uploaded files in Chroma."""
    embedder = get_embedder()
    collection = get_collection()

    all_chunks = []
    all_metadata = []
    all_ids = []

    for uploaded_file in uploaded_files:
        for page_text, page_number in extract_pages(uploaded_file):
            chunks = split_into_chunks(page_text)

            for chunk_number, chunk in enumerate(chunks):
                page_label = page_number if page_number is not None else 0
                unique_text = (
                    f"{uploaded_file.name}|{page_label}|{chunk_number}|{chunk}"
                )
                chunk_id = sha256(unique_text.encode("utf-8")).hexdigest()

                all_chunks.append(chunk)
                all_metadata.append(
                    {
                        "source": uploaded_file.name,
                        "page": page_label,
                    }
                )
                all_ids.append(chunk_id)

    if not all_chunks:
        raise ValueError(
            "I couldn't extract text. If this is a scanned PDF, it may need OCR."
        )

    vectors = embedder.encode(
        all_chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        metadatas=all_metadata,
        embeddings=vectors.tolist(),
    )

    return len(all_chunks)


def retrieve(query, top_k=TOP_K):
    """Find document chunks most relevant to a query."""
    embedder = get_embedder()
    collection = get_collection()
    count = collection.count()

    if count == 0:
        return []

    query_vector = embedder.encode(
        query,
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(top_k, count),
    )

    found = []
    for text, metadata in zip(
        results["documents"][0],
        results["metadatas"][0],
    ):
        found.append({"text": text, "metadata": metadata})

    return found


def format_evidence(results):
    """Format retrieved chunks with their source names."""
    parts = []

    for item in results:
        source = item["metadata"]["source"]
        page = item["metadata"]["page"]

        if page:
            citation = f"{source}, page {page}"
        else:
            citation = source

        parts.append(f"[Source: {citation}]\n{item['text']}")

    return "\n\n".join(parts)


def ask_model(prompt):
    """Send a prompt to Gemini and return its text response."""
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return (response.text or "").strip()


def answer_question(question):
    """Retrieve evidence over multiple hops, then generate an answer."""
    all_evidence = []
    seen_chunks = set()
    seen_queries = {question.strip().casefold()}
    hop_trace = []
    current_query = question

    for hop_number in range(1, MAX_HOPS + 1):
        retrieved = retrieve(current_query)
        new_evidence = []

        for item in retrieved:
            metadata = item["metadata"]
            chunk_key = (
                metadata.get("source"),
                metadata.get("page"),
                item["text"],
            )

            # Keep the same chunk from being added repeatedly.
            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                new_evidence.append(item)
                all_evidence.append(item)

        hop_trace.append(
            {
                "number": hop_number,
                "query": current_query,
                "evidence": new_evidence,
            }
        )

        if not new_evidence or hop_number == MAX_HOPS:
            break

        evidence_so_far = format_evidence(all_evidence)
        previous_queries = "\n".join(
            f"- {hop['query']}" for hop in hop_trace
        )

        next_query = ask_model(
            "You are controlling an iterative multi-hop document search. "
            "Read the question and all evidence found so far. "
            "If the evidence is enough to answer the whole question, return "
            "only DONE. Otherwise, return one short search query targeting "
            "the missing information. Do not answer the question yet. "
            "Do not repeat a previous search query.\n\n"
            f"Question: {question}\n\n"
            f"Previous search queries:\n{previous_queries}\n\n"
            f"Evidence found so far:\n{evidence_so_far}"
        )

        next_query = next_query.strip().strip('"').strip("'")

        if not next_query or next_query.upper().startswith("DONE"):
            break

        normalized_query = next_query.casefold()
        if normalized_query in seen_queries:
            break

        seen_queries.add(normalized_query)
        current_query = next_query

    if not all_evidence:
        raise ValueError(
            "No relevant document text was found. Upload and index documents first."
        )

    final_answer = ask_model(
        "Answer the question using only the document evidence below. "
        "Cite supporting sources using the exact filenames shown, "
        "for example [company.txt] or [notes.pdf, page 2]. "
        "If the evidence is insufficient, say so.\n\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{format_evidence(all_evidence)}"
    )

    return final_answer, {"hops": hop_trace}


with st.sidebar:
    st.header("Your documents")

    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Index documents", type="primary"):
        if not uploaded_files:
            st.warning("Choose at least one file first.")
        else:
            try:
                with st.spinner("Reading and indexing your documents..."):
                    number_of_chunks = index_files(uploaded_files)
                st.success(f"Indexed {number_of_chunks} text chunks.")
            except Exception as error:
                st.error(f"Indexing failed: {error}")

    try:
        indexed_count = get_collection().count()
        st.caption(f"Indexed chunks saved: {indexed_count}")
    except Exception:
        st.caption("The search index will be created when you index your first files.")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        trace = message.get("trace")
        if trace and "hops" in trace:
            with st.expander("See how the multi-hop search worked"):
                for hop in trace["hops"]:
                    st.write(
                        f"**Hop {hop['number']} search:** {hop['query']}"
                    )

                    if hop["evidence"]:
                        for item in hop["evidence"]:
                            st.write(
                                f"- {item['metadata']['source']}: "
                                f"{item['text']}"
                            )
                    else:
                        st.caption("No new evidence found in this hop.")


question = st.chat_input("Ask a question about your documents")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    try:
        with st.spinner(
            f"Searching across up to {MAX_HOPS} hops and preparing an answer..."
        ):
            answer, trace = answer_question(question)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "trace": trace,
            }
        )
    except Exception as error:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Something went wrong: {error}",
            }
        )

    st.rerun()