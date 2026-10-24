from pathlib import Path

import ollama
from sentence_transformers import SentenceTransformer, util


# Load each text file as a document
documents_folder = Path("documents")
documents = []

for path in documents_folder.glob("*.txt"):
    text = path.read_text(encoding="utf-8").strip()
    if text:
        documents.append({"source": path.name, "text": text})

if not documents:
    raise RuntimeError("Put at least one .txt file inside the documents folder.")

# Turn the documents into embeddings so we can search by meaning
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

document_embeddings = embedding_model.encode(
    [document["text"] for document in documents],
    convert_to_tensor=True,
    normalize_embeddings=True,
)


def retrieve(query, number_of_results=2):
    """Find the documents most related to a search query."""
    query_embedding = embedding_model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    matches = util.semantic_search(
        query_embedding,
        document_embeddings,
        top_k=min(number_of_results, len(documents)),
    )[0]

    return [documents[match["corpus_id"]] for match in matches]


def ask_model(prompt):
    """Send a prompt to the local Ollama model."""
    response = ollama.chat(
        model="gemma3:1b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


def format_documents(found_documents):
    return "\n\n".join(
        f"Source: {document['source']}\n{document['text']}"
        for document in found_documents
    )


question = input("Ask a question: ").strip()

# HOP 1: Search using the user's original question
first_results = retrieve(question)
first_evidence = format_documents(first_results)

# Ask the model what to search for next, using the first evidence
follow_up_query = ask_model(
    "You are preparing a second search for a two-hop question-answering system.\n"
    "Read the question and the evidence. Identify the person, company, place, "
    "or other key entity found in the evidence that will help answer the question. "
    "Return only a short search query for the missing information.\n\n"
    f"Question: {question}\n\n"
    f"Evidence from the first search:\n{first_evidence}"
)

# HOP 2: Search again using the follow-up query
second_results = retrieve(follow_up_query)
second_evidence = format_documents(second_results)

# Use evidence from both hops to form the answer
answer = ask_model(
    "Answer the question using only the evidence below. "
    "Cite the supporting filename in square brackets, such as [project.txt]. "
    "If the evidence does not contain the answer, say that there is not enough evidence.\n\n"
    f"Question: {question}\n\n"
    f"Evidence from hop 1:\n{first_evidence}\n\n"
    f"Evidence from hop 2:\n{second_evidence}"
)

print("\n--- Hop 1 evidence ---")
print(first_evidence)

print("\n--- Follow-up search ---")
print(follow_up_query)

print("\n--- Hop 2 evidence ---")
print(second_evidence)

print("\n--- Final answer ---")
print(answer)