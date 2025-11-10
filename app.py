# import streamlit as st
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from transformers import pipeline
# from PyPDF2 import PdfReader
# import docx

# # --- Load embedding model ---
# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # --- Load QA model ---
# qa_model = pipeline(
#     "text2text-generation",
#     model="google/flan-t5-base",
#     device=-1,          # forces CPU
#     trust_remote_code=True
# )

# # --- Function to load text from different file types ---
# def load_text(file):
#     if file.name.endswith(".txt"):
#         return file.getvalue().decode("utf-8")
#     elif file.name.endswith(".pdf"):
#         reader = PdfReader(file)
#         return "\n".join(page.extract_text() or "" for page in reader.pages)
#     elif file.name.endswith(".docx"):
#         doc = docx.Document(file)
#         return "\n".join([p.text for p in doc.paragraphs])
#     else:
#         return ""

# # --- Function to split text into chunks ---
# def chunk_text(text, chunk_size=150):
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size):
#         chunks.append(" ".join(words[i:i + chunk_size]))
#     return chunks

# # --- Initialize FAISS index ---
# def create_faiss_index(chunks):
#     embeddings = embedder.encode(chunks)
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(np.array(embeddings, dtype=np.float32))
#     return index, embeddings

# # --- Streamlit UI ---
# st.title("📘 Custom Chatbot (Document-Based)")

# uploaded_file = st.file_uploader("Upload your document (.txt, .pdf, .docx)")
# if uploaded_file:
#     doc_text = load_text(uploaded_file)
#     if doc_text.strip() == "":
#         st.warning("Unable to read content from this file.")
#     else:
#         chunks = chunk_text(doc_text)
#         index, doc_embeddings = create_faiss_index(chunks)
#         st.success(f"Loaded '{uploaded_file.name}' with {len(chunks)} chunks.")

#         question = st.text_input("Ask a question based on your document:")
#         if st.button("Get Answer") and question.strip():
#             # Retrieve top 5 relevant chunks
#             q_emb = embedder.encode([question])
#             D, I = index.search(np.array(q_emb, dtype=np.float32), 5)
#             context = " ".join([chunks[i] for i in I[0]])

#             # Prepare prompt with context
#             prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
            
#             # Generate answer
#             answer = qa_model(prompt, max_new_tokens=200)[0]['generated_text']
#             st.success(answer)

# import streamlit as st
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from transformers import pipeline
# import os

# # --- Load embedding model ---
# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # --- Load QA model ---
# qa_model = pipeline(
#     "text2text-generation",
#     model="google/flan-t5-base",
#     device=-1,          # CPU only
#     trust_remote_code=True
# )

# # --- Load document from project folder ---
# PROJECT_DOC = "document.txt"  # your source file
# if not os.path.exists(PROJECT_DOC):
#     st.error(f"Document {PROJECT_DOC} not found in project folder.")
#     st.stop()

# with open(PROJECT_DOC, "r", encoding="utf-8") as f:
#     doc_text = f.read()

# # --- Split into chunks (~100 words each) ---
# def chunk_text(text, chunk_size=100):
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size):
#         chunks.append(" ".join(words[i:i + chunk_size]))
#     return chunks

# chunks = chunk_text(doc_text, chunk_size=100)

# # --- Embed chunks ---
# doc_embeddings = embedder.encode(chunks)
# dimension = doc_embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(np.array(doc_embeddings, dtype=np.float32))

# # --- Streamlit UI ---
# st.title("📘 Custom Chatbot (Document-Based)")
# st.write(f"Document loaded: **{PROJECT_DOC}**")

# question = st.text_input("Ask a question based on your document:")

# if st.button("Get Answer") and question.strip():
#     q_embedding = embedder.encode([question])
#     D, I = index.search(np.array(q_embedding, dtype=np.float32), 5)  # top 5 chunks
#     context = " ".join([chunks[i] for i in I[0]])
#     prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
#     answer = qa_model(prompt, max_length=150)[0]['generated_text']
#     st.success(answer)
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import torch
import os

# --- Force CPU device ---
device = torch.device("cpu")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# --- Load QA model ---
qa_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1,  # CPU
    trust_remote_code=True
)

PROJECT_DOC = "document.txt"
if not os.path.exists(PROJECT_DOC):
    st.error(f"Document '{PROJECT_DOC}' not found in project folder.")
    st.stop()

with open(PROJECT_DOC, "r", encoding="utf-8") as f:
    doc_text = f.read()

def chunk_text(text, chunk_size=100):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = chunk_text(doc_text, chunk_size=100)

# --- Use Streamlit cache to store embeddings & FAISS index ---
@st.cache_resource
def create_index(chunks):
    # Embed in batches to reduce memory spikes
    batch_size = 64
    embeddings_list = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = embedder.encode(batch_chunks)
        embeddings_list.append(batch_embeddings)
    doc_embeddings = np.vstack(embeddings_list).astype(np.float32)

    # FAISS index
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(doc_embeddings)
    return index, doc_embeddings

index, doc_embeddings = create_index(chunks)

# --- Streamlit UI ---
st.title("📘 Custom Chatbot (Document-Based)")
st.write(f"Document loaded: **{PROJECT_DOC}**")

question = st.text_input("Ask a question based on your document:")

if st.button("Get Answer") and question.strip():
    q_embedding = embedder.encode([question]).astype(np.float32)
    D, I = index.search(q_embedding, 5)  # top 5 chunks
    context = " ".join([chunks[i] for i in I[0]])

    prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    answer = qa_model(prompt, max_new_tokens=200)[0]['generated_text']
    st.success(answer)
