import os
import pickle
import asyncio
import faiss
import numpy as np
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from groq import Groq
from PyPDF2 import PdfReader
import os
from groq import Groq

# ---------------- Config ----------------
UPLOAD_FOLDER = "uploads"
DB_FAISS_PATH = "vector_store"
CHUNKS_PATH = "chunks.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama3-8b-8192"
MAX_CHUNKS = 3  # max chunks to send to Groq to avoid context length exceeded

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- Init Models ----------------
embedder = SentenceTransformer(EMBED_MODEL)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- Chat Memory ----------------
chat_memory = {}  # session_id -> list of messages

def get_memory(session_id):
    if session_id not in chat_memory:
        chat_memory[session_id] = []
    return chat_memory[session_id]

# ---------------- Utils ----------------
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return text
    else:  # txt or other text files
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def build_faiss(chunks):
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, DB_FAISS_PATH)


def search_faiss(query, k=MAX_CHUNKS):
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index(DB_FAISS_PATH)
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return [chunks[i] for i in I[0]]


async def run_rag(user_prompt, session_id="default"):
    memory = get_memory(session_id)
    retrieved = search_faiss(user_prompt)
    context = "\n\n".join(retrieved[:MAX_CHUNKS])  # limit top chunks

    system_prompt = """You are a helpful AI assistant.
Answer using the following structure:
- Summary: brief overview
- Details: bullet points with reasoning"""

    # Include past conversation for continuity
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(memory)
    messages.append({"role": "user", "content": user_prompt + "\n\nContext:\n" + context})

    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=GROQ_MODEL,
        messages=messages
    )
    assistant_reply = response.choices[0].message.content
    memory.append({"role": "assistant", "content": assistant_reply})  # store reply
    return assistant_reply

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return jsonify({"status": "error", "message": "No file uploaded."})

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    text = extract_text(filepath)
    chunks = chunk_text(text)
    build_faiss(chunks)

    return jsonify({"status": "success", "message": "File uploaded and processed successfully."})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_prompt = data.get("prompt")
    session_id = data.get("session_id", "default")  # multiple sessions
    if not user_prompt:
        return jsonify({"status": "error", "message": "No prompt provided."})

    try:
        answer = asyncio.run(run_rag(user_prompt, session_id))
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

    return jsonify({"status": "success", "answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
