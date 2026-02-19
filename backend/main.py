import os
import pickle
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import faiss
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------

INDEX = "faiss_index.bin"
META = "docs_metadata.pkl"

MODEL = "all-MiniLM-L6-v2"
TOP_K = 4

GROQ_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

# --------------------------------------

app = FastAPI(title="Study Abroad Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class Query(BaseModel):
    question: str

print("Loading embedder...")
embedder = SentenceTransformer(MODEL)

index = faiss.read_index(INDEX)
docs = pickle.load(open(META, "rb"))

print("Backend ready.")

# -------- Retrieval --------

def retrieve(q):
    e = embedder.encode([q], convert_to_numpy=True).astype("float32")
    _, I = index.search(e, TOP_K)
    return [docs[i] for i in I[0] if i >= 0]

# -------- Health --------

@app.get("/health")
async def health():
    return {"status": "ok"}

# -------- Chat --------

@app.post("/chat")
async def chat(q: Query):

    hits = retrieve(q.question)

    context = "\n\n".join(h["text"] for h in hits)

    prompt = f"""
You are an expert international study abroad visa advisor.

Answer clearly and professionally using ONLY the context below.

Rules:
- Use bullet points
- Mention country explicitly
- Be concise
- If unsure, say so
- Do NOT hallucinate
- End with "Always verify on official government websites."

Context:
{context}

Question:
{q.question}

Answer:
"""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a professional visa consultant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 400
    }

    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
        data = r.json()

        print("GROQ RESPONSE:", data)

        answer = data["choices"][0]["message"]["content"]

    except Exception as e:
        print("LLM FAILURE:", e)
        return {
            "answer": "Groq temporarily unavailable. Please retry.",
            "sources": []
        }

    return {
        "answer": answer,
        "sources": []
    }
