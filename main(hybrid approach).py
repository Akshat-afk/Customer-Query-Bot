import os
import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from supabase import create_client
from datetime import datetime, timezone
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# --- Config ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)  # type: ignore

genai.configure(api_key=os.getenv("GEN_API_KEY")) # type: ignore

# --- Models ---
class Query(BaseModel):
    channel: str
    from_: str
    text: str

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def fetch_kb():
    try:
        data = supabase.table("knowledge_base").select("id, topic, body").execute()
        kb_entries = data.data or []
        for entry in kb_entries:
            entry["embedding"] = embed_model.encode(entry["topic"] + " " + entry["body"])
        return kb_entries
    except Exception as e:
        print("Supabase KB fetch error:", e)
        return []

KB = fetch_kb()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_kb(query_text):
    if not KB:
        return None
    query_vec = embed_model.encode(query_text)
    best_score = 0
    best_entry = None
    for entry in KB:
        score = cosine_similarity(query_vec, entry["embedding"])
        if score > best_score:
            best_score = score
            best_entry = entry
    
    return best_entry

def lookup_supabase(identifier: str):
    try:
        data = supabase.table("authors").select("*").or_(
            f"email.eq.{identifier},ISBN.eq.{identifier},book.eq.{identifier}"
        ).execute()
        return data.data[0] if data.data else None
    except Exception as e:
        print("Supabase author lookup error:", e)
        return None

def generate_response_gemini(user_query, context_text):
    prompt = f"""
You are a friendly BookLeaf Publishing assistant.

User query: "{user_query}"

Use the following context to answer clearly:
"{context_text}"

Answer concisely in natural language.
"""
    try:
        model = genai.GenerativeModel("models/gemini-flash-lite-latest") # type: ignore
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        print("Gemini generation error:", e)
        return "I'm unable to provide an answer at the moment."

def escalate(query_text, reason):
    ticket = {
        "ticket_id": f"TKT-{datetime.now(timezone.utc).timestamp()}",
        "query": query_text,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    log_interaction(query_text, f"Escalated: {reason}", 0.0)
    return {"escalated": True, "ticket": ticket}

def log_interaction(query, response, confidence):
    log = {
        "query": query,
        "response": response,
        "confidence": confidence,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open("logs.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(log) + "\n")

# --- Main Endpoint ---
@app.post("/api/query")
async def handle_query(q: Query):
    record = lookup_supabase(q.from_)

    kb_hit = search_kb(q.text)
    context_text = ""
    if record:
        context_text += f"Book: {record.get('title','N/A')}, Live date: {record.get('book_live_date','N/A')}, Royalty status: {record.get('royalty_status','N/A')}\n"
    if kb_hit:
        context_text += f"KB context: {kb_hit['topic']} - {kb_hit['body']}"

    if not context_text:
        return escalate(q.text, "no_record_or_kb")

    response_text = generate_response_gemini(q.text, context_text)

    confidence = 0.9 if kb_hit or record else 0.7

    escalated = confidence < 0.8

    log_interaction(q.text, response_text, confidence)

    return {"response": response_text, "confidence": confidence, "escalated": escalated}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)