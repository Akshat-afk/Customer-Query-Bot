from fastapi import FastAPI
from supabase import create_client
from pydantic import BaseModel
import google.generativeai as genai
import json, datetime, os
from dotenv import load_dotenv
import numpy as np

load_dotenv()
app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)  # type: ignore

genai.configure(api_key=os.getenv("GEN_API_KEY")) # type: ignore

class Query(BaseModel):
    channel: str
    from_: str
    text: str

def embed_text(text: str):
    """Generate Gemini embeddings for text."""
    try:
        response = genai.embed_content( # type: ignore
        model="models/embedding-001",
        content=text
        )
        return response["embedding"]
    except Exception as e:
        print("Embedding error:", e)
        return np.zeros(768)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def classify_intent(text: str):
    """Classify using Gemini if possible, else fallback to keywords."""
    prompt = f"""
    You are a customer query classifier for BookLeaf Publishing.
    Read this query and respond ONLY in JSON with two fields: intent and confidence.

    Example:
    {{
      "intent": "royalty",
      "confidence": 0.92
    }}

    Query: "{text}"
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash") # type: ignore
        response = model.generate_content(prompt)
        result = json.loads(response.text.strip())
        return result.get("intent", "unknown"), float(result.get("confidence", 0.5))
    except Exception as e:
        print("Gemini classification fallback:", e)
        # fallback keyword method
        keywords = {"live": "is_live", "royalty": "royalty", "copy": "author_copy"}
        for k, v in keywords.items():
            if k in text.lower():
                return v, 0.9
        return "unknown", 0.5

def lookup_supabase(identifier: str):
    """Find author record by email, ISBN, or book."""
    try:
        data = supabase.table("authors").select("*").or_(
            f"email.eq.{identifier},ISBN.eq.{identifier},book.eq.{identifier}"
        ).execute()
        return data.data[0] if data.data else None
    except Exception as e:
        print("Supabase author lookup error:", e)
        return None

def fetch_kb_entries():
    """Fetch all KB entries from Supabase."""
    try:
        data = supabase.table("knowledge_base").select("id, topic, body").execute()
        return data.data or []
    except Exception as e:
        print("Supabase KB error:", e)
        return []

def search_kb(text: str):
    """Semantic KB search using Gemini embeddings."""
    kb_entries = fetch_kb_entries()
    if not kb_entries:
        return None

    query_vec = embed_text(text)
    best_entry, best_score = None, 0

    for entry in kb_entries:
        kb_text = f"{entry['topic']} {entry['body']}"
        kb_vec = embed_text(kb_text)
        score = cosine_similarity(query_vec, kb_vec)
        if score > best_score:
            best_entry, best_score = entry, score

    if best_entry and best_score > 0.7:
        return {
            "response": f"{best_entry['topic']}: {best_entry['body']}",
            "confidence": float(best_score)
        }
    return None

def build_response(intent, record):
    """Generate contextual responses using Supabase data + KB context."""
    if intent == "is_live":
        if record["book_live_date"]:
            return f"Your book '{record['title']}' went live on {record['book_live_date']}."
        return f"Your book '{record['title']}' is not live yet."
    elif intent == "royalty":
        return f"Your royalty status is {record['royalty_status']}."
    elif intent == "author_copy":
        return "Author copies are dispatched 7–14 business days after your book goes live."
    elif intent == "dashboard":
        return "If you cannot access your dashboard, try logging out or requesting a reset."
    elif intent == "login_credentials":
        return "Login credentials are emailed automatically within 2 minutes of payment confirmation."
    else:
        return "I'm not sure. Let me connect you to a human agent."

def escalate(query_text, reason):
    ticket = {
        "ticket_id": f"TKT-{datetime.datetime.now(datetime.timezone.utc).timestamp()}",
        "query": query_text,
        "reason": reason,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    log_interaction(query_text, f"Escalated: {reason}", 0.0)
    return {"escalated": True, "ticket": ticket}

def log_interaction(query, response, confidence):
    log = {
        "query": query,
        "response": response,
        "confidence": confidence,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    with open("logs.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(log) + "\n")

# --- API Endpoint ---
@app.post("/api/query")
async def handle_query(q: Query):
    intent, confidence = classify_intent(q.text)
    if confidence < 0.8:
        # fallback to KB if intent is low-confidence
        kb_hit = search_kb(q.text)
        if kb_hit:
            log_interaction(q.text, kb_hit["response"], kb_hit["confidence"])
            return {"response": kb_hit["response"], "confidence": kb_hit["confidence"], "escalated": False}
        return escalate(q.text, "low_confidence")

    record = lookup_supabase(q.from_)
    if not record:
        kb_hit = search_kb(q.text)
        if kb_hit:
            log_interaction(q.text, kb_hit["response"], kb_hit["confidence"])
            return {"response": kb_hit["response"], "confidence": kb_hit["confidence"], "escalated": False}
        return escalate(q.text, "no_record_found")

    response = build_response(intent, record)
    log_interaction(q.text, response, confidence)
    return {"response": response, "confidence": confidence, "escalated": confidence < 0.8}


#This is not working because i have exeeded the rate limit for my projects but this is a vaild solution if i get acces to the premium version