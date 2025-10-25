import re

def chunk_faq_text(raw_text):
    entries = re.split(r'_{2,}', raw_text)
    cleaned = []
    for e in entries:
        e = e.strip()
        if e:
            lines = e.splitlines()
            header = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            cleaned.append({"topic": header, "body": body})
    return cleaned

# Usage
with open("/Users/akki/Desktop/AKKI/Presonal Projects/Customer Query Bot/KB(raw).txt", "r", encoding="utf-8") as f:
    raw = f.read()

kb_entries = chunk_faq_text(raw)
print(kb_entries[:3])

from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()


# --- Config ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) # type: ignore

for entry in kb_entries:
    supabase.table("knowledge_base").insert({
        "topic": entry["topic"],
        "body": entry["body"]
    }).execute()