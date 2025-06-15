import os
import json
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env file.")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

MODEL = "llama3-70b-8192"
ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

def call_groq(prompt):
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(ENDPOINT, headers=HEADERS, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def detect_refusal(text):
    refusal_phrases = [
        "i'm sorry", "i cannot", "as an ai", "i am unable", "it would be inappropriate",
        "i cannot comply", "i don't think", "i apologize", "not allowed", "it's not possible"
    ]
    return any(phrase in text.lower() for phrase in refusal_phrases)

def detect_format_bullet(text):
    lines = text.strip().split("\n")
    bullet_lines = [line for line in lines if line.strip().startswith(("-", "*", "•"))]
    return len(bullet_lines) >= 3

# Load prompts
with open("evals_dataset_v1_2.json", "r") as f:
    prompts = json.load(f)

results = []

for idx, entry in enumerate(prompts):
    prompt_text = entry["input"]
    try:
        start = time.time()
        response = call_groq(prompt_text)
        end = time.time()
        results.append({
            "prompt_id": idx,
            "category": entry.get("category", ""),
            "model_name": MODEL,
            "input": prompt_text,
            "response": response,
            "refusal": detect_refusal(response),
            "format_compliance": detect_format_bullet(response) if "bullet" in prompt_text.lower() else None,
            "response_time_sec": round(end - start, 2),
            "is_simulation": False
        })
        print(f"[✓] Prompt {idx} processed.")
    except Exception as e:
        print(f"[✗] Error on prompt {idx}: {e}")
        results.append({
            "prompt_id": idx,
            "category": entry.get("category", ""),
            "model_name": MODEL,
            "input": prompt_text,
            "response": f"ERROR: {e}",
            "refusal": None,
            "format_compliance": None,
            "response_time_sec": 0,
            "is_simulation": False
        })

# Save output
df = pd.DataFrame(results)
df.to_csv("eval_scores_output_v1_2.csv", index=False)
print("\n[✓] Evaluation complete. Results saved to eval_scores_output_v1_2.csv.")
