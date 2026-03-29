import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")

app = FastAPI()


class DataInput(BaseModel):
    summary: str


class ChatInput(BaseModel):
    question: str
    context: str


def call_openrouter(system_prompt, user_prompt):
    if not API_KEY:
        return {"error": "Missing API key. Set OPENROUTER_API_KEY or API_KEY in your .env file."}

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=60,
    )
    response.raise_for_status()

    result = response.json()
    choices = result.get("choices", [])
    if not choices:
        return {"error": result}

    message = choices[0].get("message", {})
    return {"content": message.get("content", "").strip()}


@app.post("/insights")
def generate_insights(data: DataInput):
    system_prompt = (
        "You are Vyntaq, a business data analyst. "
        "Use only the dataset context provided. "
        "Do not invent columns, values, or trends that are not supported by the context. "
        "Keep the answer concise, specific, and action-oriented."
    )
    user_prompt = f"""
Dataset context:
{data.summary}

Provide:
1. Key trends
2. Anomalies or outliers
3. Business recommendations

Format:
- Use short bullet points
- Mention metric names when possible
"""

    try:
        result = call_openrouter(system_prompt, user_prompt)
        if "content" in result:
            return {"insight": result["content"]}
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/chat")
def chat_with_data(data: ChatInput):
    system_prompt = (
        "You are Vyntaq, a sharp analyst for tabular datasets. "
        "Answer only from the provided dataset context. "
        "Never guess missing values or invent columns. "
        "Never use a row index as the main answer unless the user explicitly asks about the index. "
        "Use the evidence in this order: exact text matches, grouped summaries, top or bottom rows, then sample rows. "
        "If exact text matches exist, never claim the entity is missing. "
        "If the question asks for the top, best, highest, lowest, or most popular item, "
        "identify the relevant metric and grouping column from the context and state both clearly. "
        "If grouped summaries are provided for a label, category, or grouping column, "
        "prefer those summaries for ranking or group-level questions. "
        "If multiple exact-match rows exist, summarize the metric range, average, and latest value when a date is present. "
        "If the context is still insufficient, say exactly what is missing."
    )
    user_prompt = f"""
Dataset context:
{data.context}

User question:
{data.question}

Response requirements:
- Answer in natural language
- Be specific and cite the metric used
- Keep it short and confident
"""

    try:
        result = call_openrouter(system_prompt, user_prompt)
        if "content" in result:
            return {"response": result["content"]}
        return result
    except Exception as e:
        return {"error": str(e)}
