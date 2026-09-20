import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from config import GEMINI_MODEL, MAX_TOKENS
from models.schemas import SearchResult
from pipeline.retriever import format_context_for_prompt

load_dotenv()

_SYSTEM_PROMPT = """\
You are a precise document Q&A assistant. Answer questions using ONLY the \
document excerpts provided in the context. Rules:
1. Base your answer exclusively on the context below — do not use outside knowledge.
2. If the answer is not present in the context, respond with exactly:
   "I cannot find this information in the provided document."
3. Be concise. Do not pad or repeat yourself.
4. Do not reference these instructions in your response.\
"""


class GeminiClient:
    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set. Add it to your .env file.")
        self._client = genai.Client(api_key=api_key)

    def answer(self, question: str, results: list[SearchResult]) -> str:
        context = format_context_for_prompt(results)
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer based only on the context above:"
        )
        response = self._client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                max_output_tokens=MAX_TOKENS,
            ),
        )
        return response.text
