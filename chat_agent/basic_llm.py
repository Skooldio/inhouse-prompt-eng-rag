import random
from common.chat_models import AssistantResponse
import google.generativeai as genai
import os

GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-2.0-flash')

async def get_assistant_response(prompt: str):
    # Simulate an asynchronous call to an LLM
    response_msg = model.generate_content(
        contents=prompt
    ).text

    return AssistantResponse(
      content=response_msg,
      rag=None  # Assuming no RAG output for this basic example
    )