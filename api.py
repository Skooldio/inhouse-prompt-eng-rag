from fastapi import FastAPI
from chat_agent.basic_llm import get_assistant_response
from uuid import uuid4
from datetime import datetime, timezone
from dotenv import load_dotenv
from common.chat_models import (
    ChatRequest,
    ChatResponse,
    Message,
    Candidate,
)

load_dotenv()

app = FastAPI()

@app.post("/messages")
async def messages(request: ChatRequest):
    response_id = str(uuid4())
    now = datetime.now(timezone.utc)
    assistant_response = await get_assistant_response(request.message)
    assistant_msg = Message(content=assistant_response.content, format="text")
    candidate = Candidate(
        message=assistant_msg,
        rag=assistant_response.rag if assistant_response.rag else None,
    )

    response = ChatResponse(
        id=response_id,
        created=now,
        candidates=[candidate],
    )

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)