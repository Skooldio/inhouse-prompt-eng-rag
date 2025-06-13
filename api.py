from fastapi import FastAPI, HTTPException
from chat_agent.basic_llm import get_assistant_response
from chat_agent.llm_retrieval import get_assistant_response as get_rag_assistant_response
from uuid import uuid4
from datetime import datetime, timezone
from dotenv import load_dotenv

from evaluation.ragas_evaluator import evaluate_rag_response
from common.eval_models import EvaluationResponse
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

@app.post("/evaluate")
async def evaluate_response(request: ChatRequest):
    """
    Endpoint that returns both the RAG assistant response and RAGAS evaluation metrics.
    
    This endpoint first gets the RAG assistant's response and then evaluates it using RAGAS metrics.
    The evaluation includes metrics like answer relevancy and context relevancy.
    """
    try:
        response_id = str(uuid4())
        now = datetime.now(timezone.utc)
    
        # Get response from RAG assistant
        assistant_response = await get_rag_assistant_response(request.message)
    
        # Prepare the response
        assistant_msg = Message(content=assistant_response.content, format="text")
        candidate = Candidate(
            message=assistant_msg,
            rag=assistant_response.rag if assistant_response.rag else None,
        )
        
        chat_response = ChatResponse(
            id=response_id,
            created=now,
            candidates=[candidate],
        )
        # Extract contexts for evaluation if available
        contexts = []
        if assistant_response.rag:
            contexts = [doc.page_content for doc in assistant_response.rag.retrieved_documents]
        
        # Evaluate the response using RAGAS (without answer_similarity since we don't have ground truth)
        evaluation = {}
        evaluation_error = None
        
        try:
            print(f"Question: {request.message}")
            print(f"Answer: {assistant_response.content}")
            
            if not contexts:
                raise ValueError("No contexts available for evaluation.")
            
            print("Evaluating response with RAGAS metrics...")
            evaluation = evaluate_rag_response(
                question=request.message,
                answer=assistant_response.content,
                contexts=contexts,
            )
        except Exception as e:
            evaluation_error = f"Evaluation partially failed: {str(e)}"
        
        return EvaluationResponse(
            chat_response=chat_response,
            evaluation=evaluation if evaluation else None,
            evaluation_error=evaluation_error
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)