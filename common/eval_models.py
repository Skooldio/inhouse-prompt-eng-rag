from pydantic import BaseModel
from typing import Dict, Optional

from common.chat_models import ChatResponse

class EvaluationResponse(BaseModel):
    """Response model for the evaluation endpoint."""
    chat_response: ChatResponse
    evaluation: Optional[Dict[str, float]] = None
    evaluation_error: Optional[str] = None