from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ChatRequest(BaseModel):
    message: str

class Message(BaseModel):
    content: str = Field(description="Text or markdown content of the message")
    format: Optional[str] = Field("text", description="Format of the content, e.g., text, markdown, html")

class RAGDocument(BaseModel):
    id: str
    source: str
    title: Optional[str]
    score: Optional[float]
    page_content: str

class RAGOutput(BaseModel):
    retrieved_documents: List[RAGDocument]

class Candidate(BaseModel):
    message: Optional[Message]
    rag: Optional[RAGOutput]

class ChatResponse(BaseModel):
    id: str
    created: datetime
    candidates: List[Candidate]

class AssistantResponse(BaseModel):
    content: str
    rag: Optional[RAGOutput] = None
