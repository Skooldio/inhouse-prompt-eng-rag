from common.chat_models import AssistantResponse, RAGOutput, RAGDocument
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from google.oauth2 import service_account
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma
import google.generativeai as genai

from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")

saCredentials = service_account.Credentials.from_service_account_file(
    filename="./service_account.json"
)

embeddings = VertexAIEmbeddings(model="text-embedding-004", credentials=saCredentials)


vector_store = Chroma(
    collection_name="rag_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# llm = init_chat_model(
#     "gemini-2.0-flash", model_provider="google_vertexai", credentials=saCredentials
# )

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible and format into bullet point.
Always give the reference of the answer with the source and summary of related documents.

## Important:
- Answer in the same language as the question.

{context}

Question: {question}

Helpful Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10})


async def get_assistant_response(user_prompt: str):
    retrieved_docs = retriever.invoke(user_prompt)

    print(f"Retrieved {len(retrieved_docs)} documents for the query: {user_prompt}")

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = custom_rag_prompt.invoke(
        {"question": user_prompt, "context": docs_content}
    )

    response_msg = model.generate_content(contents=prompt.text).text

    rag_documents = []
    for doc in retrieved_docs:
        # Extract metadata with fallbacks
        metadata = getattr(doc, "metadata", {})
        rag_documents.append(
            {
                "id": metadata.get("id", str(hash(doc.page_content))),
                "source": metadata.get("source", "unknown"),
                "title": metadata.get("title", metadata.get("source", "Document")),
                "score": (
                    float(metadata.get("score", 0.0)) if "score" in metadata else None
                ),
                "page_content": doc.page_content.replace("\n", " "),
            }
        )
    rag_output = RAGOutput(
        retrieved_documents=[RAGDocument(**doc) for doc in rag_documents]
    )
    return AssistantResponse(
        content=response_msg,
        rag=rag_output,  # Assuming no RAG output for this basic example
    )
