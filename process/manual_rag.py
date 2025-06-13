from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma
from google.oauth2 import service_account

saCredentials = service_account.Credentials.from_service_account_file(
    filename="./service_account.json"
)

embeddings = VertexAIEmbeddings(model="text-embedding-004", credentials=saCredentials)


vector_store = Chroma(collection_name="rag_collection",
                      embedding_function=embeddings,
                      persist_directory="./chroma_langchain_db")

file_path = "./eBook-How-to-Build-a-Career-in-AI.pdf"


loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)

all_splits = text_splitter.split_documents(docs)

document_ids = vector_store.add_documents(documents=all_splits)