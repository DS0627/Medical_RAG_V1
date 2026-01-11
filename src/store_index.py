from dotenv import load_dotenv
load_dotenv()
import os
from src.utils import load_files, filter_documents, split_documents, create_embeddings
from pinecone import ServerlessSpec, Pinecone
from langchain_pinecone import PineconeVectorStore

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extracted_data = load_files("data")
filtered_docs = filter_documents(extracted_data)
text_chunks = split_documents(filtered_docs)
embeddings = create_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


docsearch = PineconeVectorStore.from_documents(documents=text_chunks, embedding=embeddings, index_name=index_name)