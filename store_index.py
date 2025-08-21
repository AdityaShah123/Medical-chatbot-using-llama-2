from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")

# print("PINECONE_API_KEY:", PINECONE_API_KEY)
# print("PINECONE_ENVIRONMENT:", PINECONE_ENVIRONMENT)

extracted_data = load_pdf("data/")

# extracted_data

text_chunks = text_split(extracted_data)

# print("length of my chunk:", len(text_chunks))
# text_chunks

embeddings = download_hugging_face_embeddings()

# embeddings
# query_result = embeddings.embed_query("Hello world")
# print("Length", len(query_result))

index_name = "medical-chatbot"
pc = Pinecone(api_key="pcsk_5vV1cz_GXRV64yx98MJYrSu7kX8LWQfnAw28e6qdCtPdMxk5SfWDy3z5A46ZVtaVyR4bzQ")

if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # if using all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

docsearch = PineconeVectorStore.from_texts(
    [t.page_content for t in text_chunks],
    embedding=embeddings,
    index_name=index_name
)