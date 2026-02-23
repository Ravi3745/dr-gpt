import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "doc-gpt")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

BOOK_NAME = "Hemergency.pdf"   # change this to the file you want to delete

print(f"Deleting all vectors where source = {BOOK_NAME}")

index.delete(
    filter={"source": BOOK_NAME}
)

print("Deletion completed.")