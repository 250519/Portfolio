import os
import uuid
from dotenv import load_dotenv
from langchain_docling import DoclingLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone

load_dotenv()

# 1. Load document
FILE_PATH = "/Users/harshshivhare/Portfolio/About me.pdf"
loader = DoclingLoader(file_path=FILE_PATH)
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "],
    length_function=len
)
chunks = splitter.split_documents(docs)

# 3. Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# 4. Initialize Pinecone (V2)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
print("api:key", os.getenv("PINECONE_API_KEY"))
print("Pinecone initialized successfully.")
# Option 1: Use index name (recommended approach)
# INDEX_NAME = "portfolio"  # Replace with your actual index name
# index = pc.Index(INDEX_NAME)

# Option 2: If you must use host, fix the format (remove https://)
index = pc.Index(
    host="portfolio-umh37tw.svc.aped-4627-b74a.pinecone.io"
)

# 5. Prepare all vectors at once
upsert_data = []
for i, doc in enumerate(chunks):
    text = doc.page_content.strip()
    if not text:
        continue  # skip empty chunks
    vector = embedding_model.embed_query(text)
    
    upsert_data.append({
        "id": f"chunk-{i}-{uuid.uuid4()}",
        "values": vector,
        "metadata": {
            "text": text,
            "source": "about_me.pdf",
            "document_type": "portfolio",
            "chunk_index": i
        }
    })

# 6. Upsert into Pinecone (single call)
try:
    index.upsert(vectors=upsert_data)
    print("✅ All chunks embedded and uploaded to Pinecone in a single upsert.")
except Exception as e:
    print(f"❌ Error uploading to Pinecone: {e}")
    print("Check your API key, index name, and network connection.")