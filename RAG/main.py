
from pathlib import Path
import ollama

# STEP 1: Read the document
BASE_DIR = Path(__file__).parent
FILE_PATH = BASE_DIR / "personal_profile.txt"

document_text = FILE_PATH.read_text(encoding="utf-8")

print("Document loaded successfully!")
print("Total characters:", len(document_text))


# STEP 2: Split the document into chunks
def create_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


chunks = create_chunks(document_text)

print("\nChunking completed!")
print("Total chunks:", len(chunks))


# STEP 3: Generate embeddings for each chunk
EMBEDDING_MODEL = "nomic-embed-text"

embeddings = []

for index, chunk in enumerate(chunks, start=1):
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=chunk
    )

    vector = response["embeddings"][0]
    embeddings.append(vector)

    print(f"Generated embedding for chunk {index}")

print("\nEmbedding generation completed!")
print("Total embeddings:", len(embeddings))

if embeddings:
    print("Vector dimensions:", len(embeddings[0]))
    print("First 10 vector values:", embeddings[0][:10])



# STEP 4: Store chunks and embeddings in ChromaDB

import chromadb

# Create a persistent database inside the project folder
DB_PATH = BASE_DIR / "chroma_db"

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

# Create or access a collection
collection = client.get_or_create_collection(
    name="personal_profile"
)

# Give each chunk a unique ID
ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]

# Store text, embeddings, and IDs
collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)

print("\nIndexing completed successfully!")
print("Chunks stored in ChromaDB:", collection.count())
print("Database location:", DB_PATH)


# STEP 5: Verify the stored data

print("\n--- Verifying ChromaDB ---")

# Retrieve records from the collection
stored_data = collection.get(
    include=["documents", "embeddings"]
)

# Check the number of stored chunks
print("Total stored IDs:", len(stored_data["ids"]))
print("Total stored documents:", len(stored_data["documents"]))
print("Total stored embeddings:", len(stored_data["embeddings"]))

# Display the first stored chunk
if stored_data["documents"]:
    print("\nFirst stored chunk:")
    print(stored_data["documents"][0])

# Display the first stored embedding's dimensions
if stored_data["embeddings"] is not None and len(stored_data["embeddings"]) > 0:
    print(
        "\nFirst stored embedding dimensions:",
        len(stored_data["embeddings"][0])
    )