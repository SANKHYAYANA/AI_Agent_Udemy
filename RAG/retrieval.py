
import chromadb
from pathlib import Path
import ollama

# STEP 1: Connect to the existing ChromaDB database

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "chroma_db"

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = client.get_collection(
    name="personal_profile"
)

print("Connected to ChromaDB successfully!")
print("Total stored chunks:", collection.count())


# STEP 2: Check the collection configuration

print("\n--- Collection Configuration ---")

config = collection.configuration
print(config)

hnsw_config = config.get("hnsw", {})
distance_metric = hnsw_config.get("space", "Not found in HNSW config")

print("\nDistance metric:", distance_metric)

if distance_metric == "cosine":
    print("ChromaDB is using cosine distance.")
elif distance_metric == "l2":
    print("ChromaDB is using squared Euclidean (L2) distance.")
elif distance_metric == "ip":
    print("ChromaDB is using inner product distance.")
else:
    print("Check the configuration output to identify the metric.")


# STEP 3: Get the user's question

question = input("\nAsk a question about the profile: ")
#
#
# STEP 4: Convert the question into an embedding

response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

question_embedding = response["embeddings"][0]
#
#
# STEP 5: Search for the most relevant chunks

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3,
    include=["documents", "distances"]
)
#
#
# STEP 6: Display the retrieved chunks and their distances

print("\n--- Retrieved Results ---")

documents = results["documents"][0]
distances = results["distances"][0]

for i, (document, distance) in enumerate(
    zip(documents, distances), start=1
):
    print(f"\nResult {i}:")
    print(document)
    print("Distance:", distance)

print("\nRetrieval completed!")


# STEP 7: AUGMENTATION
print("********************************************Augmentation************************************ ")
# Combine all retrieved chunks into one context
context = "\n\n".join(documents)

# Combine the context with the original question
augmented_prompt = f"""
You are an AI assistant answering questions about a personal profile.

Use the following context to answer the user's question.

CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Answer using the provided context.
- Do not invent personal information.
- If the answer is not available in the context,
  say that you do not have enough information.
"""

print("\n--- Augmented Prompt ---")
print(augmented_prompt)

print("\nAugmentation completed!")

# STEP 8: GENERATION

response = ollama.chat(
    model="llama3.1:latest",
    messages=[
        {
            "role": "user",
            "content": augmented_prompt
        }
    ]
)

answer = response["message"]["content"]

print("\n--- AI Answer ---")
print(answer)

print("\nGeneration completed!")
