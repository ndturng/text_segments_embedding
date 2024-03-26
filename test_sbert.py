from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the SBERT model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)  # There are many other models available

# Your text(s)
texts_1 = ["Once you have your embeddings, you can use them for a \
variety of tasks such as semantic similarity, clustering, or as input \
features for machine learning models. For example, to calculate the cosine \
similarity between two embeddings."]

texts_2 = ["Remember, SBERT models are designed to capture the semantic \
meaning of sentences and paragraphs, making them particularly well-suited \
for tasks involving semantic understanding and similarity."]
# Generate embeddings
embeddings_1 = model.encode(texts_1)
embeddings_2 = model.encode(texts_2)

# Calculate cosine similarity between the first two embeddings
similarity = cosine_similarity([embeddings_1[0]], [embeddings_2[0]])

print(f"Cosine similarity: {similarity[0][0]}")
