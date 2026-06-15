from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def generate_token_embeddings(tokens):

    embeddings = model.encode(tokens)

    results = []

    for token, vector in zip(tokens, embeddings):

        results.append({
            "token": token,
            "embedding": [round(float(x), 4) for x in vector[:10]]
        })

    return results