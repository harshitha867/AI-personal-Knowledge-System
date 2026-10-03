import faiss
import numpy as np
from embeddings import model


def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(embeddings).astype("float32")
    )

    return index


def search_faiss(
    query,
    chunks,
    index,
    top_k=3
):

    query_embedding = model.encode([query])

    k = min(top_k, len(chunks))

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        k
    )

    results = []

    for idx, distance in zip(indices[0], distances[0]):

        results.append(
            (
                chunks[idx],
                distance
            )
        )

    return results