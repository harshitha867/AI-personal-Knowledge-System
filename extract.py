import fitz
from chunking import chunk_text
from embeddings import generate_embeddings
from search import semantic_search
from faiss_search import create_faiss_index
from faiss_search import search_faiss


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file while preserving page numbers.
    """

    pages = []

    try:
        doc = fitz.open(pdf_path)

        for page_number, page in enumerate(doc, start=1):

            text = page.get_text()

            if text.strip():
                pages.append((page_number, text))

        doc.close()

        return pages

    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return None


if __name__ == "__main__":

    pdf_path = "uploads/sample.pdf"

    pages = extract_text_from_pdf(pdf_path)

    if pages:

        # Create chunks
        chunks = chunk_text(pages)

        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]

        embeddings = generate_embeddings(texts)

        faiss_index = create_faiss_index(embeddings)

        print(f"\nTotal Pages: {len(pages)}")
        print(f"\nTotal Chunks: {len(chunks)}")

        print("\nEmbedding Shape:")
        print(embeddings.shape)

        print("\nFirst Chunk:\n")
        print(chunks[0]["text"])

        print("\nPage of First Chunk:")
        print(chunks[0]["page"])

        print("\nWords in First Chunk:")
        print(len(chunks[0]["text"].split()))

        print("\nWords in Last Chunk:")
        print(len(chunks[-1]["text"].split()))

        # Semantic Search Test
        query = "What is normal distribution?"

        results = semantic_search(
            query,
            texts,
            embeddings,
            top_k=3
        )

        print("\nQuery:")
        print(query)

        print("\nTop 3 Results:\n")

        for i, (result, score) in enumerate(results, start=1):

            print(f"\nResult {i}")
            print(f"Similarity Score: {score:.4f}")

            print("\nRetrieved Chunk:\n")
            print(result[:500])

            print("\n" + "-" * 80)

        print("\n")
        print("=" * 80)
        print("FAISS SEARCH RESULTS")
        print("=" * 80)

        faiss_results = search_faiss(
            query,
            texts,
            faiss_index,
            top_k=3
        )

        for i, (result, distance) in enumerate(faiss_results, start=1):

            print(f"\nResult {i}")
            print(f"Distance: {distance:.4f}")

            print("\nRetrieved Chunk:\n")
            print(result[:500])

            print("\n" + "-" * 80)

    else:
        print("Failed to extract text from PDF.")