def chunk_text(pages, chunk_size=200, overlap=50):
    """
    Split page-wise text into overlapping chunks
    while preserving the source page number.
    """

    chunks = []

    for page_number, text in pages:

        words = text.split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk = " ".join(words[start:end])

            chunks.append({
                "text": chunk,
                "page": page_number
            })

            start += chunk_size - overlap

    return chunks