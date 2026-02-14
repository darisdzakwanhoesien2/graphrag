def simple_chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_documents(documents):
    all_chunks = []

    for doc in documents:
        split_chunks = simple_chunk_text(doc["text"])

        for i, chunk in enumerate(split_chunks):
            all_chunks.append({
                "chunk_id": f"{doc['doc_id']}_p{doc['page']}_c{i}",
                "doc_id": doc["doc_id"],
                "page": doc["page"],
                "text": chunk
            })

    return all_chunks


# from langchain.text_splitter import RecursiveCharacterTextSplitter

# def chunk_documents(documents):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=150
#     )

#     chunks = []

#     for doc in documents:
#         splits = splitter.split_text(doc["text"])

#         for i, chunk in enumerate(splits):
#             chunks.append({
#                 "chunk_id": f"{doc['doc_id']}_p{doc['page']}_c{i}",
#                 "doc_id": doc["doc_id"],
#                 "page": doc["page"],
#                 "text": chunk
#             })

#     return chunks