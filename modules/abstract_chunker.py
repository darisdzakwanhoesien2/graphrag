def chunk_abstracts(documents):

    chunks = []

    for i, doc in enumerate(documents):

        chunks.append({
            "chunk_id": f"{doc['source_folder']}_paper_{i}",
            "doc_id": doc["source_folder"],
            "title": doc["title"],
            "doi": doc["doi"],
            "authors": doc["authors"],
            "journal": doc["journal"],
            "year": doc["year"],
            "cited_by": doc["cited_by"],
            "tags": doc["tags"],
            "text": doc["text"]
        })

    return chunks


# def chunk_abstracts(documents):

#     chunks = []

#     for i, doc in enumerate(documents):
#         chunks.append({
#             "chunk_id": f"{doc['doc_id']}_paper_{i}",
#             "doc_id": doc["doc_id"],
#             "title": doc["title"],
#             "doi": doc["doi"],
#             "authors": doc["authors"],
#             "journal": doc["journal"],
#             "year": doc["year"],
#             "text": doc["text"]
#         })

#     return chunks
