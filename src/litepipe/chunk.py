# 1) Split på headers (giver Documents m/ metadata om headers)
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
)
docs = header_splitter.split_text(text)  # -> list[Document]

# 2) Split videre til max-size (bevarer metadata)
rc_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=100)
chunks = rc_splitter.split_documents(docs)  # -> list[Document]

# Brug returværdien:
for d in chunks:
    print(d.metadata)       # fx {"h1": "Title", "h2": "Section A"}
    print(d.page_content)   # selve chunk-teksten
    print("----")