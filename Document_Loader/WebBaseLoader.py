from langchain_community.document_loaders import WebBaseLoader

loader=WebBaseLoader([
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    "https://thespiderman.news.blog/articles/"
])

# loader=WebBaseLoader("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
docs=loader.load()
print(docs)
print("---------------------------------------")
print(f'Total Document loaded : {len(docs)}')
print("---------------------------------------")
print(f'First 500 chars of page_content: {docs[0].page_content[:500]}')
print("---------------------------------------")

print(f'Metadata: {docs[0].metadata}')
