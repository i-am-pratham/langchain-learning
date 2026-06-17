from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    path='my_docs',
    glob= "**/*.txt",
    loader_cls=TextLoader,
    show_progress=True
)


# docs= loader.load()
# print("Total document loaded",len(docs))

# for doc in docs:
#     print(doc.page_content)
#     print('-'*50)
#     print(doc.metadata)
#     print('-'*50)


for doc in loader.lazy_load():
    print(doc.page_content)
