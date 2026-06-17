from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='WA_Fn-UseC_-Telco-Customer-Churn.csv')

docs=loader.load()

print(f'Total documents loaded : {len(docs)}')
print(type(docs))
print(type(docs[0]))
print(type(docs[0].metadata))
print(type(docs[0].page_content))

print("--------------------------------------------")

print(f'Page content of first row : {docs[0].page_content}')
print("--------------------------------------------")
print(f'Page content of last row : {docs[-1].page_content}')
print("--------------------------------------------")
print(f'Metadata of first row : {docs[0].metadata}')


for doc in docs:
    if "Bank transfer (automatic)" in doc.page_content:
        print(doc.page_content)
        print("--------------------------------------------------")


