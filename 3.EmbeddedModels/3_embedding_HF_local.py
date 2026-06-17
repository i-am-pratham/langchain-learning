from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents=[
    "Delhi is capital of India",
    "Mumbai is capital of Maharashtra",
    "Ahemdabad is the capital of Gujrat"
    "Jaipur is the capital of Rajasthan"
]
vector= embedding.embed_documents(documents)
print(vector)