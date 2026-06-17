from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding= GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",dimensions=32)

documents=[
    "Delhi is capital of India",
    "Mumbai is capital of Maharashtra",
    "Ahemdabad is the capital of Gujrat",
    "Jaipur is the capital of Rajasthan"
]

result= embedding.embed_documents(documents)
print(str(result))