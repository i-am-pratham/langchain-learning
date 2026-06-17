# This is the mini project in which we will find which query ia matching to whivh sentence . This will be foundation to RAG

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

document = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query='Tell me information about bumra'

doc_embedding=embedding.embed_documents(document)
query_embedding=embedding.embed_query(query)

scores=cosine_similarity([query_embedding],doc_embedding)[0]

index, score =sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(document[index])
print(score)