# Build sequential_chain.py — a 2-step content pipeline:
# Step 1: Takes {topic} → generates a short 3-line blog post intro
# Step 2: Takes the blog post intro → generates a catchy Twitter/X post based on it (max 280 characters)
# Requirements:

# Two separate ChatPromptTemplate — one per step
# Lambda bridge between them with correct variable name
# StrOutputParser after each model call
# Single chain.invoke() call
# Test with topic = "Generative AI in 2025"
# Print the final tweet

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=0.1 
)

model = ChatHuggingFace(llm=llm)
parser= StrOutputParser()

prompt1=PromptTemplate(
    template="Generates a short 3-line blog post intro based on:\n{text}",
    input_variables=['text']
)

prompt2=PromptTemplate(
    template=" Generates a single catchy Twitter/X post (max 280 characters) based on:\n{blog}",
    input_variables=['blog']
)


chain= prompt1 | model | parser | (lambda x:{'blog':x}) | prompt2 | model | parser

result= chain.invoke({"text":"Generative AI in 2025"})

print(result)