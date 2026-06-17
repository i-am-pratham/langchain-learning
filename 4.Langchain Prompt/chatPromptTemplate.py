from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain_co
from dotenv import load_dotenv

load_dotenv()

model= ChatGoogleGenerativeAI(model='gemini-2.5-flash')


# template = ChatPromptTemplate([
#     ("system", "You are a {role}. Always respond in {language}."),
#     ("human", "{question}")
# ])

# prompt = template.invoke({
#     "role": "cricket expert",
#     "language": "simple English",
#     "question": "What is a googly?"
# })

# chain = template | model
# result = chain.invoke({
#     "role": "cricket expert",
#     "language": "simple English",
#     "question": "What is a googly?"
# })
# print(result)
# print(prompt.to_messages())

template= ChatPromptTemplate([
    ("system","AI is a {language} language tutor who teaches at {level} level"),
    ("human","{query}")
])

chain= template| model
result= chain.invoke({
    "language" : "Japanese",
    "level" : "beginner",
    "query" : "How do I say 'Where is the train station?' in Japanese?"
})

print(result.content)

# prompt is a ChatPromptValue — a list of formatted messages
# ready for model.invoke()