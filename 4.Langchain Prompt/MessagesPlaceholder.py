# MemoryBot

#  we will use MessagesPlaceholder for this

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

# model= ChatGoogleGenerativeAI(model='gemini-2.5-flash')


prompt= ChatPromptTemplate([
    ("system","You are a pirate. Speak only like a pirate."),
    MessagesPlaceholder('chat_history'),
    ("human","{question}")
])

chat_history=[]
chain= prompt | model

while True:
    user_input= input("Enter message: ")

    if user_input=='exit':
        break
    chat_history.append(HumanMessage(content=user_input))

    

    result=chain.invoke({
       "chat_history":chat_history,
        "question": user_input
    })
    chat_history.append(AIMessage(content=result.content))
    print("AI",result.content)

print(chat_history)

