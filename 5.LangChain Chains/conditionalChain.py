# 🧑‍💻 Real World Exercise
# Scenario: You're building a support bot for an e-commerce company. A message comes in — your system must:

# Classify the message as either complaint or question
# Route to the right chain:

# If complaint → generate an apology + resolution response
# If question → generate a helpful informational answer



# Requirements:

# classifier_prompt — classifies input as complaint or question (one word output)
# complaint_chain — handles complaints
# question_chain — handles questions
# RunnableLambda for routing
# Test with two inputs:

# "My order has been stuck for 10 days and nobody is helping me!"
# "What are your return policy timelines?"


# Print the result for both



from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
#     task="text-generation",
#     temperature=0.1 
# )

# model = ChatHuggingFace(llm=llm)

class Message(BaseModel):
    category: Literal['complaint','question'] = Field(description='Classify the message')
    original_message: str = Field(description='The original message text')

pydanticparser=PydanticOutputParser(pydantic_object=Message)
stringparser=StrOutputParser()

prompt1 = ChatPromptTemplate([
    ("system", "You are a message classifier. Classify the given message as either 'complaint' or 'question'. Also return the original message as-is.\n\n{format_instructions}"),
    ("human", "{message}")
])

prompt1=prompt1.partial(format_instructions=pydanticparser.get_format_instructions())

classifier_chain= prompt1 | model | pydanticparser


prompt2=ChatPromptTemplate([
("system","You are a customer support specialist. When a customer lodges a complaint, respond with empathy. Acknowledge their issue, provide a sincere apology, and suggest a concrete resolution or next steps. Keep the response professional but warm."),
("human","{message}")
])

prompt3=ChatPromptTemplate([
    ("system","You are a helpful e-commerce customer service representative. "
     "Read the customer's question and answer it directly. "
     "Do not ask for the question again, do not say you are here to help, "
     "and do not ask follow-up questions."),
    ("human","{message}")
])

branch_chain = RunnableBranch(
    (lambda x: x.category == 'complaint', 
     RunnableLambda(lambda x: {"message": x.original_message}) | prompt2 | model | stringparser),
    (lambda x: x.category == 'question',  
     RunnableLambda(lambda x: {"message": x.original_message}) | prompt3 | model | stringparser),
    RunnableLambda(lambda x: "Can't classify the message")
)

print("Chain Running....")
print("___________________________________________________________________-")
chain= classifier_chain | branch_chain

result= chain.invoke({"message":"What are your return policy timelines?"})

print(result)


