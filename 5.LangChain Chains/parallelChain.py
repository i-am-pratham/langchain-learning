from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
#     task="text-generation",
#     temperature=0.1 
# )

# model = ChatHuggingFace(llm=llm)
parser= StrOutputParser()


prompt1= ChatPromptTemplate([
    ("system","You are a sentiment analysis expert."),
    ("human","Analyze the sentiment and only tell Positive, Negative of this Feedback:\n{feedback}")
])

prompt2= ChatPromptTemplate([
    ("system","You are a professional customer support agent.Respond in 2-3 lines only."),
    ("human","Write a short reply to this customer feedback:\n {feedback}")
])

prompt3= ChatPromptTemplate([
    ("system","You are a product manager who identifies issues from customer feedback.Respond in 2-3 lines only."),
    ("human","Give one action item for the product team based on: \n{feedback}")
])


parallel_chain=RunnableParallel({
     "sentiment":prompt1 | model | parser,
    "reply":prompt2| model | parser,
    "action":prompt3| model | parser
})

result= parallel_chain.invoke({'feedback':"The checkout process is painfully slow and I gave up on my order."})

print(result)
parallel_chain.get_graph().print_ascii()