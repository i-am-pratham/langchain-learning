from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

examples = [
    {
        "input":  "gonna be late tmrw",
        "output": "I wanted to inform you that I will be arriving late tomorrow."
    },
    {
        "input":  "can u check this for me",
        "output": "Could you please review this at your convenience?"
    },
    {
        "input":  "thx for ur help",
        "output": "Thank you sincerely for your assistance."
    },
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),   # casual text
    ("ai",    "{output}")   # formal version
])

few_shot = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional tone converter. Rewrite casual text as formal English."),
    few_shot,                    # ← all 3 examples go here automatically
    ("human", "{question}")      # ← the actual user input
])

parser= StrOutputParser()

chain= final_prompt | model | parser

result= chain.invoke({'question': "Buddy give it ASAP!"})
print(result)