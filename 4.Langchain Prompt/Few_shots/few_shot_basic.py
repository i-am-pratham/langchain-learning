from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

examples=[
    {"english": "Good morning", "hindi": "Shubh Prabhat"},
    {"english": "Thank you",    "hindi": "Dhanyavaad"},
    {"english": "My name is Prathamesh", "hindi": "Mera naam Prathamesh hai"}
]

example_prompt=PromptTemplate(
    template="English:{english}\nHindi:{hindi}",
    input_variables=['english','hindi']
)

prompt=FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix='Translate English to Hindi:',
    suffix='English: {english}\n Hindi:',
    input_variables=['english']
)
parser= StrOutputParser()

chain= prompt | model | parser

result= chain.invoke({'english': "Hii buddy"})
print(result)