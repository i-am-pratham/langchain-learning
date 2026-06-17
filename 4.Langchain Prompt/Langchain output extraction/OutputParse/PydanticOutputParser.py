from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=0.1 
)

model = ChatHuggingFace(llm=llm)

class InterviewScore(BaseModel):
    final_score:int= Field(description="must be an integer representing a score out of 100")
    action_items: list[str]= Field(description="must be a list of strings containing things the student needs to practice")

parser= PydanticOutputParser(pydantic_object=InterviewScore)

prompt=PromptTemplate(
    template='Extract the student information from evaluation paragraph.\n{evaluation_text}\n\n{format_instructions}\nCRITICAL: DO NOT include any // or # comments inside your JSON output.',
    input_variables=['evaluation_text'],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)

chain= prompt | model | parser

evaluation_text = """
The candidate had a solid fundamental understanding of REST APIs. However, 
they struggled significantly when asked about authentication tokens like JWTs. 
They also spoke a bit too fast during the system design portion. Overall, 
I would give them a 78 out of 100. They need to review OAuth2 flows and 
practice speaking at a more measured pace.
"""
print("Running chain...")

result= chain.invoke({"evaluation_text":evaluation_text})

print(result)