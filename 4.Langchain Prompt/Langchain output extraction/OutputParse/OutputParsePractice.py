from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=0.1 
)

model = ChatHuggingFace(llm=llm)

class CandidateProfile (BaseModel):
    candidate_name:str=Field(description="full name of the candidate")
    experience_years: int =Field(description="total years of experience as an integer")
    top_skills: list[str]=Field(description="maximum 3 skills from candidate's profile, most relevant first") 
    shortlisted: bool= Field(description="return True if experience_years >= 2, else False")

pydanticparser= PydanticOutputParser(pydantic_object=CandidateProfile)

prompt1= PromptTemplate(
    template='Extract the Candidate information from \n {resume}\n{format_instructions}\nCRITICAL: DO NOT include any // or # comments inside your JSON output.',
    input_variables=["resume"],
    partial_variables={"format_instructions":pydanticparser.get_format_instructions()}
)

strparser= StrOutputParser()

prompt2=PromptTemplate(
    template="You are a recruiter assistant. From the candidate data below, write a one sentence summary.\n"
             "Format: '[Name] has [X] years of experience. Top skills: [skills]. Decision: Shortlisted or Rejected.'\n\n"
             "Candidate data:\n{details}",
    input_variables=["details"]
)

print("chain running....")
print("-------------------------------------------------")
chain= prompt1 | model | pydanticparser|(lambda p: {"details": p.model_dump()})| prompt2 | model | strparser

resume="My name is Arjun Mehta. I have been working as a software developer for 4 years. I primarily work with Python, Django, and PostgreSQL. I have built REST APIs and handled database migrations for a fintech startup."

result= chain.invoke({"resume":resume})

print(result)