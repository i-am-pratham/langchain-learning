# In this we have usecase that 1st time uuser will give topic and lmm will give detailed report. 2nd time this detailed report will again go to llm and it will give 5 line summary of that report.

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

template1= ChatPromptTemplate([
    ('system','You are an Expert Research Analyst and Technical Writer.Your objective is to transform a single topic into a comprehensive, authoritative, and deeply detailed report.  '),
    ('human','Write the detailed report on {topic}')
])

template2=ChatPromptTemplate([
    ('system','You are a Precision Executive Editor specializing in high-level briefs for C-suite leadership. Your sole task is to analyze the provided report and distill it into a strict 5-line summary.'),
    ('human','Write the 5 line summary on the following text. /n {text}')
])

parse= StrOutputParser()

chain= (template1 | model | parse |(lambda report_string:{"text":report_string}) | template2 | model | parse )

topic='Black Hole'


result= chain.invoke({'topic':topic})

print(result)