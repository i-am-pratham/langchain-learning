from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=0.1 # Lower temperature makes the model less likely to hallucinate text
)

model = ChatHuggingFace(llm=llm)

name_schema=[
    ResponseSchema(name="technical_strength",description="The candidate's strongest technical skill mentioned"),
    ResponseSchema(name='communication_flaw', description='Any negative feedback regarding how they speak or present'),
    ResponseSchema(name='hire_decision', description="Strictly output 'Yes' or 'No'")
]

parser= StructuredOutputParser.from_response_schemas(name_schema)

prompt=PromptTemplate(
    template="Give the technical_strength,communication_flaw and hire_decision from {interview_notes}\n{format_instruction}",
    input_variables=["interview_notes"],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

chain= prompt | model | parser

interview_notes="""
The candidate showed up on time. When asked about database optimization, 
they absolutely nailed the explanation of B-Trees and indexing. However, 
they mumbled a lot and rarely made eye contact when explaining their code. 
Given the client-facing nature of this role, I'm going to have to pass on them.
"""

print("Running chain...")
result=chain.invoke({"interview_notes":interview_notes})

print("________________________________________")
print(result)