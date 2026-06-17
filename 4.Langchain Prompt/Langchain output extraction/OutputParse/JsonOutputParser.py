from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=0.1 # Lower temperature makes the model less likely to hallucinate text
)

model = ChatHuggingFace(llm=llm)

# 1. Define the Strict Schema
class CharacterInfo(BaseModel):
    name: str = Field(description="The name of the character")
    age: str = Field(description="Their age at death, or years lived. Must be a string (e.g., '50 years').")
    city: str = Field(description="The primary city associated with them")

# 2. Give the parser the Pydantic rules
parse = JsonOutputParser(pydantic_object=CharacterInfo)

template = PromptTemplate(
    template='Give me name, age, city of a {character}\n\n{format_instruction}',
    input_variables=['character'],
    partial_variables={'format_instruction': parse.get_format_instructions()}
)

chain = template | model | parse

character = 'Chatrapati Shivaji Maharaj'

print("Running chain...")
result = chain.invoke({"character": character})

print("\n--- VALIDATED JSON ---")
print(result)