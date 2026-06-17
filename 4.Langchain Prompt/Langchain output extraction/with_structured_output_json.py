from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", vertexai=False, temperature=0.0)

# 1. Define the blueprint as a raw JSON Schema dictionary
movie_schema = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string", 
            "description": "The exact name of the film"
        },
        "release_year": {
            "type": "integer", 
            "description": "The calendar year the movie came out"
        },
        "genres": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of genres matching the movie"
        },
        "maturity_rating": {
            "type": "string",
            "enum": ["G", "PG", "PG-13", "R", "NC-17"],
            "description": "The official age rating classification"
        }
    },
    # Tell the model which fields are absolutely non-negotiable
    "required": ["title", "release_year", "maturity_rating"]
}

# 2. Bind the raw dictionary directly to the model
structural_model = model.with_structured_output(movie_schema)

prompt = ChatPromptTemplate.from_messages([
    ('system', "You are an automated entertainment archival assistant. Extract metadata cleanly."),
    ('human', "Process this raw text block:\n\n{raw_text}")
])

chain = prompt | structural_model

sample_input = "We just finished screening Christopher Nolan's Oppenheimer (2023). It's a heavy historical biography drama, definitely earned its R rating for the language and intensity."

result = chain.invoke({"raw_text": sample_input})

print("--- JSON SCHEMA DICTIONARY OUTPUT ---")
print(type(result)) # This returns a standard Python dictionary natively!
print(result)