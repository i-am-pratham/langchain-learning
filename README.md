# LangChain Learning Journey 🚀

Documenting my hands-on journey into Generative AI — learning LangChain from the ground up, one component at a time, with the goal of building production-grade RAG pipelines and AI agents.

I'm a fresh CS graduate currently transitioning into an **AI Backend Engineer / GenAI Developer** role. This repo is my proof of work — every file here is code I wrote and tested myself while learning.

📌 Posting my daily progress on [LinkedIn](https://linkedin.com/in/prathamesh-sarode2004) — follow along if you're learning this too.

---

## 📂 Repository Structure

```
langchain-learning/
├── 1.LLMs/                          → Basic LLM invocation
├── 2.ChatModels/                    → Gemini, HuggingFace (Cloud + Local)
├── 3.EmbeddedModels/                → Embeddings + Semantic Search
├── 4.Langchain Prompt/
│   ├── Few_shots/                   → FewShotPromptTemplate, FewShotChatMessagePromptTemplate
│   └── Langchain output extraction/
│       └── OutputParse/             → Str, Json, Pydantic, Structured parsers
├── 5.LangChain Chains/              → Sequential, Parallel, Conditional, Custom chains
├── Document_Loader/                 → TextLoader, PDFLoader, CSVLoader, WebBaseLoader
├── requirements.txt
└── .gitignore
```

---

## ✅ Topics Covered So Far

### 1. Models
- LLMs vs Chat Models — when to use which
- Google Gemini, HuggingFace (Inference API + Local), OpenAI
- Embedding Models + Cosine Similarity for semantic search

### 2. Prompts
- `PromptTemplate` with save/load as JSON
- `ChatPromptTemplate` with system/human/AI roles
- `MessagesPlaceholder` — building a memory-aware chatbot
- Few-Shot Prompting — `FewShotPromptTemplate` & `FewShotChatMessagePromptTemplate`

### 3. Structured Output
- `with_structured_output()` — TypedDict, Pydantic, raw JSON Schema
- Function Calling vs JSON Mode under the hood

### 4. Output Parsers
- `StrOutputParser`, `JsonOutputParser`
- `PydanticOutputParser` — validated structured extraction
- `StructuredOutputParser` (legacy)

### 5. Chains & LCEL
- Simple Chain, Sequential Chain, Parallel Chain
- Conditional Chain — `RunnableBranch` based routing
- Custom Chain — `RunnableLambda` for DB lookups / business logic
- Runnable Primitives — `RunnableSequence`, `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`, `RunnableBranch`

### 6. Document Loaders (RAG — in progress)
- `TextLoader`, `PyPDFLoader`, `CSVLoader`, `WebBaseLoader`

### 🔜 Coming Next
- Text Splitters → Vector Stores → Retrievers
- Full RAG pipeline (PDF → Embed → Store → Retrieve → Answer)
- LangGraph (agentic workflows, state management)

---

## 🛠️ Tech Stack

- **Language:** Python
- **Framework:** LangChain, LangChain-Google-GenAI, LangChain-HuggingFace
- **LLMs:** Google Gemini, Meta Llama (via HuggingFace), OpenAI
- **Validation:** Pydantic
- **Planned:** ChromaDB, FAISS, FastAPI, Streamlit

---

## 🚀 Setup

```bash
git clone https://github.com/i-am-pratham/langchain-learning.git
cd langchain-learning
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file in the root with your API keys:
```
GOOGLE_API_KEY=your_key_here
HUGGINGFACEHUB_API_TOKEN=your_key_here
```

---

## 📫 Connect

- **LinkedIn:** [linkedin.com/in/prathamesh-sarode2004](https://linkedin.com/in/prathamesh-sarode2004)
- Learning in public — posting daily progress breakdowns with code walkthroughs

⭐ If you're on a similar learning path, feel free to fork this and follow along.