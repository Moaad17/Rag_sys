# === 1. Charger le document ===
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os


loader = TextLoader("restaurant.txt", encoding="utf-8")
docs = loader.load()


# === 2. Chunking ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20
)

chunks = text_splitter.split_documents(docs)


# === 3. Embeddings ===
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# === 4. Chroma ===
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)


# === 5. Retriever ===
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# === 6. API Groq ===
load_dotenv()


api_key = os.getenv("GROQ_API_KEY")

model="openai/gpt-oss-120b"

llm = ChatGroq(
    model=model,
    temperature=0,
    groq_api_key=api_key
)


# === 7. Prompt ===
prompt = ChatPromptTemplate.from_template("""
Tu es un assistant qui répond aux questions en utilisant uniquement
les informations fournies dans le contexte.

Si la réponse ne se trouve pas dans le contexte, dis clairement :
"Je ne trouve pas cette information dans le document."

Ne fais aucune supposition.

Contexte :
{context}

Question :
{question}

Réponse :
""")


# === 8. Question ===
query = "Quels sont les horaires du restaurant ?"


# === 9. Retrieval ===
results = retriever.invoke(query)


# === 10. Construire le contexte ===
context = "\n\n".join(
    doc.page_content for doc in results
)


# === 11. Envoyer au LLM ===
messages = prompt.format_messages(
    context=context,
    question=query
)

response = llm.invoke(messages)


# === 12. Réponse ===
print("\n=== Réponse ===")
print(response.content)