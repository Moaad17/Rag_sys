# 🍽️ Chatbot IA pour Restaurants — Pipeline RAG

Chatbot intelligent permettant aux entreprises d'automatiser 
les réponses aux questions fréquentes de leurs clients 
(horaires, tarifs, services).

## Architecture
Document client → Chunking → Embeddings (Sentence Transformers) 
→ Base vectorielle (Chroma) → Retrieval → LLM (Groq) → Réponse

## Démo
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Stack
Python, LangChain, Chroma, Sentence Transformers, Groq, Streamlit