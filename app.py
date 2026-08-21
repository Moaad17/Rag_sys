import streamlit as st
from chatbot import retriever, llm, prompt

st.set_page_config(
    page_title="Assistant Restaurant",
    page_icon="🍽️"
)

st.title("🍽️ Assistant Dar Zellij")

question = st.text_input("Votre question :")

if question:

    # 1. Recherche dans Chroma
    results = retriever.invoke(question)

    # 2. Afficher les chunks récupérés
    st.write("## 🔎 Chunks récupérés")

    for i, doc in enumerate(results):
        st.write(f"### Chunk {i + 1}")
        st.write(doc.page_content)

    # 3. Construire le contexte
    context = "\n\n".join(
        doc.page_content for doc in results
    )

    # 4. Envoyer au LLM
    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)

    # 5. Afficher la réponse
    st.write("## 🤖 Réponse")
    st.write(response.content)