import os
import streamlit as st
from typing import List
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
import time

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=COHERE_API_KEY,
        user_agent="lekta-rag/0.1"
    )

@st.cache_resource(show_spinner=False)
def get_llm():
    return init_chat_model(
        "meta-llama/llama-4-scout-17b-16e-instruct",
        model_provider="groq"
    )

@st.cache_resource(show_spinner=True)
def process_urls(
    urls: List[str],
    chunk_size: int,
    chunk_overlap: int,
    _embeds
) -> InMemoryVectorStore:
    loader = WebBaseLoader(web_paths=urls)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)
    return InMemoryVectorStore.from_documents(splits, _embeds)

def retrieve_docs(question: str, k: int) -> List[Document]:
    vs: InMemoryVectorStore = st.session_state.get('vector_store')
    return vs.similarity_search(question, k=k) if vs else []

def generate_answer(question: str, docs: List[Document], custom_prompt: str) -> str:
    context = "\n\n".join(d.page_content for d in docs)
    prompt = custom_prompt.format(question=question, context=context)
    return get_llm().invoke(prompt).content

st.title("RAG por URL")

st.sidebar.header("Fragmentación de Documentos")
chunk_size = st.sidebar.slider("Tamaño de fragmento", 100, 10000, 1000, step=100)
chunk_overlap = st.sidebar.slider("Solapamiento", 0, 1000, 100, step=50)

st.sidebar.header("Parámetros de Búsqueda")
k = st.sidebar.number_input("Número de resultados (k)", min_value=1, max_value=20, value=5)

st.sidebar.header("System Prompt")
custom_prompt = st.sidebar.text_area(
    label="System Prompt",
    value=(
        "Eres un asistente especializado en responder preguntas utilizando únicamente información proporcionada en la documentación recuperada. "
        "Sé claro, preciso y directo.\n"
        "- No inventes ni especules.\n"
        "- Resume sin perder precisión.\n"
        "- Máximo 3 frases.\n\n"
        "Pregunta: {question}\nDocumentación: {context}\nRespuesta:"
    ),
    height=200,
    label_visibility="collapsed"
)

st.header("1. Cargar y procesar URLs")
urls_input = st.text_area("Introduce las URLs (una por línea)", height=150)
if st.button("Procesar URLs") and urls_input:
    urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
    with st.spinner("Procesando documentos y generando embeddings..."):
        vs = process_urls(urls, chunk_size, chunk_overlap, get_embeddings())
        st.session_state['vector_store'] = vs
    st.success("✅ Embeddings en memoria generados correctamente.")

st.header("2. Realizar Consulta")
question = st.text_input("Pregunta", "")
if question:
    vs = st.session_state.get('vector_store')
    if not vs:
        st.warning("Procesa primero algunas URLs para cargar la base de conocimiento.")
    else:
        with st.spinner("Recuperando documentos relevantes..."):
            docs = retrieve_docs(question, k)
        st.subheader(f"Fragmentos recuperados (Top {len(docs)})")
        for i, doc in enumerate(docs, 1):
            with st.expander(f"Fragmento {i}"):
                st.write(doc.page_content)
         with st.spinner("Generando respuesta..."):
            start_time = time.time()  # ⬅️ Inicio del temporizador
            answer = generate_answer(question, docs, custom_prompt)
            elapsed_time = time.time() - start_time  # ⬅️ Fin del temporizador

        st.subheader("Respuesta")
        st.write(answer)
        st.caption(f"⏱️ Tiempo de respuesta: {elapsed_time:.2f} segundos") 
