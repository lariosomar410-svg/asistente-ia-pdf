import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Cargar variables de entorno desde el archivo .env
load_dotenv()

st.set_page_config(page_title="Asistente de Documentos IA", page_icon="🤖", layout="wide")

# Barra lateral para ingresar o modificar la API Key de Google
st.sidebar.title("⚙️ Configuración")
env_api_key = os.getenv("GOOGLE_API_KEY", "")

api_key = st.sidebar.text_input(
    "Google API Key:",
    value=env_api_key,
    type="password",
    help="Ingresa tu clave de Google AI Studio"
)

st.title("🤖 Asistente Virtual con IA (RAG)")
st.caption("Consulta tus documentos PDF en tiempo real con Inteligencia Artificial")

if not api_key:
    st.warning("⚠️ Por favor ingresa una Google API Key válida en la barra lateral para comenzar.")
    st.stop()

# 1. Carga del archivo PDF
pdf_doc = st.file_uploader("Carga tu archivo PDF para analizar", type=["pdf"])

if pdf_doc is not None:
    temp_pdf_path = f"temp_{pdf_doc.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_doc.getbuffer())

    try:
        with st.spinner("Procesando documento y generando vectores..."):
            loader = PDFPlumberLoader(temp_pdf_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)

            # Generación de embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key
            )
            vectorstore = FAISS.from_documents(splits, embeddings)
            retriever = vectorstore.as_retriever()

            # Configuración del modelo LLM activo según la API actual
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.8-flash",
                google_api_key=api_key,
                temperature=0.3
            )

            template = """Eres un asistente virtual experto para responder preguntas basadas únicamente en el contexto provisto.
Si no sabes la respuesta o no está en el documento, indica claramente que no se encuentra en la información proporcionada.
Sé preciso y profesional.

Contexto del documento:
{context}

Pregunta: {question}
"""
            prompt = ChatPromptTemplate.from_template(template)

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs if doc.page_content)

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

        st.success("¡Documento procesado correctamente! Ya puedes realizar preguntas.")

        # 2. Interfaz de consulta
        user_query = st.text_input("Escribe tu pregunta sobre el documento:")

        if user_query:
            with st.spinner("Consultando al asistente..."):
                response = rag_chain.invoke(user_query)
                st.markdown("### 💡 Respuesta:")
                st.write(response)

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)