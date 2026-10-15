# 🤖 Asistente Virtual de Documentos PDF (RAG)

Una aplicación web interactiva construida con **Streamlit**, **LangChain** y **Google Gemini** que permite realizar consultas en tiempo real sobre archivos PDF utilizando la arquitectura **RAG (Retrieval-Augmented Generation)**.

---

## 🚀 Características

- 📄 **Carga de PDF:** Procesa y extrae texto plano mediante `pdfplumber`.
- 🔍 **Búsqueda Vectorial:** Divide el contenido en fragmentos (*chunks*) e indexa el texto en **FAISS**.
- 🧠 **Modelos de Google Gemini:** Utiliza la API de Google AI Studio para embeddings y generación de respuestas precisas.
- 🔑 **Gestión de API Key:** Configuración sencilla a través de la barra lateral de Streamlit o variables de entorno.

---

## 🛠️ Tecnologías Utilizadas

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Orquestación IA:** [LangChain](https://www.langchain.com/)
- **Modelo de Lenguaje y Embeddings:** Google Gemini (`langchain-google-genai`)
- **Base de Datos Vectorial:** FAISS (`faiss-cpu`)
- **Lectura de PDF:** `pdfplumber`

---

## 📦 Instalación y Configuración Local

1. **Clona el repositorio:**
   ```bash
   git clone [https://github.com/lariosomar410-svg/asistente-ia-pdf.git](https://github.com/lariosomar410-svg/asistente-ia-pdf.git)
   cd asistente-ia-pdf