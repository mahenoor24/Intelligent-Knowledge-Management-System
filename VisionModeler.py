import streamlit as st
import whisper
import ollama
import chromadb
from crewai import Crew
from PyPDF2 import PdfReader
import speech_recognition as sr

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
db = chroma_client.get_or_create_collection(name="pdf_assistant")

# Load Whisper model for voice input
whisper_model = whisper.load_model("base")

st.title("📄 PDF-Assistant: AI-Powered Q&A for PDFs")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

def process_pdf(file):
    reader = PdfReader(file)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

if uploaded_file:
    with st.spinner("Processing PDF..."):
        document_text = process_pdf(uploaded_file)
        db.add(documents=[document_text], metadatas=[{"source": uploaded_file.name}], ids=["pdf_1"])
    st.success("PDF processed successfully!")

# Voice input using microphone
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
    try:
        with st.spinner("Transcribing..."):
            audio_text = whisper_model.transcribe(audio)["text"]
        return audio_text
    except Exception as e:
        return f"Error: {str(e)}"

if st.button("Speak your query"):  
    query = recognize_speech()
    st.write("You said:", query)
else:
    query = st.text_input("Ask a question about the document")

if query:
    with st.spinner("Fetching answer..."):
        result = ollama.chat(model="llama3", messages=[{"role": "user", "content": query}])
    st.write("Answer:", result["message"]["content"])
