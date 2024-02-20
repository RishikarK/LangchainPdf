import streamlit as st
import time
from PyPDF2 import PdfReader
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

processed_data = {}  # Store processed data globally


def get_pdf_text(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def preprocess_pdfs(folder_path):
    pdf_files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]
    for pdf_file in pdf_files:
        try:
            raw_text = get_pdf_text(os.path.join(folder_path, pdf_file))
            text_chunks = get_text_chunks(raw_text)
            processed_data[pdf_file] = get_store(text_chunks)
        except PyPDF2.errors.PdfReadError as e:
            st.sidebar.error(f"Error processing PDF {pdf_file}: {str(e)}")
    return pdf_files


def user_input(user_question):
    global processed_data
    chain = get_conversational_chain()
    for pdf_file, vector_store in processed_data.items():
        docs = vector_store.similarity_search(user_question)
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        st.write(f"Answer from {pdf_file}: ", response["output_text"])


def main():
    st.set_page_config("Q&A PDF")
    st.header("Ask a question about multiple PDF files")

    user_question = st.text_input("Ask a question about the PDF Files")

    folder_path = "/home/enterpi/Desktop/PdfReader/AA Articles/"

    pdf_files = preprocess_pdfs(folder_path)

    if user_question:
        user_input(user_question)

    if pdf_files:
        st.sidebar.title("PDF Files:")
        for pdf_file in pdf_files:
            st.sidebar.write(pdf_file)
            st.sidebar.success(f"PDF {pdf_file} processed successfully.")
    else:
        st.sidebar.warning("No PDF files found in the specified folder.")


if __name__ == "__main__":
    main()
