from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = "your-key-here"


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get__store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = OpenAIEmbeddings()
    
    new = FAISS.load_local("faiss_index", embeddings)
    docs = new.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    st.write("Answer : ", response["output_text"])
    print(response)
    


def uploading():
    msg = st.toast('Gathering information...')
    time.sleep(1)
    msg.toast('Getting Ready...')
    time.sleep(1)
    msg.toast('Done!', icon = "🎉")



def main():
    st.set_page_config("Q&A PDF")
    st.header("Ask question From multiple PDF's")

    user_question = st.text_input("Ask a Question from the PDF Files")

    pdf_folder_path = "/home/enterpi/Desktop/PdfReader/AA Infographics"

    if user_question and pdf_folder_path:
        if os.path.isdir(pdf_folder_path):
            pdf_files = [os.path.join(pdf_folder_path, file) for file in os.listdir(pdf_folder_path) if file.endswith('.pdf')]
            if pdf_files:
                raw_text = get_pdf_text(pdf_files)
                text_chunks = get_text_chunks(raw_text)
                get__store(text_chunks)
                user_input(user_question)
            else:
                st.write("No PDF files found in the specified folder.")
        else:
            st.write("Invalid folder path. Please provide a valid folder path.")

    # with st.sidebar:
    #     st.title("Menu:")
    #     # Remove the file uploader widget

if __name__ == "__main__":
    main()
