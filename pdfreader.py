import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
import os

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "Your key here"

# PDF Processing
st.title("PDF Question Answering")
titl=''

pdf_path = st.file_uploader("Upload PDF file", type="pdf")
if pdf_path is not None:
    pdfreader = PdfReader(pdf_path)
    print(pdfreader.metadata)

    titl= pdfreader.metadata.title
    if(titl):
        st.subheader(titl)
    else:
        titl="Document"
        st.subheader(titl)
       
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Question Answering
    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    query = st.text_input("Ask The Question")

    if st.button("Search"):
        if query:
            docs = document_search.similarity_search(query)
            result = chain.run(input_documents=docs, question=query)
            st.write(result)
        else:
            st.warning("Please enter a question.")
