import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
import os
import time

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "Your key here"

# Text Processing
st.title("Text Question Answering")

# Input text (instead of PDF upload)
text_input = st.text_area("Enter your text here")




if text_input:
    # Split text into chunks
    st.write("Splitting text into chunks...")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(text_input)

    # Question Answering
    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    query = st.text_input("Ask a question")

    if st.button("Search"):
        if query:
            with st.spinner("Searching..."):
                start_time = time.time()  # Start time measurement
                while time.time() - start_time < 5:  # Show the countdown for 5 seconds
                    st.spinner("Searching... Elapsed Time: {:.2f} seconds".format(time.time() - start_time))
                    time.sleep(0.1)  # Update every 0.1 second
                docs = document_search.similarity_search(query)
                similarity_search_time = time.time() - start_time  # Calculate elapsed time

            st.write(f"Similarity Search Time: {similarity_search_time:.2f} seconds")

            with st.spinner("Answering..."):
                start_time = time.time()  # Start time measurement
                while time.time() - start_time < 5:  # Show the countdown for 5 seconds
                    st.spinner("Answering... Elapsed Time: {:.2f} seconds".format(time.time() - start_time))
                    time.sleep(0.1)  # Update every 0.1 second
                result = chain.run(input_documents=docs, question=query)
                question_answering_time = time.time() - start_time  # Calculate elapsed time

            st.write(f"Question Answering Time: {question_answering_time:.2f} seconds")

            st.write(result)
        else:
            st.warning("Please enter a question.")
