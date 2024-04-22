import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Define CSS styles directly in Python for Streamlit
css = '''
    <style>
        /* Reset some default styles */
        body, html {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background-color: #f2f2f2;
            height: 100%;
        }

        /* Center content vertically */
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        /* Chat container */
        .chat-container {
            max-width: 400px;
            width: 100%;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
        }

        /* Chat message */
        .chat-message {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            overflow: hidden;
            font-size=30;
        }
    
        /* Avatar */
        .avatar {
            flex-shrink: 0;
            width: 50px;
            height: 50px;
            margin-right: 10px;
            overflow: hidden;
             align-self: flex-start
        }

        .avatar img {
            margion-top:-2px:
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        /* Message bubble */
        .message-bubble {
            flex-grow: 1;
            padding: 10px;
            background-color: #f1f0f0;
            border-radius: 20px;
            animation: typing 2s steps(40) 1s forwards;
        }

        /* Typing animation */
        @keyframes typing {
            from { width: 0 }
            to { width: 100% }
        }

        /* User message style */
        .user .message-bubble {
            background-color: #2b313e;
            color: #fff;
        }

        /* Bot message style */
        .bot .message-bubble {
            background-color: #475063;
            color: #fff;
        }
    </style>
'''

sidebar_css = '''
            <style>
                .sidebar .sidebar-content {
                    background-color: #fff; /* Specify your desired background color here */
                }
                  .sidebar .stButton button {
        width: 200px; /* Set desired width for the button */
        height: 50px; /* Set desired height for the button */
        font-size: 16px; /* Optional: Adjust font size */
    }
            </style>
        '''


# Function to load and analyze chat history
def analyze_chat_history(chat_data):
    # Extract questions from chat data
    questions = [item['question'] for item in chat_data]

    # Count question frequencies
    question_counts = Counter(questions)

    # Most asked question
    most_asked_question = question_counts.most_common(1)[0][0]
    most_asked_question_count = question_counts.most_common(1)[0][1]

    return question_counts, most_asked_question, most_asked_question_count


# Main Streamlit function
def main():

    load_dotenv()
    st.set_page_config(page_title="Chat PDFs 📚")

    # Inject CSS styles into the Streamlit app
    st.markdown(css, unsafe_allow_html=True)

    st.header("Chat with PDFs 📚")

    # Initialize session state variables
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

   
    st.markdown(sidebar_css, unsafe_allow_html=True)

      # Toggle to activate feature
    on = st.sidebar.checkbox('Analytics')

    if on:
        st.toast("Alayzed the Documents")

        # Load JSON data
        with open('chat_history.json', 'r') as json_file:
            chat_data = json.load(json_file)

        # Analyze chat history
        question_counts, most_asked_question, most_asked_question_count = analyze_chat_history(chat_data)

        # Display summary information
        st.write(f"Total unique questions: {len(question_counts)}")
        st.write(f"Most asked question: '{most_asked_question}' (asked {most_asked_question_count} times)")

        # Button to show detailed analytics in a separate pop-up screen
        if st.button('Show Graph Analytics'):
            # Open a separate pop-up screen for detailed analytics
            popup = st.expander('Detailed Analytics', expanded=True)

            # Create DataFrame for detailed plotting
            df_plot = pd.DataFrame(question_counts.most_common(), columns=['Question', 'Frequency'])

            # Plotting using Matplotlib inside the pop-up screen
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(df_plot['Question'], df_plot['Frequency'], color='skyblue')
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Question')
            ax.set_title('Most Frequently Asked Questions')
            ax.invert_yaxis()  # Invert y-axis to show most common at the top

            # Display detailed analytics in the pop-up screen
            popup.pyplot(fig)

    # Sidebar to handle PDF processing
    with st.sidebar:
        st.subheader("Your documents")
        folder_path = "AA Infographics"  # Provide the folder path here
        pdf_docs = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]

        # Process button to create vector store and conversation chain
        if st.button("Process"):
            raw_texts = [get_pdf_text([pdf]) for pdf in pdf_docs]
            text_chunks = [chunk for text in raw_texts for chunk in get_text_chunks(text)]
            st.session_state.vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)

        # Upload button to process a new PDF
        uploaded_file = st.file_uploader("Upload a new PDF", type="pdf" ,)
        if uploaded_file is not None and st.button("Process New Pdf"):
            raw_texts = [get_pdf_text([uploaded_file])]
            text_chunks = [chunk for text in raw_texts for chunk in get_text_chunks(text)]
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = get_vectorstore(text_chunks)
            else:
                st.session_state.vectorstore.add_texts(text_chunks)
            st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)

    # Chat interface
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation:
            response = st.session_state.conversation({'question': user_question,    })
            chat_messages = [
                (message.content, "user") if i % 2 == 0 else (message.content, "bot")
                for i, message in enumerate(response['chat_history'])
            ]
            for message, sender in chat_messages:
                if sender == "user":
                    # Display user's question
                    st.markdown(f'''
                        <div class="chat-message user">
                            <div class="avatar">
                                <img src="data:image/jpeg;base64,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">
                            </div>
                            <div class="message">{message}</div>
                        </div>
                    ''', unsafe_allow_html=True)
                elif sender == "bot":
                    # Display bot's answer
                    st.markdown(f'''
                        <div class="chat-message bot">
                            <div class="avatar">
                                <img src="https://womensafetywing.telangana.gov.in/sahas/wp-content/uploads/2023/05/ksm-66.png">
                            </div>
                            <div class="message">{message}</div>
                        </div>
                    ''', unsafe_allow_html=True)

                    # Save each message to chat history JSON
                    save_chat_history([message], user_question)

    # Display conversation history in the sidebar
    st.sidebar.subheader("Conversation History")
    history_file_path = "chat_history.json"

    if os.path.exists(history_file_path):
        with open(history_file_path, 'r') as history_file:
            history_data = json.load(history_file)

        for idx, item in enumerate(history_data):
            question = item['question']
            # st.sidebar.markdown(f"**{idx + 1}.** {question}")

            # Display corresponding answer when question is clicked
            if st.sidebar.button(f"{idx + 1} : {question} "):
                answer = item['content']
                st.write(f"**Question:** {question}")
                st.write(f"**Answer:** {answer}")
            st.sidebar.write("---")  # Divider between items
    
    

# Define a function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Define a function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Define a function to create a vector store from text chunks
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Define a function to create a conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
    # llm = ChatGoogleGenerativeAI(model="gemini-pro",
    #                          temperature=0.3,convert_system_message_to_human=True)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to save chat history to JSON
def save_chat_history(chat_messages, user_question):
    file_path = "chat_history.json"
    data = []

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)

    # Append each chat message to the data list
    for message in chat_messages:
        data.append({
            'content': message,  # Extracting the message content
            'question': user_question
        })

    # Save the updated data list to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == '__main__':
    main()
