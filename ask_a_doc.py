import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

if "response" not in st.session_state:
    st.session_state.response = None
    
def generate_response (uploaded_file, openai_api_key, query_text):
    #Load document if file is uploaded
    if uploaded_file is not None:
        documents=[uploaded_file.read().decode()]
        #Split documents into chunks
        text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
        texts=text_splitter.create_documents(documents)
        #Select embeddings
        embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)
        #Create a vectorstore from documents
        db=Chroma.from_documents(texts,embeddings)
        #Create retriever interface
        retriever=db.as_retriever()
        #Create QA chain
        qa=RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key),chain_type='stuff',retriever=retriever)
        return qa.run(query_text)
#Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')
#File upload
uploaded_file=st.file_uploader('Upload an article',type='txt')
#Query text
query_text=st.text_input('Enter your question:',placeholder='Please provide a short summary.',disabled=not uploaded_file)
#Form input and query
result=[]
with st.form('myform',clear_on_submit=True):
    openai_api_key=st.text_input('OpenAI API Key',type='password',disabled=not (uploaded_file and query_text))
    submitted=st.form_submit_button('Submit',disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            st.session_state.response = generate_response(uploaded_file, openai_api_key, query_text)
if st.session_state.response:
    st.info(st.session_state.response)
