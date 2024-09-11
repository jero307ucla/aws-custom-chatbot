from langchain.llms.bedrock import Bedrock
from langchain_community.chat_models import BedrockChat
import json
import os
import boto3
from langchain.llms.base import LLM
from typing import Optional, List, Dict
from dotenv import load_dotenv
import streamlit as st
from csv_processor import parse_csv, process_csv
from pdf_docx_processor import get_response, camelot_parser, textract
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType

## Client Side
from langchain.prompts import PromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains import RetrievalQA

from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory

import pandas as pd
from langchain.chains.llm import LLMChain
#File uploader and parser
from langchain_community.embeddings import BedrockEmbeddings
## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
## Pdf Loader
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
## import FAISS
from langchain_community.vectorstores import FAISS
import uuid

from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

#get rid of langchain?? :(
load_dotenv()

# AWS credentials
aws_access_key_id = os.getenv('ACCESS_KEY')
aws_secret_access_key = os.getenv('SECRET_ACCESS_KEY')
aws_region = os.getenv('REGION_NAME')
BUCKET_NAME = os.getenv("BUCKET_NAME")
folder_path="/tmp/"


bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )

s3_client = boto3.client(
    service_name="s3", 
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region)

textract_client = boto3.client(
    service_name="textract", 
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region)

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

# Initialize Boto3 client
def create_llm(temperature=0):
    # Initialize the custom LLM with the Bedrock client
    model_id = 'anthropic.claude-v2'
    # model_id = 'anthropic.claude-3-opus-20240229-v1:0'
    # model_id = 'amazon.titan-text-premier-v1:0'
    llm = Bedrock(
        model_id=model_id, 
        client=bedrock_client, 
        model_kwargs={"temperature": temperature}
    )
    # llm = BedrockChat(
    #     model_id=model_id,
    #     client=bedrock_client
    # )
    return llm

# Create conversational buffer memory
def client_memory():
    llm = create_llm()
    memory = ConversationBufferMemory(llm=llm, max_token_limit=512, memory_key='chat_history',return_messages=True)
    return memory

def get_uid():
    return str(uuid.uuid4())

## Split the pages / text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

## create vector store
def create_vector_store(request_id, documents):
    vectorstore_faiss=FAISS.from_documents(documents, bedrock_embeddings)
    file_name=f"{request_id}.bin"
    folder_path="/tmp/"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    ## upload to S3
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")

    return True

def load_index():
    # file_name=f"{request_id}.bin"
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")

# Streamlit UI
st.title("AWS Audit Artifact POC")

# Initialize memory and chat history if not yet created
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'memory' not in st.session_state:
    st.session_state.memory = client_memory()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
for message in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(message["user"])
    with st.chat_message("assistant"):
        st.markdown(message["assistant"])
    st.session_state.memory.save_context(inputs={"human":message["user"]},outputs={"AI":message["assistant"]})

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "csv", "png"], accept_multiple_files=False)
    if st.button("Clear chat history"):
        st.session_state.memory.clear()
        st.session_state.chat_history = []
        # uploaded_file = None
if uploaded_file is not None:
    request_id = get_uid()
    # st.write(f"Request Id: {request_id}")
    saved_file_name = None
    loader = None

    if 'pdf' in uploaded_file.type:
        saved_file_name = f"{request_id}.pdf"
        # saved_file_name += ".pdf"
        # loader = PyPDFLoader(saved_file_name)
    elif 'doc' in uploaded_file.type:
        saved_file_name = f"{request_id}.docx"
        # loader = Docx2txtLoader(saved_file_name)
    elif 'csv' in uploaded_file.type:
        saved_file_name = f"{request_id}.csv"
    elif 'png' in uploaded_file.type:
        saved_file_name = f"{request_id}.png"

    with open(saved_file_name, mode="wb") as w:
        w.write(uploaded_file.getvalue())
    # st.write(uploaded_file.type)
    
    if 'pdf' in uploaded_file.type:
        loader = PyPDFLoader(saved_file_name)
    elif 'doc' in uploaded_file.type:
        loader = Docx2txtLoader(saved_file_name)
    # elif 'csv' in uploaded_file.type:
    #     saved_file_name = f"{request_id}.docx"
    # loader = PyPDFLoader(saved_file_name) if 'application/pdf' in uploaded_file.type else Docx2txtLoader(saved_file_name)
    if 'csv' not in uploaded_file.type and 'png' not in uploaded_file.type:
        # camelot_parser(uploaded_file)
        # textract(textract_client, saved_file_name)
        pages = loader.load_and_split()

        st.write(f"Total Pages: {len(pages)}")

        ## Split Text
        splitted_docs = split_text(pages, 1000, 200)
        st.write(f"Splitted Docs length: {len(splitted_docs)}")
        # st.write("===================")
        # st.write(splitted_docs[0])
        # st.write("===================")
        # st.write(splitted_docs[1])
        result = None
        if st.session_state.vector_store is None:
            with st.spinner('Creating vector store ...'):
                result = create_vector_store(request_id, splitted_docs)
                st.session_state.vector_store = result
        if st.session_state.vector_store:
            with st.chat_message("assistant"):
                st.markdown(f"You have successfully uploaded {uploaded_file.name}.")
            # st.write("Hurray!! PDF processed successfully")
            load_index()
            dir_list = os.listdir(folder_path)
            # st.write(f"Files and Directories in {folder_path}")
            # st.write(dir_list)
            #File must be uploaded to S3 correctly
            # s3_client.upload_fileobj(uploaded_file, Bucket=BUCKET_NAME, Key=saved_file_name)
            # s3_client.upload_file(Filename=folder_path + "/" + saved_file_name, Bucket=BUCKET_NAME, Key=saved_file_name)
            # textract(client=textract_client, bucket_name=BUCKET_NAME, document_key=saved_file_name)
            ## create index
            faiss_index = FAISS.load_local(
                index_name="my_faiss",
                folder_path = folder_path,
                embeddings=bedrock_embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            with st.chat_message("assistant"):
                st.markdown(f"Error: Please check logs.")
            # st.write("Error!! Please check logs.")
    elif 'csv' in uploaded_file.type:
        df, schema = parse_csv(uploaded_file, llm=create_llm())
        # df = df.dropna()
        # st.dataframe(df.head(5))
    elif 'png' in uploaded_file.type:
        # s3_client.upload_fileobj(uploaded_file, Bucket=BUCKET_NAME, Key=saved_file_name)
        textract(client=textract_client, bucket_name=BUCKET_NAME, document_key=saved_file_name)

else:
    st.session_state.vector_store = None
    st.write('Vector store cleared upon file removal.')
# with st.container():
question = st.chat_input("Please ask your question")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    if st.session_state.vector_store:
        with st.spinner('Working on your request ...'):
            chat_response = None
            if "csv" in uploaded_file.type:
                chat_response = process_csv(llm=create_llm(), csv=saved_file_name, prompt=question, chat_history=st.session_state.chat_history)
            else:
                chat_response = get_response(llm=create_llm(), vectorstore=faiss_index, question=question, memory=st.session_state.memory, chat_history=st.session_state.chat_history, csv=uploaded_file, iscsv= 'csv' in uploaded_file.type)

    with st.chat_message("assistant"):
        st.markdown(chat_response)

    st.session_state.chat_history.append({"user": question, "assistant": chat_response})
    st.session_state.memory.save_context(inputs={"human": question},outputs={"AI": chat_response})
    # st.write("INDEX IS READY")


        
