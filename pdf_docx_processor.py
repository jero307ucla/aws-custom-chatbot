from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_community.vectorstores import FAISS
import pandas as pd
import camelot
from csv_processor import parse_csv
# from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers.string import StrOutputParser
from utils import create_llm
from trp import Document

def get_response(llm,vectorstore, question, memory, chat_history, csv=None, iscsv=False):
    ## create prompt / template
    context = """
    You are an AI assistant who can answer queries about the uploaded document. Use the following chat history  to answer the Human's question directly. 
    ------
    Chat history:
    {chat_history}
    ------
    """

    prompt_template = """
    System: 
    You are an AI assistant who can answer queries about the uploaded document. Use the following chat history to answer the Human's question directly.
    Context:
    {context}
    ------
    Chat History:
    {chat_history}
    </hs>
    ------
    Human:
    {question}
    Assistant: 
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=[
            "question",
            "chat_history",
            "context"
        ]
    )

    qachat = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        # return_source_documents=True,
        get_chat_history=lambda h : h,
        combine_docs_chain_kwargs={"prompt": prompt}
        # condense_question_prompt=prompt
        #.format(question=question, chat_history=""),
    )

    return qachat({"question": question, "chat_history": chat_history})["answer"]

#Extracts images and text from pdf separately to be stored as vector embeddings
def camelot_parser(file):
    tables = camelot.read_pdf(r"C:\Users\jerome\Internship\ArtifactChat\Sample Data\Science_Worksheet_Tables.pdf", pages="all", )
    if tables:
        tab = st.tabs(["Table"])
        # with tab:
        for table in tables:
            parse_csv(file=file, llm=create_llm(), dataframe=table.df)
    pass

def unstructured_parser(file_path):
    pass

def textract(client, bucket_name, document_key, filename=None):
    # with open(filename, "rb") as file:
    response = client.analyze_document(
        Document={
            'S3Object': {'Bucket': bucket_name, 
                         'Name': document_key}
            },
        FeatureTypes=["TABLES"])  # This will specifically extract table
        # response = client.analyze_document(
        #     Document={
        #         'Bytes': file.read(),
        #     },
        #     FeatureTypes=["TABLES"])

    print(response)

    doc = Document(response)

    for page in doc.pages:
        # Print tables
        for table in page.tables:
            for r, row in enumerate(table.rows):
                for c, cell in enumerate(row.cells):
                    print("Table[{}][{}] = {}".format(r, c, cell.text))
            df = pd.DataFrame(table[1:], columns=table[0])
            st.table(df)
            # st.dataframe(table)
    pass