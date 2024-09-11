import streamlit as st
import pandas as pd
import sqlite3
import os
import pandas as pd
from langchain.chains.llm import LLMChain
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.output_parsers.string import StrOutputParser

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


# from langchain.agents.agent_toolkits import create_pandas_dataframe_agent
    
def generate_pandas_query(llm, prompt, schema):
    template = """
    Human: You are a database engineer that has profound Pandas skills. 
    You are an expert at generating Pandas queries from a natural language question.
    Given the input prompt, create a syntactically correct Pandas statement to run as a line of Python code that can be stored in a variable.
    Use the given schema and the table 'df' as reference to the data.
    Few instructions to remember when generating the Pandas Query:
    * Make sure to give only the query with no extra keywords/words.
    * Only use the relevant columns provided in the schema based on the question. Be careful to avoid referencing columns that do not exist in schema.
    * Focus on the keywords indicating calculation. 
    * Please think step by step and always validate the reponse. 
    * Do not generate invalid queries.
    * Use correct column names in the query as provided in the schema.
    * Reference correct DataFrame name 'df'.
    * Use only Pandas compatible functions.

    Given the columns "{schema}" for the DataFrame 'df', generate a single valid Pandas query that can answer the prompt "{prompt}". In your response, return nothing but only a single python pandas query that always applies to the DataFrame 'df' and is immediately executable. If the prompt does not make sense, return a single word 'INVALID'.
    Assistant: 
    """
    prompt_template = PromptTemplate(template=template, input_variables=['schema','prompt'])
    chain = LLMChain(prompt=prompt_template, llm=llm)
    query = chain.invoke({"schema":str(schema), "prompt":prompt})

    return query['text']
    #schema is not correctly passed in

def parse_csv(file, llm, chat_history=None, dataframe=None):
    df = None
    if dataframe is None:
        df = pd.read_csv(file)
        df = df.dropna()
    else:
        df = dataframe
    # headers = 
    db_path = "uploaded_data.db"
    
    # Connect to the SQLite database (creates the file if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Generate a SQL table name from the uploaded file name (e.g., "sales_data" from "sales_data.csv")
    table_name = os.path.splitext(file.name)[0]

    # Step 4: Create a table and insert DataFrame data into the SQLite table
    dc = df.to_sql(table_name, conn, if_exists='replace', index=False)
    dc
    # Commit and close the connection
    conn.commit()
    conn.close()

    st.write(f"Data has been successfully uploaded to the database with table name '{table_name}'.")

    # Provide a way to query the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute a sample query
    query = f"SELECT * FROM {table_name} LIMIT 5;"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    st.write("Sample Query Result:")
    st.write(pd.DataFrame(rows, columns=df.columns))

    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

    # Display database dialect and usable table names
    st.write("Database Dialect:", db.dialect)
    st.write("Usable Table Names:", db.get_usable_table_names())
    
    template = """Human:
        You are a SQLite expert. 
        Given input question "{input}", first create syntactically correct SQLite query to run, then look at the results of the query and return the answer to the input question.
        Unless the user specifies a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per SQLite. 
        You can order the results to return the most informative data in the database.
        Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
        Few instructions to remember when generating the SQL Query:
        * Make sure to give only the query with NO DESCRIPTIVE TEXT and NO EXTRA WORDS.
        * Only use the relevant columns provided in the schema relevant to the question. Be careful to avoid referencing columns that do not exist in the schema.
        * Focus on the keywords indicating calculation. 
        * Exclude all null and none values when calculating any percentage breakdown of findings, and ensure any percentage breakdown is rounded to 1 decimal place. 
        * Think step by step and always validate the reponse. 
        * Recitify each column names by referencing them from the metadata.
        * Do not generate invalid queries.
        * Use correct column names in the query as provided in the DB schema.
        * Reference correct database and table name as given below.
        Write ONLY THE SQL QUERY and nothing else. 
        Do NOT add additional text before or after the query.
        Do not wrap the SQL query in any other text, not even backticks.
        Please output a SQL query and do not speak directly to me, omitting any additional text, summary or instructions.
        The schema is:
        {table_info}
        Assistant: SQL query:
    """
    sql_prompt = PromptTemplate(template=template,input_variables=["input", "top_k", "table_info"])

    write_query = create_sql_query_chain(llm, db, prompt=sql_prompt)
    execute_query = QuerySQLDataBaseTool(db=db)

    chain1 = write_query | execute_query
    
    # agent_executor = create_sql_agent(llm, db=db, verbose=True)

    # chain.invoke({"question": "How many employees are there"})
    # chain.get_prompts()[0].pretty_print()
    answer_prompt = PromptTemplate.from_template(
    """
    
    Given the following user question, corresponding SQL query, and SQL result, answer the user question in a format that continues the conversation from the chat history. 
    If there is an error in the SQL result, reply in a way that takes into account the chat history and the user question.
    If the question is a clarification or followup question rather than a query, you can use the Query and Result to help you repsond to the user's question. But it might not always be relevant.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
    )

    answer = answer_prompt | llm | StrOutputParser()
    chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            result=itemgetter("query") | execute_query
        )
        | answer
    )

    # Provide a way to query the database
    query = st.text_area("Query your table")
    if st.button("Run Query"):
        # result = db.run(query)
        q = chain1.invoke({"question": query})
        response = chain.invoke({"question": query})
        # response = chain.invoke(
        #     {"input": query}
        # )
        st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(response)
    if st.button("Clear Database"):
        clear_database(db_path=db_path)    
        # response = chain.invoke({"question": query})
        # st.write("Query Result:")
        # st.write(response)
        # db.run(response)
        # st.write()

    return df, list(df.columns.values)

def clear_database(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    # Drop all tables
    for table_name in tables:
        print(table_name[0])
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name[0]}";')
    
    conn.commit()
    conn.close()

    st.write("All tables have been cleared.")

def reformat(llm, prompt, result):
    template = """
    Human: Given the prompt: "{prompt}" and the following result: "{result}", return a response that phrases the response as a coherently-worded answer. Do not include in your output any description of the response. Provide the response to the prompt directly.
    Assistant: 
    """
    prompt_template = PromptTemplate(template=template, input_variables=['prompt','result'])
    chain = LLMChain(prompt=prompt_template, llm=llm)
    query = chain.invoke({"result":result, "prompt":prompt})

    return query['text']

def process_csv(llm, csv, prompt, use_pandas_agent=False):
    df, schema = parse_csv(csv)
    # st.dataframe(df.head(5))
    if use_pandas_agent:    # Use pandas agent provided by langchain
        # agent = create_pandas_dataframe_agent(create_llm(), df, verbose=True)
        agent = create_csv_agent(
            llm,
            csv,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )
        response = agent.run(prompt)
        return response
    else:                   # Use custom chain
        #Create array of headers

        #Take in prompt and pass into
        query = generate_pandas_query(llm=llm, prompt=prompt, schema=schema)
        st.markdown(query)
        try:
            result = eval(query, {'df':df, 'pd':pd})
            response = None
            # st.markdown(type(result))
            if isinstance(result, (pd.DataFrame, pd.Series)):
                with st.chat_message("assistant"):
                    st.dataframe(result)
                    response = "Here is the DataFrame visualization that presents the information most relevant to answering your query."
            elif isinstance(result, (int, str)):
                # with st.chat_message("assistant"):
                response = reformat(llm=llm,prompt=prompt, result=result)
                    # st.markdown(response)
            return response
        except:
            with st.chat_message("assistant"):
                st.markdown("Error with execution of query.")
        return "Error"
