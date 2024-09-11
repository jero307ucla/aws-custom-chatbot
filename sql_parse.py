from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories.dynamodb import DynamoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

current_session_id = "tests-abc"
history = DynamoDBChatMessageHistory(table_name=DYNAMODB_TABLE_NAME, session_id=current_session_id)

user_query = "How many employees are there?"

### SQL CHAIN 
system_message_sql = """
    You are a {dialect} expert. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    Here is the database schema:
    <schema>{table_info}</schema>
    
    Here are previous messages: {messages}
    Write ONLY THE SQL QUERY and nothing else. 
    Do not wrap the SQL query in any other text, not even backticks.
    For example:
    Question: Name 10 artists
    SQL Query: <sql_query> SELECT Name FROM Artist LIMIT 10; <sql_query>
    Your turn:
    Question: {question}
    SQL Query:
    """

human_message = [{"type": "text", "text": "{question}"}]

template = [
    ("system", system_message_sql),
    MessagesPlaceholder(variable_name="messages"),
    ("human", human_message),
]

prompt = ChatPromptTemplate.from_messages(template)

def get_table_info(_):
    return db.get_table_info()

def get_dialect(_):
    return db.dialect()

sql_chain =  (
    RunnablePassthrough.assign(table_info=get_table_info, dialect=get_dialect)
    | prompt
    | llm
    | StrOutputParser()
  )

### FULL CHAIN 
system_message_full = """
Based on the question from the user, sql_query, and sql_response, write a natural language response.
Do not exaplain the whole process of extracting response. 
SQL Query: <sql_query> {query} </sql_query>
Previous messages: {messages}
User question: {question}
SQL Response: {response}"""

human_message = [{"type": "text", "text": "{question}"}]

template = [
    ("system", system_message_full),
    ("human", human_message),
]

prompt = ChatPromptTemplate.from_messages(template)


chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
  response=lambda vars: db.run(vars["query"]),
)
| prompt
| llm
)


### MEMORY
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: DynamoDBChatMessageHistory(
        table_name=DYNAMODB_TABLE_NAME, session_id=session_id
    ),
    input_messages_key="question",
    history_messages_key="messages",
)

config = {
    "configurable": {"session_id": current_session_id},
}


#INVOKING
response = chain_with_history.invoke({
    "question": user_query}, 
    config=config)