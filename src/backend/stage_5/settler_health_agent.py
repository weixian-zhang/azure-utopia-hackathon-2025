import sqlite3
import os
from pydantic import BaseModel
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    QuerySQLCheckerTool,
    QuerySQLDatabaseTool,
    ListSQLDatabaseTool
)
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# https://docs.langchain.com/oss/python/langchain/sql-agent


db_path = os.path.join(os.path.dirname(__file__), 'settlers.sqlite')
sqlite_conn = sqlite3.connect(db_path)
sqlite = SQLDatabase.from_uri('sqlite:///' + db_path)


@tool
def sql_executor(query: str) -> str:
    """
    Execute the SQL query and return the results."""
    executor_tool = QuerySQLDatabaseTool(
        db=sqlite
    )
    result = executor_tool.run(query)
    return result


llm: AzureChatOpenAI = AzureChatOpenAI(
            deployment_name="gpt-4o",
            model="gpt-4o",
            api_version="2024-12-01-preview",
            temperature=1.0,
        )

db_tools = SQLDatabaseToolkit(db=sqlite, llm=llm)


class AgentSQLQueryResponse(BaseModel):
    sql_query: str


class SettlerHealthAgent:

    def __init__(self):

            # DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
            # database.
        system_prompt = system_prompt = """
            You are an agent designed to interact with a SQL database.
            Given an input question, create a syntactically correct {dialect} query to run,
            then look at the results of the query and return the answer. Unless the user
            specifies a specific number of examples they wish to obtain, always limit your
            query to at most {top_k} results.

            You can order the results by a relevant column to return the most interesting
            examples in the database. Never query for all the columns from a specific table,
            only ask for the relevant columns given the question.

            You MUST double check your query before executing it. If you get an error while
            executing a query, rewrite the query and try again.

            To start you should ALWAYS look at the tables in the database to see what you
            can query. Do NOT skip this step.

            When asked to update existing records and create new records, explain which database fields were updated or added.

            If user ask for a passenger, return the name instead of passenger_id.

            if user ask to list all passengers with abnormal vital signs, list "all" passenger names instead of just one passenger

            if no vital records found for a passenger, respond with "no vital records found for this passenger, so no rows were affected."

            when evaluating blood pressure, ensure to check both systolic and diastolic readings.

            "List" means select all matching records. Example if list all passengers with abnormal vital signs, you should return all matching passenger names.

            If user question is not related to the database, politely inform user you cannot assist. Don't need to explain reason.

            Use only the database to answer the question. Do not use any prior knowledge.

            Answer short and concise.

            Then you should query the schema of the most relevant tables.
            """.format(
                dialect=sqlite.dialect,
                top_k=3,
            )
        
        self.agent = create_react_agent(
            model=llm,
            tools=db_tools.get_tools(), #[SettlerHealthAgent.sql_executor, SettlerHealthAgent.sql_checker],
            # response_format={"schema": AgentSQLQueryResponse},
            #tools=[SettlerHealthAgent.list_sql_tables, SettlerHealthAgent.sql_checker],
            checkpointer=MemorySaver(),  # Enables memory!
            prompt=system_prompt
        )

    

    @staticmethod
    @tool
    def sql_checker(query: str) -> str:
        """
        Check if the generated SQL query is correct. If not, rewrite the query based on the error message provided by the tool.
        """
        checker_tool = QuerySQLCheckerTool(
            llm=llm,
            db=sqlite
        )
        result = checker_tool.run(query)
        return result
    

    
    
        # Follow these steps:
        # 1. First, use list_sql_tables to list all avaialable tables
        # 2. Generate an appropriate SQL query based on the user's question
        # 3. Use sql_checker to validate your SQL query
        # 5. Finally, provide a natural language response based on the query results

    def invoke(self, query: str) -> str:

        # chain of thoughts
        # system_prompt = f'''
        # *You are a helpful assistant specialized in providing health advice to settlers 
        # based on their medical records stored in SQLite database.

        # Available SQLite database tables and schema:
        # - passenger: Contains passenger information (passenger_id, passenger_name)
        # - vital: Contains vital signs (vital_id, passenger_id, heart_rate_bpm, systolic_bp_mmHg, diastolic_bp_mmHg, spo2_pct, body_temp_c, respiration_rate_bpm)

        # user question examples:
        # 1. What is the health status of settler Lucas Wong?
        # 2. List all settlers with abnormal vital signs.
        # 3. Which settlers are at risk based on their vital signs?
        # 4. Generate a summary report of settlers' health conditions.

        # Steps to generate SQLite query:

        # 1. select passenger_id from passenger where name = 'Lucas Wong';
        #     1.1 join passenger to vital table on passenger_id
        #     1.2 select all heart_rate_bpm, systolic_bp_mmHg, diastolic_bp_mmHg, spo2_pct, body_temp_c, respiration_rate_bpm are within normal ranges
        #     1.3 execute the SQL query using tool sql_executor
        #     1.4 analyze the results and determine health status

        # 2. select all passenger_id, name from passenger table
        #     2.1 join passenger to vital table on passenger_id
        #     2.2 select all passenger_id, name where heart_rate_bpm, systolic_bp_mmHg, diastolic_bp_mmHg, spo2_pct, body_temp_c, respiration_rate_bpm are outside normal ranges
        #     2.3 execute the SQL query using tool sql_executor
        #     2.4 analyze the results and determine abnormal health signs

        # 3. select all passenger_id, name from passenger table
        #     3.1 join passenger to vital table on passenger_id
        #     3.2 select all passenger_id, name where heart_rate_bpm, systolic_bp_mmHg, diastolic_bp_mmHg, spo2_pct, body_temp_c, respiration_rate_bpm are at risk levels
        #     3.3 execute the SQL query using tool sql_executor
        #     3.4 analyze the results and determine settlers at risk

        # 4. select all passenger_id, name from passenger table
        #    4.1 join passenger to vital table on passenger_id
        #    4.2 select all passenger_id, name where heart_rate_bpm, systolic_bp_mmHg, diastolic_bp_mmHg, spo2_pct, body_temp_c, respiration_rate_bpm are within normal ranges
        #    4.3 execute the SQL query using tool sql_executor
        #    4.4 analyze the results and generate summary report


        # 5. what is the passenger_id of settler Tomoko Sato?
        #     5.1 select passenger_id from passenger where name = 'Tomoko Sato';
        #     5.2 execute the SQL query using tool sql_executor
        #     5.3 analyze the results and provide the passenger_id

        # 6. How many people has the average respiration rate lower than 13?
        #     6.1 select count(*) from vital where respiration_rate_bpm < 13;
        #     6.2 join vital to passenger table on passenger_id
        #     6.3 execute the SQL query using tool sql_executor
        #     6.4 analyze the results and provide the count

        # execute the above steps to generate the appropriate SQLite query to answer the user question.

        # '''

        messages = [
            HumanMessage(content=f"{query}")
        ]

        config = {"configurable": {"thread_id": datetime.now().strftime("%Y%m%d%H%M%S")}}

        response = self.agent.invoke({"messages": messages}, config=config)

        result = response['messages'][-1].content

        return result

        # while iterations > 0:
        #     response: AIMessage = self.llm.bind_tools([self.list_sql_tables, self.sql_checker]).invoke(messages)
        #     messages.append(response)

    
        #     sql_query = result.content

        #     result = self.sqlite.run(sql_query, fetch='all')

        #     iterations -= 1

        # return result



if __name__ == "__main__":
    agent = SettlerHealthAgent()
    user_question_1 = "What is the health status of settler Lucas Wong?"
    user_quesrtion_2 = "How many people has the average respiration rate lower than 13?"
    user_quesrtion_3 = "What is the passenger id for Priya Nair?"
    q4 = 'What is the capital of France?'
    q5 = 'Who has the lowest blood oxygen level?'
    q6 = 'List all passengers who have a blood pressure reading above 130/85.'
    q7 = 'Add a new health vital record for passenger id 1005 with heart rate 78, blood pressure 120/80, respiration rate 16, body temp of 36, and spo2 98%. the date is 2025-10-27, 10am.'
    q8 = 'All vital data for passenger 1020 has been deleted. However, there were no vital records found for this passenger, so no rows were affected.'
    response = agent.invoke(q6)
    print(response)