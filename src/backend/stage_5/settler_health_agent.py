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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


db_path = os.path.join(os.path.dirname(__file__), 'settlers.sqlite')
sqlite_conn = sqlite3.connect(db_path)
sqlite = SQLDatabase.from_uri('sqlite:///' + db_path)

llm: AzureChatOpenAI = AzureChatOpenAI(
            deployment_name="gpt-4o",
            model="gpt-4o",
            api_version="2024-12-01-preview",
            temperature=1.0
        )

class AgentSQLQueryResponse(BaseModel):
    sql_query: str


class SettlerHealthAgent:

    def __init__(self):
        
        self.agent = create_react_agent(
            model=llm,
            tools=[SettlerHealthAgent.sql_executor, SettlerHealthAgent.sql_checker],
            # response_format={"schema": AgentSQLQueryResponse},
            #tools=[SettlerHealthAgent.list_sql_tables, SettlerHealthAgent.sql_checker],
            checkpointer=MemorySaver()  # Enables memory!
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
    

    @tool
    def sql_executor(query: str) -> str:
        """
        Execute the SQL query and return the results."""
        executor_tool = QuerySQLDatabaseTool(
            db=sqlite
        )
        result = executor_tool.run(query)
        return result
    
        # Follow these steps:
        # 1. First, use list_sql_tables to list all avaialable tables
        # 2. Generate an appropriate SQL query based on the user's question
        # 3. Use sql_checker to validate your SQL query
        # 5. Finally, provide a natural language response based on the query results

    def invoke(self, query: str) -> str:

        # chain of thoughts
        system_prompt = f'''
        *You are a helpful assistant specialized in providing health advice to settlers 
        based on their medical records stored in SQLite database.

        Available SQLite database tables and schema:
        - passenger: Contains passenger information (passenger_id, passenger_name)
        - vital: Contains vital signs (vital_id, passenger_id, heart_rate_bpm, systolic_bp_mmHg, diastolic_bp_mmHg, spo2_pct, body_temp_c, respiration_rate_bpm)

        user question examples:
        1. What is the health status of settler Lucas Wong?
        2. List all settlers with abnormal vital signs.
        3. Which settlers are at risk based on their vital signs?
        4. Generate a summary report of settlers' health conditions.

        Steps to generate SQLite query:

        1. select passenger_id from passenger where name = 'Lucas Wong';
            1.1 join passenger to vital table on passenger_id
            1.2 select all heart_rate_bpm, systolic_bp_mmHg, diastolic_bp_mmHg, spo2_pct, body_temp_c, respiration_rate_bpm are within normal ranges

        2. select all passenger_id, name from passenger table
            2.1 join passenger to vital table on passenger_id
            2.2 select all passenger_id, name where heart_rate_bpm, systolic_bp_mmHg, diastolic_bp_mmHg, spo2_pct, body_temp_c, respiration_rate_bpm are outside normal ranges

        3. select all passenger_id, name from passenger table
            3.1 join passenger to vital table on passenger_id
            3.2 select all passenger_id, name where heart_rate_bpm, systolic_bp_mmHg, diastolic_bp_mmHg, spo2_pct, body_temp_c, respiration_rate_bpm are at risk levels

        4. select all passenger_id, name from passenger table
           4.1 join passenger to vital table on passenger_id
           4.2 select all passenger_id, name where heart_rate_bpm, systolic_bp_mmHg, diastolic_bp_mmHg, spo2_pct, body_temp_c, respiration_rate_bpm are within normal ranges

        tools:
        - sql_checker: to check if the generated SQL query is correct. If not, rewrite the query based on the error message provided by the tool.
        - after generating the SQL query, use the sql_executor tool to run the query and get the results.

        '''

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"generate SQLite query to answer user question: {query}")
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
    user_quesrtion_2 = "is settler Tomoko Sato at risk based on her vital signs?"
    user_quesrtion_3 = "What is the average heart rate for passenger 18?"
    response = agent.invoke(user_quesrtion_3)
    print(response)