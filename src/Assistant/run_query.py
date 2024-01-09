"""
This module contains the RunSQLQuery class, which represents:
a function that executes a SQL query on the specified SQLite database.
"""

import sqlite3
import os
import json

from openai_assistant_helper import AIAssistant
from openai_function_helper import Function, Property

with open("api_info.json", "r", encoding="utf-8") as config_file:
    config = json.load(config_file)
    OPENAI_API_KEY = config["openai_api_key"]
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    

class GetDBSchema(Function):
    """
    Represents a function that returns the schema of the database.
    Returns:
        None
    """
    def __init__(self):
        """
        Initializes a new instance of the GetDbSchema class.

        Args:
            sql_path (str): The path to the SQLite database.

        Returns:
            None
        """
        super().__init__(
            name="get_db_schema",
            description="Get the schema of the database",
            parameters=[Property(
                    name="sql_path",
                    description="The path to the SQLite database",
                    type="string",
                    required=True,
                )
            ]
        )
    def func(self, sql_path):
        """
        Retrieves the SQL statements used to create all tables in the given SQLite database file.

        Parameters:
            sql_path (str): The path to the SQLite database file.

        Returns:
            str: A string containing the SQL statements used to create all tables in the database.
        """
        conn = sqlite3.connect(sql_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        create_statements = cursor.fetchall()
        conn.close()
        return '\n\n'.join([statement[0] for statement in create_statements])
    
class RunSQLQuery(Function):
    """
    Run a SQL query on the database.

    :param query: The SQL query to run.
    :type query: string
    :param database: The database to run the query on.
    :type database: string
    """
    def __init__(self):
        """
        Initialize the run_sql_query class.

        :param query: The SQL query to run.
        :type query: string
        :param database: The database to run the query on.
        :type database: string
        """
        super().__init__(
            name="run_sql_query",
            description="Run a SQL query on the database",
            parameters=[
                Property(
                    name="query",
                    description="The SQL query to run",
                    type="string",
                    required=True,
                ),
                Property(
                    name="database",
                    description="The database to run the query on",
                    type="string",
                    required=True,
                ),
            ]
        )

    def func(self, query, database):
        """
        Executes a SQL query on the specified SQLite database.

        Parameters:
            query (str): The SQL query to execute.
            database (str): The SQLite database to run the query on.

        Returns:
            str: A string representation of the query results, separated by newlines.
        """
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return '\n'.join([str(result) for result in results])


assistant = AIAssistant(instruction="""You are a SQL expert. User asks you questions about the database.
                        First obtain the schema of the database to check the tables and columns, then generate SQL queries to answer the questions.
                        """,
                        model="gpt-3.5-turbo-1106",
                        functions=[GetDBSchema(), RunSQLQuery()],
                        use_code_interpreter=True, # Using the code interperter to generate CSV files
                        )

if __name__ == "__main__":
    assistant.chat()
