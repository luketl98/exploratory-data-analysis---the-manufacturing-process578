import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError


# Class to handle database connections and operations
class RDSDatabaseConnector:
    """
    A class to handle database connections and operations for an Amazon RDS instance.

    This class is responsible for connecting to a relational database using SQLAlchemy
    and fetching data using provided SQL queries.

    Attributes:
        engine (sqlalchemy.engine.Engine): The SQLAlchemy engine used to connect to
                                           the RDS database.

    Example Usage:
        credentials = {
            "RDS_USER": "your_username",
            "RDS_PASSWORD": "your_password",
            "RDS_HOST": "your_host",
            "RDS_PORT": "1234",
            "RDS_DATABASE": "your_database"
        }
        connector = RDSDatabaseConnector(credentials)
        df = connector.fetch_data("SELECT * FROM table_name")
        connector.close_connection()
    """

    def __init__(self, credentials: dict[str, str]) -> None:
        """
        Initialise the RDSDatabaseConnector with database credentials.

        Parameters:
            credentials (dict): A dictionary containing the database
                                connection credentials.
                - 'RDS_USER': Username for the RDS database.
                - 'RDS_PASSWORD': Password for the RDS database.
                - 'RDS_HOST': Host address for the RDS database.
                - 'RDS_PORT': Port number for the RDS database.
                - 'RDS_DATABASE': The database name to connect to.

        Returns:
            None
        """

        # Create the SQLAlchemy engine to connect to the database
        self.engine = create_engine(
            f"postgresql://{credentials['RDS_USER']}:"
            f"{credentials['RDS_PASSWORD']}@"
            f"{credentials['RDS_HOST']}:{credentials['RDS_PORT']}/"
            f"{credentials['RDS_DATABASE']}"
        )

    def fetch_data(self, query: str) -> pd.DataFrame:
        """
        Fetch data from the database by executing the provided SQL query.

        Parameters:
            query (str): The SQL query to execute.

        Returns:
            pd.DataFrame: A DataFrame containing the query results.

        Raises:
            SQLAlchemyError: If an error occurs while executing the SQL query.
            Exception: For any other unexpected errors.
        """
        try:
            # Establish a connection and execute the SQL query
            with self.engine.connect() as connection:
                result = pd.read_sql(query, connection)
            return result
        except SQLAlchemyError as e:
            # Handle SQLAlchemy-specific errors and return an empty DataFrame
            print(f"An error occurred while executing the query: {e}")
            return pd.DataFrame()
        except Exception as e:
            # Handle unexpected errors and return an empty DataFrame
            print(f"An unexpected error occurred: {e}")
            return pd.DataFrame()

    def close_connection(self) -> None:
        """
        Close the database connection by disposing of the SQLAlchemy engine.

        This method should be called to clean up the connection after all
        operations are completed.

        Returns:
            None
        """
        # Dispose of the engine to release any resources held by the connection
        self.engine.dispose()
