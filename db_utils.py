import pandas as pd
import yaml
from sqlalchemy import create_engine


def load_data_from_csv(file_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Print the shape of the DataFrame
    print("Data shape:", data.shape)

    # Print 1st few rows of DataFrame to understand what the data looks like
    print("First few rows of data:")
    print(data.head())

    # Return the DataFrame
    return data


# Function to load database credentials from a YAML file
def load_db_credentials(file_path='credentials.yaml'):
    with open(file_path, 'r') as file:
        credentials = yaml.safe_load(file)
    return credentials


# Class to handle database connections and operations
class RDSDatabaseConnector:
    def __init__(self, credentials):
        self.engine = create_engine(
            f"postgresql://{credentials['RDS_USER']}:"
            f"{credentials['RDS_PASSWORD']}@"
            f"{credentials['RDS_HOST']}:{credentials['RDS_PORT']}/"
            f"{credentials['RDS_DATABASE']}"
        )

    def fetch_data(self, query):
        # Fetches data from the database and returns a Pandas DataFrame
        with self.engine.connect() as connection:
            result = pd.read_sql(query, connection)
        return result

    def save_data_to_csv(self, data_frame, filename):
        # Saves a DataFrame to a CSV file
        data_frame.to_csv(filename, index=False)

    def close_connection(self):
        # Closes the database connection
        self.engine.dispose()

# class DataTransform:


if __name__ == "__main__":
    credentials = load_db_credentials('credentials.yaml')
    db_connector = RDSDatabaseConnector(credentials)

    # Define the SQL query for the required data
    query = "SELECT * FROM failure_data;"  # Adjust this query as needed

    # Fetch data from the database
    data = db_connector.fetch_data(query)

    # Save the fetched data to a CSV file, if data is not empty
    if not data.empty:
        csv_filename = 'failure_data.csv'
        db_connector.save_data_to_csv(data, csv_filename)
        print("Data successfully saved to 'failure_data.csv'.")

        # Load the data from the CSV file and print its characteristics
        print("\nLoading data from CSV to verify contents:")
        loaded_data = load_data_from_csv(csv_filename)

    else:
        print("No data was fetched from the database.")

    # Close the database connection
    db_connector.close_connection()
