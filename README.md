# Project Title

## Table of Contents
- [Project Description](#project-description)
- [Installation Instructions](#installation-instructions)
- [Usage Instructions](#usage-instructions)
- [File Structure](#file-structure)
- [License](#license)

## Project Description
This project is designed to optimise a manufacturing machine process to increase efficiency and profitability. The aim is to use exploratory data analysis (EDA) to identify key factors that affect machine performance and suggest improvements. Utilising packages such as; matplotlib, numpy, pandas, seaborn, scipy, SQLalchemy, to visualise data using a range of methods, extract and display information for EDA, impute missing data, apply transformations and more.

## Installation Instructions
1. Clone the repository: git clone #TODO Add git repository here
2. Install required Python packages: pip install -r requirements.txt --- #TODO ADD requirements here?

## Usage instructions 
To use the new feature for inputting min. or max. values for machine settings and calculating the number and percentage of sessions above and below that value, follow these steps:

1. Ensure you have the necessary data loaded into the DataFrame.
2. Create an instance of the `EDAExecutor` class.
3. Call the `calculate_sessions_above_below` method with the appropriate parameters.

Example usage:
```python
from db_utils import EDAExecutor, load_db_credentials, RDSDatabaseConnector

# Load database credentials and connect
credentials = load_db_credentials("credentials.yaml")
db_connector = RDSDatabaseConnector(credentials)

# Create an instance of EDAExecutor
eda_executor = EDAExecutor(db_connector)

# Fetch and save data
data = eda_executor.fetch_and_save_data(query="SELECT * FROM failure_data;")

# Calculate sessions above and below a threshold for a specific column
result = eda_executor.calculate_sessions_above_below(data, column="Torque [Nm]", threshold=50)

# Print the result
print(result)
```

## File Structure
- `db_utils.py` - Contains functions to connect to the database and perform data analysis.
- `README.md` - Provides project documentation.
- `.gitignore` - Specifies intentionally untracked files to ignore.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.
