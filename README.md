# EDA - The Manufacturing Process

## Table of Contents
- [Project Description](#project-description)
- [Installation Instructions](#installation-instructions)
- [Usage Instructions](#usage-instructions)
- [File Structure](#file-structure)
- [License](#license)

## Project Description
This project aims to optimise a manufacturing machine process by leveraging exploratory data analysis (EDA) to identify key factors affecting machine performance and suggest improvements. The project is structured around a comprehensive EDA pipeline, encapsulated within the `EDAExecutor` class, which facilitates data fetching, transformation, analysis, and visualisation.

### Key Features:

1. **Data Fetching and Storage**:
   - The project connects to a database using the `RDSDatabaseConnector` class to fetch data based on SQL queries.

2. **Data Reformatting**:
   - The `reformat_data` method ensures data is in the correct format, converting columns to appropriate data types such as categorical and boolean.

3. **Statistical Exploration**:
   - The `explore_stats` method provides a detailed statistical overview of the dataset, including column descriptions, distinct value counts, and null value analysis.

4. **Data Visualisation**:
   - The `visualise_data` method generates various plots, including scatter plots, histograms, bar plots, and correlation heatmaps, to provide insights into data relationships and distributions.

5. **Null Imputation**:
   - The `handle_null_imputation` method identifies and optionally imputes missing values in the dataset, focusing on specific machine settings.

6. **Skewness Transformation**:
   - The `handle_skewness_transformation` method identifies and optionally transforms skewed data using a method of choice depending on user input, focusing on specific machine settings.

7. **Outlier Detection and Handling**:
   - The `handle_outlier_detection` method identifies and optionally removes outliers from the dataset depending on user input, focusing on specific machine settings.

8. **Drop Columns and Save Data**:
   - The `drop_columns_and_save_data` method drops columns that are not required for further analysis and saves the transformed data to a CSV file.

9. **Further Analysis**:
   - The `analyse_further` method carries out more in-depth analysis on the transformed data, including further statistical tests, visualisations such as violin plots and boxplots, analysis of operating ranges, failure risk factors, and regression analysis and more.

10. **Machine Setting Calculator**:
   - The `MachineSettingCalculator` class allows the user to calculate the number and percentage of sessions above and below a user-defined minimum and maximum value for a given machine setting. A table is generated to display the results.


10. **Comprehensive EDA Pipeline**:
   - The main block of the script allows for a step-by-step execution of the EDA process, controlled by flags to enable or disable specific steps such as data reformatting, statistical exploration, visualisation, null imputation, skewness transformation, outlier detection, and further analysis.

### Technologies Used:
- **Python Libraries**: The project utilises a range of Python libraries, including `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `SQLAlchemy`, `missingno`, `plotly`, `statsmodels`, and `tabulate` for data manipulation, analysis, and visualisation.

- **Database Connectivity**: SQLAlchemy is used for database interactions, data retrieval and manipulation.

This project provides a robust framework for conducting EDA on manufacturing data, offering insights into machine performance and potential areas for optimisation.

## Installation Instructions

1. **Clone the Repository**:
   - Open your terminal or command prompt.
   - Run the following command to clone the repository:
     ```bash
     git clone https://github.com/luketl98/exploratory-data-analysis---the-manufacturing-process578.git
     ```

2. **Navigate to the Project Directory**:
   - Change into the project directory:
     ```bash
        cd exploratory-data-analysis---the-manufacturing-process578
     ```

3. **Install Required Python Packages**:
   - Ensure you have Python and pip installed on your system.
   - Run the following command to install all necessary dependencies:
     ```bash
     pip install -r requirements.txt
     ```

4. **Set Up Database Credentials**:
   - Ensure you have a `credentials.yaml` file in the project directory with your database credentials.

5. **Run the Project**:
   - Execute the main script to start the EDA process:
     ```bash
     python main.py
     ```
   - Follow any on-screen instructions to proceed with the analysis.

## Usage Instructions
To effectively use the EDA pipeline for analysing manufacturing data, follow these steps:

1. **Prepare Your Environment**:
   - Ensure you have cloned the repository and installed all necessary dependencies as outlined in the installation instructions.

2. **Set Up Database Credentials**:
   - Create a credentials.yaml file in the project directory with your database credentials. This file should include keys such as RDS_USER, RDS_PASSWORD, RDS_HOST, RDS_PORT, and RDS_DATABASE.

3. **Run the EDA Process**:
   - Execute the main script to start the EDA process:
     ```bash
     python main.py
     ```
   - The script is designed to run through a series of data analysis steps, which can be controlled via flags in the main block of the script. These steps include data reformatting, statistical exploration, visualisation, null imputation, skewness transformation, outlier detection, and further analysis.

4. **Interact with the Script**:
   - Follow any on-screen instructions to proceed with the analysis. The script may prompt you to make choices regarding data transformations or outlier handling.

5. **Review the Results**:
   - The script will generate various visualisations and statistical summaries to help you understand the data. These include scatter plots, histograms, bar plots, correlation heatmaps, and more.

6. **Save and Utilise Transformed Data**:
   - The transformed data can be saved to CSV files for further analysis or reporting. The script will handle this automatically if the corresponding flag is set.

7. **Conduct Further Analysis**:
   - Use the further_analysis method to perform in-depth analysis on the transformed data, including operating range analysis, failure analysis, and risk factor investigation.

## File Structure
- `db_utils.py` - Contains functions to connect to the database.
- `executor.py` - Contains the `EDAExecutor` class for running EDA processes.
- `main.py` - The main script to execute the EDA pipeline.
- `transformations.py` - Contains classes and functions for data transformations.
- `visualisations.py` - Contains the `Plotter` class for data visualisations.
- `calculator.py` - Contains the `MachineSettingCalculator` class for machine setting calculations.
- `df_info.py` - Contains classes and functions for data frame information and statistics.
- `README.md` - Provides project documentation.
- `.gitignore` - Specifies intentionally untracked files to ignore.
- `credentials.yaml` - Stores database credentials (ensure this file is not tracked by git for security reasons).
- `requirements.txt` - Lists all Python dependencies required for the project.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.
