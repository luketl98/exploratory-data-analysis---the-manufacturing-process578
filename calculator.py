import pandas as pd
from tabulate import tabulate


class MachineSettingCalculator:
    """
    Class to handle machine setting calculations and analysis.

    This class provides methods to calculate session statistics based on
    machine settings, initialise and update settings, and display results
    in a tabular format.
    """

    def __init__(self, pre_transform_data: pd.DataFrame) -> None:
        """
        Initialise the MachineSettingCalculator with pre-transformed data.

        Parameters:
            pre_transform_data (pd.DataFrame): The DataFrame containing
            pre-transformed data for analysis.
        """
        self.pre_transform_data = pre_transform_data

    def calculate_setting_limit(
        self, data: pd.DataFrame, column: str, min_value: float = None,
        max_value: float = None
    ) -> dict:
        """
        Calculate the number and percentage of sessions above and below given
        min and max thresholds for a specified machine setting column.

        Parameters:
            data (pd.DataFrame): The DataFrame containing the data.
            column (str): The column to analyse.
            min_value (float, optional): The minimum threshold value.
            max_value (float, optional): The maximum threshold value.

        Returns:
            dict: A dictionary containing the calculated statistics, including
            sessions missed and failed sessions for both min and max thresholds.
        """
        total_sessions = len(data)
        total_failed_sessions = data[data["Machine failure"]].shape[0]

        # Calculate sessions below the minimum threshold, if specified
        sessions_missed_min = (
            data[data[column] < min_value].shape[0] if min_value is not None else 0
        )

        # Calculate sessions above the maximum threshold, if specified
        sessions_missed_max = (
            data[data[column] > max_value].shape[0] if max_value is not None else 0
        )

        # Calculate percentages for sessions missed
        percentage_missed_min = (
            (sessions_missed_min / total_sessions) * 100 if min_value is not None else 0
        )
        percentage_missed_max = (
            (sessions_missed_max / total_sessions) * 100 if max_value is not None else 0
        )

        # Calculate failed sessions below the minimum threshold
        failed_sessions_min = (
            data[(data[column] < min_value) & data["Machine failure"]].shape[0]
            if min_value is not None
            else 0
        )

        # Calculate failed sessions above the maximum threshold
        failed_sessions_max = (
            data[(data[column] > max_value) & data["Machine failure"]].shape[0]
            if max_value is not None
            else 0
        )

        # Calculate percentages for failed sessions
        percentage_failed_sessions_min = (
            (failed_sessions_min / total_failed_sessions) * 100
            if min_value is not None
            else 0
        )
        percentage_failed_sessions_max = (
            (failed_sessions_max / total_failed_sessions) * 100
            if max_value is not None
            else 0
        )

        return {
            "min": min_value,
            "max": max_value,
            "sessions_missed_min": sessions_missed_min,
            "percentage_missed_min": percentage_missed_min,
            "sessions_missed_max": sessions_missed_max,
            "percentage_missed_max": percentage_missed_max,
            "failed_sessions_min": failed_sessions_min,
            "failed_sessions_max": failed_sessions_max,
            "percentage_failed_sessions_min": percentage_failed_sessions_min,
            "percentage_failed_sessions_max": percentage_failed_sessions_max,
        }

    def initialise_machine_settings(self) -> dict:
        """
        Initialise machine settings with descriptive names.

        Returns:
            dict: A dictionary mapping setting keys to descriptive names.
        """
        return {
            "a": "Torque [Nm]",
            "b": "Rotational speed [rpm]",
            "c": "Tool wear [min]",
            "d": "Process temperature [K]",
            "e": "Air temperature [K]",
        }

    def initialise_table(self, machine_settings: dict) -> dict:
        """
        Initialise a table with min and max values set to None for each setting.

        Parameters:
            machine_settings (dict): A dictionary of machine settings.

        Returns:
            dict: A dictionary with settings as keys and min/max values as None.
        """
        return {
            setting: {"min": None, "max": None} for setting in machine_settings.values()
        }

    def display_table(self, table: dict) -> None:
        """
        Display the table with calculated statistics.

        Parameters:
            table (dict): A dictionary containing machine settings and their
            min/max values.
        """
        table_data = []
        for setting, values in table.items():
            # Calculate statistics for each setting using the current min/max values
            stats = self.calculate_setting_limit(
                self.pre_transform_data, setting, values["min"], values["max"]
            )

            # Append statistics for the minimum threshold
            table_data.append(
                [
                    setting,
                    f"Min: {values['min']}" if values["min"] is not None else "N/A",
                    stats["sessions_missed_min"],
                    f"{stats['percentage_missed_min']:.2f}",
                    stats["failed_sessions_min"],
                    f"{stats['percentage_failed_sessions_min']:.2f}",
                ]
            )

            # Append statistics for the maximum threshold
            table_data.append(
                [
                    "",
                    f"Max: {values['max']}" if values["max"] is not None else "N/A",
                    stats["sessions_missed_max"],
                    f"{stats['percentage_missed_max']:.2f}",
                    stats["failed_sessions_max"],
                    f"{stats['percentage_failed_sessions_max']:.2f}",
                ]
            )

        # Define headers for the table
        headers = [
            "Setting",
            "Min / Max",
            "Sessions\nMissed",
            "% Missed",
            "Failed Sessions\nAvoided",
            "% Failed\nAvoided",
        ]
        # Print the table using tabulate for a formatted grid display
        print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".2f"))

    def update_setting(self, machine_settings: dict, table: dict) -> None:
        """
        Update the min or max value for a selected machine setting.

        Parameters:
            machine_settings (dict): A dictionary of machine settings.
            table (dict): A dictionary containing machine settings and their
            min/max values.
        """
        print("\nSelect a machine setting to update:\n")
        # Display available machine settings for selection
        for key, setting in machine_settings.items():
            print(f"{key}: {setting}")

        # Prompt user to select a setting to update
        selected_key = input("\nEnter a, b, c, d, or e: ").lower()

        if selected_key in machine_settings:
            selected_setting = machine_settings[selected_key]
            # Ask whether to update the min or max value
            min_or_max = input("\nDo you want to set a 'min' or 'max' value? ").lower()

            if min_or_max in ["min", "max"]:
                while True:
                    try:
                        # Prompt user to enter a numeric value for the selected setting
                        value = float(
                            input(
                                f"\nEnter the {min_or_max} value for "
                                f"{selected_setting}: "
                            )
                        )
                        table[selected_setting][min_or_max] = value
                        break
                    except ValueError:
                        print("\nInvalid input. Please enter a numeric value.")
            else:
                print("\nInvalid selection. Please enter 'min' or 'max'.")
        else:
            print("\nInvalid selection. Please try again.")

    def continue_analysis_prompt(self) -> bool:
        """
        Prompt the user to continue or end the analysis.

        Returns:
            bool: True if the user wants to continue, False otherwise.
        """
        while True:
            # Ask user if they want to update another setting
            continue_analysis = input(
                "\nDo you want to update another setting? (yes/no): "
            ).lower()
            if continue_analysis in ["yes", "no"]:
                return continue_analysis == "yes"
            else:
                print("\nInvalid input. Please enter 'yes' or 'no'.")

    def run_machine_setting_calculator(self) -> None:
        """
        Run the machine setting analysis process.

        This method initialises machine settings, displays the table of
        statistics, and allows the user to update settings and continue
        analysis until they choose to stop.
        """
        # Initialise machine settings and table
        machine_settings = self.initialise_machine_settings()
        table = self.initialise_table(machine_settings)

        while True:
            # Display total number of sessions and failed sessions
            print(f"\nTotal number of sessions: {len(self.pre_transform_data)}")
            total_failed_sessions = self.pre_transform_data[
                self.pre_transform_data["Machine failure"]
            ].shape[0]
            print(f"Total number of failed sessions: {total_failed_sessions}")

            # Display the current table of statistics
            self.display_table(table)
            # Allow user to update settings
            self.update_setting(machine_settings, table)

            # Check if the user wants to continue updating settings
            if not self.continue_analysis_prompt():
                # Final display of total sessions and failed sessions
                print(f"\nTotal number of sessions: {len(self.pre_transform_data)}")
                print(f"Total number of failed sessions: {total_failed_sessions}")
                print("\nFinal Table:")
                self.display_table(table)
                break
