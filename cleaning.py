import pandas as pd
from datetime import datetime
from io import StringIO

# Load the main dataset
file_path = 'data/academic.csv'
data = pd.read_csv(file_path)

# Load the department reference dataset
department_file_path = 'data/department.csv'
departments = pd.read_csv(department_file_path)

# Create a dictionary mapping the first three characters to the full name for matching
correct_department_mapping = {name[:3].lower(): name for name in departments['name']}

# Function to clean and standardize time to 24-hour format
def clean_time(time_str):
    try:
        # If time contains AM/PM, parse accordingly
        if 'AM' in time_str or 'PM' in time_str:
            return datetime.strptime(time_str.strip(), '%I:%M:%S %p').strftime('%H:%M:%S')
        # Otherwise, assume it's already in 24-hour format
        else:
            return datetime.strptime(time_str.strip(), '%H:%M:%S').strftime('%H:%M:%S')
    except:
        return None  # Return None if parsing fails

# Function to clean and standardize dates to DD-MM-YYYY
def clean_date(date_str):
    try:
        # Handle various formats and standardize to DD-MM-YYYY
        return datetime.strptime(date_str.strip(), '%Y-%m-%d').strftime('%d-%m-%Y') \
            if '-' in date_str and len(date_str.split('-')[0]) == 4 \
            else datetime.strptime(date_str.strip(), '%d-%m-%Y').strftime('%d-%m-%Y')
    except:
        return None  # Return None if parsing fails

# Function to match and unify department names based on the first three characters
def match_by_first_three_chars(dept_name):
    if isinstance(dept_name, str):
        key = dept_name[:3].lower()
        return correct_department_mapping.get(key, dept_name)
    return dept_name

# Clean time and date columns
data['Hour'] = data['Hour'].apply(lambda x: clean_time(str(x)) if pd.notnull(x) else None)
data['strartingOrAsgnmtDate'] = data['strartingOrAsgnmtDate'].apply(lambda x: clean_date(str(x)) if pd.notnull(x) else None)

# Remove rows with missing or invalid time entries
data_cleaned = data.dropna(subset=['Hour', 'strartingOrAsgnmtDate'])

# Remove the 'Finish Time' column
data_cleaned = data_cleaned.drop(columns=['Finish Time'], errors='ignore')

# Unify department names in the 'SchoolDepartment' column
data_cleaned['SchoolDepartment'] = data_cleaned['SchoolDepartment'].apply(match_by_first_three_chars)

# Save the final cleaned and unified dataset
final_file_path = 'clean_dataset/Academics_Cleaned.csv'
data_cleaned.to_csv(final_file_path, index=False)

print(f"The final cleaned and unified dataset has been saved to: {final_file_path}")
