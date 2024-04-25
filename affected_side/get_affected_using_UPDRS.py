import pandas as pd

def process_csv(input_file, output_file):
    # Load the CSV file, skipping the first two rows
    data = pd.read_csv(input_file, skiprows=2)
    
    # Assuming 'ID' is in the first column (Excel column A) and Excel's 'I' is column 9 in zero-indexing
    column_i = data.columns[8]  # This is the 9th column, corresponding to Excel's 'I'
    column_f = data.columns[5]  # This is the 6th column, corresponding to Excel's 'F'
    
    # Convert the columns to numeric, dropping any non-numeric rows
    data[column_i] = pd.to_numeric(data[column_i], errors='coerce')
    data[column_f] = pd.to_numeric(data[column_f], errors='coerce')

    # Drop rows where either 'I' or 'F' is NaN after conversion
    data = data.dropna(subset=[column_i, column_f])

    # Create a new DataFrame to store the filtered data
    new_data = []

    # Iterate through the data and compare 'I' and 'F'
    for index, row in data.iterrows():
        if row[column_i] > row[column_f]:
            new_data.append([row['ID'],  'Right'])
        elif row[column_i] < row[column_f]:
            new_data.append([row['ID'],  'Left'])
        # Rows where 'I' == 'F' are ignored

    # Save the new DataFrame to a new CSV file
    new_df = pd.DataFrame(new_data, columns=['ID', 'Label'])
    new_df.to_csv(output_file, index=False)

# Usage
input_file = '\\\\files.ubc.ca\\team\\PPRC\\Camera\\Video Assessment_Atefeh\\Video_quality_feed\\CAMERA Study Booth - Tracking Log_RM-No Names.csv'
output_file = '\\\\files.ubc.ca\\team\\PPRC\\Camera\\Video Assessment_Atefeh\\Video_quality_feed\\side.csv'
process_csv(input_file, output_file)
