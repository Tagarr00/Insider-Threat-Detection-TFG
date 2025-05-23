import pandas as pd

# Define file paths and corresponding log types
files_and_types = [
    ('dataset/logon.csv', 'logon'),
    ('dataset/device.csv', 'device'),
    ('dataset/http.csv', 'http'),
    ('dataset/email.csv', 'email'),
    ('dataset/file.csv', 'file')
]

# Initialize an empty list to hold dataframes
dfs = []

#Iterate over files and types, read each CSV, add log_type column, and append to the list
for file_path, log_type in files_and_types:
    df = pd.read_csv(file_path, dtype={'id': str, 'date': str})
    df['log_type'] = log_type
    '''
    # If the log type is HTTP, drop the 'content' column if it exists
    if log_type == 'http' and 'content' in df.columns:
        df = df.drop(columns=['content'])
    '''
    dfs.append(df)

# Concatenate all dataframes in the list
combined_df = pd.concat(dfs, ignore_index=True)

# Add the "anomaly" column and set all values to 0
combined_df['anomaly'] = 0

# Convert 'date' column to datetime for efficient sorting
combined_df['date'] = pd.to_datetime(combined_df['date'])

# Sort the combined dataframe by the "date" column
combined_df.sort_values(by='date', inplace=True)

# Reset index
combined_df.reset_index(drop=True, inplace=True)

# Convert 'date' column back to its original string format
combined_df['date'] = combined_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Save the combined and sorted dataset to a new CSV file
combined_df.to_csv('dataset/dataset_unified.csv', index=False)

# Print the first few rows of the combined dataset
print(combined_df.head())
