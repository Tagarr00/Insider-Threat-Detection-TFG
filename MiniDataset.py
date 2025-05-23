import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('dataset_bueno/dataset_complete.csv')

# Assuming the date column is named 'date' and is in a standard date format
df['date'] = pd.to_datetime(df['date'])

# Define the start and end dates
start_date = '2010-07-01'
end_date = '2010-08-01'

# Convert start_date and end_date to datetime format
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter the DataFrame to include only rows between the specified dates
filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

# Convert datetime column back to original string format
# filtered_df['date'] = filtered_df['date'].dt.strftime('%m/%d/%Y %H:%M:%S')

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv('Mes-prueba.csv', index=False)