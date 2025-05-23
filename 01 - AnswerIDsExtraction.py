import os
import re
import csv

# Define the directory containing the CSV files
directory = 'dataset/ANSWERS/'

# List of CSV files to read (assuming they are the only CSV files in the directory)
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Initialize an empty list to hold log IDs
log_ids = []

# Regex pattern to extract the log ID within curly braces
pattern = re.compile(r'\{(.*?)\}')

# Loop over the CSV files and read each one
for file in csv_files:
    try:
        # Open the CSV file and read line by line
        with open(os.path.join(directory, file), 'r') as f:
            for line in f:
                # Search for all occurrences of the log ID pattern in the line
                matches = pattern.findall(line)
                for match in matches:
                    # Append the log ID (with curly braces) to the list
                    log_ids.append("{" + match + "}")
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Save the log IDs to a CSV file with a header
output_file = 'dataset/answer_ids.csv'
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id_anomaly'])  # Add header
    for log_id in log_ids:
        writer.writerow([log_id])

print("✅ IDs de anomalías guardados con encabezado en 'answer_ids.csv'.")




