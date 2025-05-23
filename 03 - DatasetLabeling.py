import pandas as pd

# Define file paths
labeled_file_path = 'dataset/dataset_unified.csv'
answer_ids_file_path = 'dataset/answer_ids.csv'

# Load the datasets
labeled_df = pd.read_csv(labeled_file_path)
answer_ids_df = pd.read_csv(answer_ids_file_path)

# Print the first few rows for debugging
print("Original labeled_df:")
print(labeled_df.head())

print("\nanswer_ids_df:")
print(answer_ids_df.head())

# Convert the answer_ids to a set for fast lookup
anomaly_ids = set(answer_ids_df['id_anomaly'])

# Check if each ID in labeled_df is in the anomaly set
labeled_df['anomaly'] = labeled_df['id'].isin(anomaly_ids).astype(int)

# Save the updated labeled dataset to a new CSV file
labeled_df.to_csv('dataset_bueno/dataset_complete.csv', index=False)

print("\n✅ Dataset completo guardado correctamente con las anomalías etiquetadas.")

'''
import pandas as pd

# Define file paths
labeled_file_path = 'dataset/answer_ids.csv'
answer_ids_file_path = 'dataset_unified.csv'

# Load the datasets
labeled_df = pd.read_csv(labeled_file_path)
answer_ids_df = pd.read_csv(answer_ids_file_path, header=None, names=['id'])

# Print the first few rows for debugging
print("Original labeled_df:")
print(labeled_df.head())

print("\nanswer_ids_df:")
print(answer_ids_df.head())

# Set the anomaly column to 1 for IDs present in answer_ids_df
labeled_df['anomaly'] = labeled_df.apply(
    lambda row: 1 if row['id'] in answer_ids_df['id'].values else row['anomaly'],
    axis=1
)

# Save the updated labeled dataset to the same CSV file
labeled_df.to_csv('dataset_complete', index=False)

'''