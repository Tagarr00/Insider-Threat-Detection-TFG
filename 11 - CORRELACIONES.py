import pandas as pd

# Cargar el dataset
df = pd.read_csv("train_vectors.csv")

# Lista de métricas a correlacionar con 'anomaly'
columns_to_check = [
    'device_out_hours_flag',
    'panic_email_count',
    'job_external',
    'resignation_email',
    'threat_mail',
    
    'email_corporate_not_user',
    'http_leak_flag',
    'http_malware_page_flag',
    'file_virus_threat',
    'job_search_flag',
    
    'angry_email_flag',
    'events_out_hours_flag',
    'high_exe_ratio_flag',
    'vader_sentiment_flag',
    'device_lots_flag',

]

# Verificar que todas las columnas están presentes
missing = [col for col in columns_to_check + ['anomaly'] if col not in df.columns]
if missing:
    raise ValueError(f"Columnas faltantes en el CSV: {missing}")


correlation_matrix = df[['anomaly'] + columns_to_check].corr().loc[columns_to_check, ['anomaly']]
correlation_matrix.to_csv("train/correlacion_anomaly.csv")
print("✅ Matriz de correlación guardada en train/correlacion_anomaly.csv")



'''
# === ESCENARIO 1
escenario1_metrics = [
    #'working_out_hours_ratio',
    'device_out_hours_flag',
    'http_leak_flag',
    'events_out_hours_ratio',
    'anomaly'
]
df[escenario1_metrics].corr().to_csv("train/correlacion_matriz_escenario1.csv")
print("📁 Matriz Escenario 1 guardada en: train/correlacion_matriz_escenario1.csv")

# === ESCENARIO 2
escenario2_metrics = [
    'job_external',
    'resignation_email',
    'device_in',
    'job_search_flag',
    'anomaly'
]
df[escenario2_metrics].corr().to_csv("train/correlacion_matriz_escenario2.csv")
print("📁 Matriz Escenario 2 guardada en: train/correlacion_matriz_escenario2.csv")

# === ESCENARIO 3
escenario3_metrics = [
    'angry_email_count',
    'threat_mail',
    'http_malware_page_flag',
    'file_virus_threat',
    'panic_email_count',
    'files_exe',
    'negative_email_ratio',
    'anomaly'
]
df[escenario3_metrics].corr().to_csv("train/correlacion_matriz_escenario3.csv")
print("📁 Matriz Escenario 3 guardada en: train/correlacion_matriz_escenario3.csv")
'''