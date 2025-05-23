import os
import pandas as pd
import numpy as np
from pyod.models.iforest import IForest  
from joblib import dump
'''
pesos_totales = [
    1.25,   # devices_activity_out_hours (continua)
    8.0,   # panic_email_count (flag)
    2.0,   # job_external (continua)
    8.0,   # resignation_email (flag)
    8.0,   # threat_mail (flag)

    8.0,   # email_corporate_not_user (flag)
    8.0,   # http_leak_flag (flag)
    8.0,   # http_malware_page_flag (flag)
    8.0,   # file_virus_threat (flag)
    2.0,  # job_search (continua) antes 2 el mejor

    3.0,   # angry_email_count (continua)
    1.25,  # events_out_hours (continua)
    0.5,   # files_exe (continua)
    0.0,   # very_negative_vader_count (continua)
    1.25   # device_in (continua) antes 1
]

'''
pesos_totales = [
    1.0,  # devices_activity_out_hours (continua)
    1.0,  # panic_email_count (flag)
    1.0,  # job_external (continua)
    1.0,  # resignation_email (flag)
    1.0,  # threat_mail (flag)

    1.0,  # email_corporate_not_user (flag)
    1.0,  # http_leak_flag (flag)
    1.0,  # http_malware_page_flag (flag)
    1.0,  # file_virus_threat (flag)
    1.0,  # job_search (continua)

    1.0,  # angry_email_count (continua)
    1.0,  # events_out_hours (continua)
    1.0,  # files_exe (continua)
    0.0,  # very_negative_vader_count (continua)
    1.0,  # device_in (continua)
]


# === COLUMNAS A USAR (coordinado con vectorize_session) ===
all_columns = [
    'devices_activity_out_hours',
    'panic_email_count',
    'job_external',
    'resignation_email',
    'threat_mail',

    'email_corporate_not_user',
    'http_leak_flag',
    'http_malware_page_flag',
    'file_virus_threat',
    'job_search',

    'angry_email_count',
    'events_out_hours',
    'files_exe',
    'very_negative_vader_count',
    'device_in'
]


# === Crear carpeta de modelos ===
os.makedirs("model_def", exist_ok=True)

# === Cargar dataset ===
df = pd.read_csv("train_vectors.csv")

if "anomaly" in df.columns:
    df = df.drop(columns=["anomaly"])  # No debe usarse en entrenamiento

# === Verificación de columnas ===
print("✅ Verificando columnas antes del entrenamiento...\n")
print("Desde el DataFrame:", list(df[all_columns].columns))
print("Desde all_columns :", all_columns)
print()

if list(df[all_columns].columns) != all_columns:
    print("❌ ERROR: Las columnas no coinciden con el orden esperado.")
    exit()
else:
    print("✅ Columnas correctas y en el orden esperado.\n")

# === Aplicar pesos
X_total = df[all_columns].values
X_final = X_total * np.array(pesos_totales)

# === Guardar nombres de features
with open("model_def/feature_names.txt", "w") as f:
    for name in all_columns:
        f.write(f"{name}\n")

# === Entrenamiento con Isolation Forest
print("\n Entrenando modelo Isolation Forest...")
#model = IForest(contamination=0.1567)

# Generar una semilla aleatoria
random_state = np.random.randint(0, 2**32 - 1)
print(f"🎲 Usando random_state = {random_state}")

model = IForest(contamination=0.1567)
model.fit(X_final)

labels = model.labels_
scores = model.decision_scores_

model_dir = "model_def/iforestHNP"
os.makedirs(model_dir, exist_ok=True)

dump(model, os.path.join(model_dir, "iforest_model.joblib"))
np.save(os.path.join(model_dir, "X_all.npy"), X_final)
np.save(os.path.join(model_dir, "y_all.npy"), labels)
np.save(os.path.join(model_dir, "decision_scores.npy"), scores)

# === Guardar la semilla usada
with open(os.path.join(model_dir, "random_state.txt"), "w") as f:
    f.write(str(random_state))

print(f"✅ Modelo IForest guardado en '{model_dir}'")