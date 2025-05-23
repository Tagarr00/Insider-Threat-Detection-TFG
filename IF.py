import numpy as np
from joblib import load

# === Cargar modelo IForest ===
try:
    model = load("model_def/iforestHNP/iforest_model.joblib")
except:
    model = None
    print("❌ Modelo IForest no cargado.")

# === Pesos personalizados ===

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
'''

def predict_anomaly(vector, return_proba=False):
    if model is None:
        print("⚠️ No hay modelo cargado.")
        return None

    if len(vector) == len(pesos_totales) + 1:  # puede venir con 'anomaly' al final
        vector = vector[:-1]
    elif len(vector) != len(pesos_totales):
        print(f"❌ ERROR EN MODEL: se esperaban {len(pesos_totales)} features y llegaron {len(vector)}")
        return None

    # Aplicar pesos
    features_scaled = np.array(vector) * np.array(pesos_totales)

    # Predecir con IForest
    score = model.decision_function([features_scaled])[0]  # cuanto mayor, más anómalo
    pred = int(score > model.threshold_)  # puedes cambiar esto por model.predict() si prefieres

    return pred, score
