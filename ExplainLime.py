# ExplainLime.py
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
from joblib import load
import os
import pandas as pd
import json


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

def explicar_con_lime(vector, session_id, session_data, model_path="model_def/iforestHNP"): #cambair segun modelo 
    
    # Crear carpeta con formato fecha_id
    logon_raw = session_data.get('logon', 'fecha_desconocida')
    fecha = str(logon_raw).replace(":", "-").replace(" ", "_")
    session_id = session_data.get('id_sesion', 'id_desconocido')
    output_dir = os.path.join("XAI", f"{fecha}__{session_id}")
    os.makedirs(output_dir, exist_ok=True)

    
    # Cargar modelo, scaler y datos de entrenamiento
    #model = load(os.path.join(model_path, "knn_model.joblib"))
    #model = load(os.path.join(model_path, "lof_model.joblib"))
    model = load(os.path.join(model_path, "iforest_model.joblib")) 
    X_train = np.load(os.path.join(model_path, "X_all.npy"))

    with open("model_def/feature_names.txt") as f:
        feature_names = [line.strip() for line in f]

    def model_predict(X):
        scores = model.decision_function(X)
        preds = (scores > model.threshold_).astype(int)
        return np.column_stack([1 - preds, preds]) #DEBERIA DEOVLVER PROBABILIDAD

    # Validación del vector
    if len(vector) == len(pesos_totales) + 1:
        vector = vector[:-1]
    elif len(vector) != len(pesos_totales):
        raise ValueError(f"Vector recibido tiene {len(vector)} features, pero se esperaban {len(pesos_totales)}.")


    # Aplicar pesos directamente
    vector_weighted = np.array(vector) * np.array(pesos_totales)
    X_train_weighted = X_train * pesos_totales
    
    #print("objeto lime")
    explainer = LimeTabularExplainer(
        X_train_weighted,
        feature_names=feature_names,
        class_names=['Normal', 'Anomalía'],
        discretize_continuous=True,
        mode="classification"
    )

    #print("Lime antes")
    
    # === Generar explicación y guardar
    #exp = explainer.explain_instance(vector_weighted, model_predict, num_features=19)
    exp = explainer.explain_instance(vector_weighted, model_predict, num_features=len(pesos_totales), num_samples=1000)
    exp.save_to_file(os.path.join(output_dir, "lime_explicacion.html"))
    #print("Lime despues")

    # === Guardar vector original
    original_vector = vector
    with open(os.path.join(output_dir, "vector_original.txt"), "w") as f:
        for name, val in zip(feature_names, original_vector):
            f.write(f"{name}: {val}\n")

    # === Guardar datos del usuario
    campos = ['user', 'pc', 'name', 'email', 'role', 'functional_unit', 'department', 'team', 'supervisor', 'logon', 'logoff']
    with open(os.path.join(output_dir, "datos_usuario.txt"), "w") as f:
        for campo in campos:
            valor = session_data.get(campo, 'N/A')
            f.write(f"{campo}: {valor}\n")
            
    #print("Guarda")

    # === Guardar registros de sesión en un solo archivo de texto (más rápido que CSV)
    campos = ['logs', 'devices', 'files', 'emails', 'http']
    datos_sesion = {campo: session_data.get(campo, []) for campo in campos}

    with open(os.path.join(output_dir, "sesion.txt"), "w", encoding='utf-8') as f:
        for campo, registros in datos_sesion.items():
            f.write(f"== {campo.upper()} ==\n")
            for registro in registros:
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
            f.write("\n")
    #print("termina guardar")
    
     # === Guardar valor de anomalía
    anomaly_value = session_data.get('anomaly', 'N/A')
    with open(os.path.join(output_dir, "anomaly.txt"), "w") as f:
        f.write(str(anomaly_value))
    
    
