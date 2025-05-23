import pandas as pd
import os
import subprocess
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, RocCurveDisplay
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore", message="X does not have valid feature names")




#############################
from Features import feature
from Vectorizer import vectorize_session
from ExplainLime import explicar_con_lime 
from LOF import predict_anomaly
############################



# === Función para cargar el archivo de registros ===
def load_logs(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')  # Convertir fechas
    df = df.sort_values(by=['user', 'date', 'pc'])
    return df

# === Función para inicializar una sesión ===
def start_session(row):
    return {
        'id_sesion': row['id'],
        'user': row['user'],
        'pc': row['pc'],
        'logon': row['date'],
        'logoff': None,
        'logs': [],
        'devices': [], 
        'files': [], 
        'emails':[],
        'http':[],

        # ===== LDAP =====
        'name': '',
        'email': '',
        'role': '',
        'functional_unit': '',
        'department': '',
        'team': '',
        'supervisor': '',

        # ===== OCEAN =====
        'O': '',
        'C': '',
        'E': '',
        'A': '',
        'N': '',        

        # ===== Horarios de trabajo =====
        'is_working_hours': 0,
        'working_hours_ratio': 0,
        'working_out_hours_ratio': 0,
            

        # ===== Devices =====
        'device_in': 0,
        'device_out': 0,
        'devices_activity_out_hours': 0,
        'device_out_hours_flag': 0,
        'device_lots_flag':0,
        'device_ratio_flag': 0,

        # ===== Email =====
        'emails_total': 0,
        'email_count_sent_user': 0,
        'email_count_sent_external_account': 0,
        'email_external_to_count': 0,
        'bcc_external_count': 0,
        'bcc_total_count': 0,
        'email_attachment_count': 0,
        'email_attachment_size_total': 0,
        'email_unusual_time_count': 0,
        'email_corporate_not_user': 0,
        #'bert_pca1': 0,
        #'bert_pca2': 0,
        'vader_sentiment': 0,
        'risk_words_count': 0,
        'job_risk_words_count': 0,
        'importance_word_count': 0,
        'resignation_word_count': 0,
        'angry_email_count': 0,
        'sad_email_count': 0,
        'fear_email_count': 0,
        'panic_email_count': 0,
        'job_external': 0,
        'resignation_email': 0,
        'threat_mail' :0,
        'very_negative_vader_count': 0,
        'external_email_ratio': 0,
        'negative_email_ratio': 0,
        'max_email_size': 0,
        'email_unusual_time_ratio': 0,
            

        # ===== Files =====
        'files_total': 0,
        'files_exe': 0,
        'files_out_hours': 0,
        'files_unique': 0,
        'importance_file': 0,
        'risk_file': 0,
        'job_risk_file': 0,
        'file_virus_threat': 0,

        # ===== HTTP =====
        'http_total': 0,
        'http_out_hours': 0,
        'leak_url_count': 0,
        'malware_url_count': 0,
        'job_url_count': 0,
        'leak_httpcontent_count': 0,
        'malware_httpcontent_count': 0,
        'job_httpcontent_count': 0,
        'job_search': 0,
        'http_malware_page': 0,
        'http_malware_page_flag': 0,
        'http_leak_flag': 0,
        'job_search_flag': 0,

        # ===== Sentimiento =====
        'negative_emotion_count': 0,
        'negative_emotion_score': 0,
        'anger_score': 0,
        'fear_score': 0,
        'sadness_score': 0,

        # ===== Eventos generales =====
        'events': [],
        'event_count': 0,
        'events_in_hours': 0,
        'device_event_ratio': 0,
        'email_event_ratio': 0,
        'file_event_ratio': 0,
        'http_event_ratio': 0,
        'events_out_hours': 0,
        'device_oo_ratio': 0,
        'email_oo_ratio': 0,
        'file_oo_ratio': 0,
        'http_oo_ratio': 0,
        'events_oo_ratio': 0,
        'events_in_hours_ratio': 0,
        'events_out_hours_ratio': 0,

        # ===== Métricas de amenazas combinadas =====
        'device_ratio_anomaly': 0,
        'leak_threat': 0,
        'job_search_threat': 0,
        'resignation_danger': 0,
        'potential_virus_threat': 0,
        'job_flag_search': 0,
        'job_flag_email': 0,
        'job_search_and_email': 0,
            

         #################anomalia###########################
        'anomaly': 0
    }


# === Función para agregar un log a la sesión ===
def add_log(session, row):
    log_entry = {
        'date': str(row['date']),
        'activity': row['activity']
    }
    session['logs'].append(log_entry)

# === Función para gestionar anomalías ===
def update_anomaly(session, row):
    if row['anomaly'] == 1:
        session['anomaly'] = 1

# === Función para agregar un dispositivo a la sesión ===
def add_device(session, row):
    device_entry = {
        'date': str(row['date']),
        'activity': row['activity']
    }
    session['devices'].append(device_entry)
    
# === Función para agregar un file a la sesión ===
def add_file(session, row):
    clean = limpiar_content(row['content'])
    file_entry = {
        'date': str(row['date']),
        'filename': row['filename'],
        'content': clean
    }
    session['files'].append(file_entry)

def limpiar_content(valor):
    if pd.isna(valor) or not isinstance(valor, str):
        return ''
    partes = valor.split(' ', 1)  # divide solo en el primer espacio
    return partes[1] if len(partes) > 1 else ''  # si no hay texto después del código, devolver vacío

## === Función para agregar un email a la sesión ===
def add_email(session, row):
    #clean = limpiar_content(row['content'])
    email_entry = {
        'date': str(row['date']),
        'from': row['from'],
        'to': row['to'].split(';') if pd.notna(row['to']) else [],
        'cc': row['cc'].split(';') if pd.notna(row['cc']) else [],
        'bcc': row['bcc'].split(';') if pd.notna(row['bcc']) else [],
        'size': row['size'],
        'attachments': row['attachments'],
        'content': row['content']
    }
    session['emails'].append(email_entry)


# === Función para agregar un registro HTTP a la sesión ===
def add_http(session, row):
    http_entry = {
        'date': str(row['date']),
        'url': row['url'],
        'content': row['content']
    }
    session['http'].append(http_entry)

with open("model_def/feature_names.txt", "r") as f:
    feature_names = [line.strip() for line in f if line.strip()]
    
# === Función para gestionar sesiones ===
def manage_sessions(df):
    sessions = []
    active_sessions = {}
    logoff_without_logon = []
    
     # Contadores de rendimiento
    total = 0
    correctas = 0
    y_true = []
    y_proba = []
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    incorrectas = 0
    errores = []  # Para guardar los errores
    y_pred = []
    fn_metrics = []

    for _, row in df.iterrows():
        user = row['user']
        pc = row['pc']

        # Crear una sesión si el usuario no está activo
        if user not in active_sessions:
            active_sessions[user] = {}

        # Iniciar sesión al recibir un Logon
        if row['activity'] == 'Logon':
            if pc not in active_sessions[user]:
                # Crear nueva sesión
                active_sessions[user][pc] = start_session(row)
            # Añadir el log
            add_log(active_sessions[user][pc], row)
            # Actualizar la anomalía
            update_anomaly(active_sessions[user][pc], row)

        # Cerrar sesión al recibir un Logoff
        elif row['activity'] == 'Logoff':
            if pc in active_sessions[user]:
                # Añadir el log de Logoff
                add_log(active_sessions[user][pc], row)
                active_sessions[user][pc]['logoff'] = row['date']
                # Marcar como anómala si el logoff es anómalo
                update_anomaly(active_sessions[user][pc], row)

                ######################################################################
                active_sessions[user][pc] = feature(active_sessions[user][pc])
                
                
                # Vectorizar sesión
                vector = vectorize_session(active_sessions[user][pc])

                # Hacer predicción
                pred, score = predict_anomaly(vector)
                #pred = 1 if proba >= 0.5 else 0
                #matriz
                y_pred.append(pred)

                real = int(active_sessions[user][pc]['anomaly'])
                y_true.append(real)
                y_proba.append(score)                

                if pred == 1:
                    nombre_usuario = active_sessions[user][pc].get('name', 'Desconocido')
                    fecha_logon = active_sessions[user][pc].get('logon', 'fecha no disponible')
                    session_id = active_sessions[user][pc]['id_sesion']
                    #print(f"AMENAZA DETECTADA")
                    #print(f"🆔 Sesión: {fecha_logon}__{session_id} — PENDIENTE DE REVISIÓN.\n")                 
                    #print() 
                    
                    
                    session_data = active_sessions[user][pc]
                    explicar_con_lime(vector, session_id, session_data)

                # Contabilizar
                total += 1

                if pred == real:
                    correctas += 1
                    if real == 1:
                        TP += 1
                    else:
                        TN += 1
                else:
                    incorrectas += 1
                    if real == 1 and pred == 0:
                        FN += 1
                        metric_entry = {
                            "id_sesion": active_sessions[user][pc]['id_sesion'],
                            "user": active_sessions[user][pc]['user']
                        }
                        for fname in feature_names:
                            metric_entry[fname] = active_sessions[user][pc].get(fname, 0)
                        fn_metrics.append(metric_entry)
                        
                    elif real == 0 and pred == 1:
                        FP += 1

                # Guardar la sesión completa QUITAR? OJO QUIZAS NO PARA VER QEU SALEN BIEN LAS SESIONES PER DA IGUAL, AHROA MISMO ESTA COMENTADA Y DA ERROR
                sessions.append(active_sessions[user][pc])
                
                # Mostrar cada 500 sesiones creadas
                if len(sessions) % 1000 == 0:
                    print(f"----------------------------------- {len(sessions)} sesiones creadas--------------------------------------")
    
                del active_sessions[user][pc]
            else:
                # Registrar logoff sin logon
                logoff_without_logon.append({
                    'user': user,
                    'pc': pc,
                    'logoff': row['date'],
                    'error': 'Logoff sin logon previo'
                })

         # Manejo de dispositivos (Connect y Disconnect)
        elif row['activity'] in ['Connect', 'Disconnect']:
            if pc in active_sessions[user]:
                add_device(active_sessions[user][pc], row)
                update_anomaly(active_sessions[user][pc], row)

         # Manejo de files
        elif row['log_type'] == 'file':
            if pc in active_sessions[user]:
                add_file(active_sessions[user][pc], row)
                update_anomaly(active_sessions[user][pc], row)


        # Manejo de correos electrónicos
        elif row['log_type'] == 'email':
            if pc in active_sessions[user]:
                add_email(active_sessions[user][pc], row)
                update_anomaly(active_sessions[user][pc], row)

        # === Manejo de registros HTTP ===
        elif row['log_type'] == 'http':
            if pc in active_sessions[user]:
                add_http(active_sessions[user][pc], row)
                update_anomaly(active_sessions[user][pc], row)

    #if errores:
        #errores_df = pd.DataFrame(errores)
        #errores_df.to_csv("errores_modelo.csv", index=False)
        #print(f"🗂️ Errores guardados en: validation/errores_modelo.csv")

    # Mostrar métricas finales
    if total > 0:
        print(f"\n🎯 Evaluación del modelo:")
        print(f"   Total sesiones evaluadas: {total}")
        print(f"   ✔️ Aciertos: {correctas}")
        print(f"   ❌ Errores: {incorrectas}")
        
        accuracy = (TP + TN) / total
        precision_model = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall_model = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score = 2 * precision_model * recall_model / (precision_model + recall_model) if (precision_model + recall_model) > 0 else 0.0

        print(f"\n📊 Métricas detalladas:")
        print(f"   TP (True Positives): {TP}")
        print(f"   TN (True Negatives): {TN}")
        print(f"   FP (False Positives): {FP}")
        print(f"   FN (False Negatives): {FN}")
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   🔍 Precision: {precision_model:.4f}")
        print(f"   🎯 Recall: {recall_model:.4f}")
        print(f"   🧮 F1-Score: {f1_score:.4f}")

        # Calcular AUC solo si hay más de una clase en y_true
        if len(set(y_true)) > 1:
            auc_score = roc_auc_score(y_true, y_proba)
            print(f"   📐 AUC: {auc_score:.4f}")
        else:
            print(" ⚠️ No se puede calcular AUC (solo hay una clase presente)")
        
        # === Matriz de confusión ===
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Anomalía", "Anomalía"], yticklabels=["No Anomalía", "Anomalía"])
        plt.xlabel("Predicho")
        plt.ylabel("Real")
        plt.title("Matriz de Confusión")
        plt.tight_layout()
        plt.savefig("matriz_confusion.png")
        print("📸 Matriz de confusión guardada como 'matriz_confusion.png'")
        plt.close()
        
        # === Curva ROC y AUC ===
        if len(set(y_true)) > 1:
            RocCurveDisplay.from_predictions(y_true, y_proba)
            plt.title("Curva ROC")
            plt.savefig("curva_roc.png")
            print("Curva ROC guardada como 'curva_roc.png'")
            plt.close()
        if fn_metrics:
            df_fn = pd.DataFrame(fn_metrics)
            df_fn.to_csv("metricas_FN.csv", index=False)
            print("📁 Métricas de falsos negativos guardadas en 'metricas_FN.csv'")

    return sessions


# === Ejecución principal ===
def main():
    file_path = os.path.join("test/test_complete.csv")
    #file_path = os.path.join("validation/validation_complete.csv")
    #output_path = os.path.join("sesiones_features.csv")
    #output_path = os.path.join("validation/sesiones_validation.csv")

    print("🔄 Cargando registros...")
    df = load_logs(file_path)

    print("🚦 Creando sesiones...")
    sessions = manage_sessions(df)

    
    
    #print("💾 Guardando sesiones...")
    #save_sessions(sessions, output_path)
    
    
    print("\n✅ Proceso completado exitosamente.")

if __name__ == "__main__":
    main()

    
    
    
