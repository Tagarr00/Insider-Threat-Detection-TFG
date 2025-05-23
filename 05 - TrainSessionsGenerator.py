import pandas as pd
import os
import subprocess
import sys



#############################
from Features import feature
from Vectorizer import vectorize_session

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
        
        'angry_email_flag': 0,
        'events_out_hours_flag': 0,
        'device_out_hours_flag': 0,
        'device_lots_flag':0,
        'high_exe_ratio_flag': 0,
        'vader_sentiment_flag': 0,

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

    
# === Función para gestionar sesiones ===
def manage_sessions(df):
    vectors = []
    #sessions = []
    active_sessions = {}
    logoff_without_logon = []

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
                #if active_sessions[user][pc]["anomaly"] == 1:
                #sessions.append(active_sessions[user][pc]) #guarda la sesion

                # Vectorizar sesión
                vector = vectorize_session(active_sessions[user][pc])
                vectors.append(vector) #guarda el vector
                
                # Guardar la sesión completa
               
                
                # Mostrar cada 500 sesiones creadas
                if len(vectors) % 1000 == 0:
                    print(f"🔹 {len(vectors)} vectores creados...")
    
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


    '''
    # Manejar sesiones activas que nunca cerraron
    for user, pcs in active_sessions.items():
        for pc, session in pcs.items():
            print(f"Sesión abierta sin logoff: Usuario: {session['user']}, PC: {session['pc']}, Logon: {session['logon']}")
            sessions.append(session)

    # Mostrar errores de logoff sin logon
    if logoff_without_logon:
        print("\nErrores de Logoff sin Logon:")
        for error in logoff_without_logon:
            print(f"Usuario: {error['user']}, PC: {error['pc']}, Logoff: {error['logoff']}")

    '''

    return vectors


def save_session(sessions, output_path):
    
    columnas = [
        'id_sesion', 'user', 'pc', 'logon', 'logoff', 'logs', 'devices', 'files',
        'emails', 'http', 'name', 'email', 'role', 'functional_unit', 'department', 'team',
        'supervisor', 'O', 'C', 'E', 'A', 'N', 'is_working_hours', 'working_hours_ratio',
        'working_out_hours_ratio', 'device_in', 'device_out', 'devices_activity_out_hours', 'emails_total', 'email_count_sent_user', 'email_count_sent_external_account', 'email_external_to_count',
        'bcc_external_count', 'bcc_total_count', 'email_attachment_count', 'email_attachment_size_total', 'email_unusual_time_count', 'email_corporate_not_user','sad_email_count','fear_email_count',
        'vader_sentiment', 'risk_words_count', 'job_risk_words_count', 'importance_word_count', 'resignation_word_count', 'angry_email_count', 'panic_email_count', 'job_external',
        'resignation_email', 'files_total', 'files_exe', 'files_out_hours', 'files_unique', 'importance_file', 'risk_file', 'job_risk_file',
        'file_virus_threat', 'http_total', 'http_out_hours', 'leak_url_count', 'malware_url_count', 'job_url_count', 'leak_httpcontent_count', 'malware_httpcontent_count',
        'job_httpcontent_count', 'job_search', 'negative_emotion_count', 'negative_emotion_score', 'anger_score', 'fear_score', 'sadness_score', 'events',
        'event_count', 'device_event_ratio', 'email_event_ratio', 'file_event_ratio', 'http_event_ratio', 'events_out_hours', 'device_oo_ratio', 'email_oo_ratio',
        'file_oo_ratio', 'http_oo_ratio', 'events_oo_ratio', 'device_ratio_anomaly', 'leak_threat', 'job_search_threat', 'resignation_danger', 'potential_virus_threat','job_flag_search', 'job_flag_email', 'http_malware_page'
        'anomaly'

    ]


    df = pd.DataFrame(sessions, columns=columnas)
    df.to_csv(output_path, index=False)
    print(f"✅ Sesiones guardadas en: {output_path}")
    
def save_vectors(vectors, output_path): ##editar este
    

    # === BINARY FLAGS ===
    all_columns_hybrid = [
        'devices_activity_out_hours',      # continua
        'panic_email_count',               # FLAG
        'job_external',                    # continua
        'resignation_email',              # FLAG
        'threat_mail',                    # FLAG

        'email_corporate_not_user',       # FLAG
        'http_leak_flag',                 # FLAG
        'http_malware_page_flag',         # FLAG
        'file_virus_threat',              # FLAG
        'job_search',                     # continua

        'angry_email_count',              # continua
        'events_out_hours',               # continua
        'files_exe',                      # continua
        'very_negative_vader_count',      # continua
        'device_in'                       # continua
    ]

    columnas = all_columns_hybrid + ['anomaly']

    df = pd.DataFrame(vectors, columns=columnas)
    df.to_csv(output_path, index=False)
    print(f"✅ Vectores guardados en: {output_path}")
    
    # Guardar sin 'anomaly'
    #output_path_sin = output_path.replace(".csv", "_noanomaly.csv")
    #df.drop(columns=['anomaly']).to_csv(output_path_sin, index=False)
    #print(f"✅ Vectores guardados SIN anomaly en: {output_path_sin}")

# === Ejecución principal ===
def main():

    #file_path = os.path.join("train/train_complete.csv")
    file_path = os.path.join("train/train_complete.csv")
    output_path = os.path.join("train_vectors.csv")
    #output_path2 = os.path.join("Mes-sessions.csv")
    
    print("🔄 Cargando registros...")
    df = load_logs(file_path)

    print("🚦 Creando sesiones...")
    vectors= manage_sessions(df)

    ###aqui meter una vectorizacion y que guarde el vector en otro lado
    
    
    print("💾 Guardando sesiones...")
    save_vectors(vectors, output_path)
    #save_session(sessions,output_path2)

    print("\n✅ Proceso completado exitosamente.")

if __name__ == "__main__":
    main()
