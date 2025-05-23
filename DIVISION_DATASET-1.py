import pandas as pd
import os
import csv
import ast

# === Función para cargar el archivo de registros ===
def load_logs(file_path):
    df = pd.read_csv(file_path) 
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')  # Convertir fechas
    df = df.sort_values(by=['user', 'date', 'pc'])
    return df

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
        'anomaly': 0
    }

# === Función para agregar un log a la sesión ===
def add_log(session, row):
    log_entry = {
        'id': row['id'],
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
        'id': row['id'],
        'date': str(row['date']),
        'activity': row['activity']
    }
    session['devices'].append(device_entry)
    
# === Función para agregar un file a la sesión ===
def add_file(session, row):
    clean = limpiar_content(row['content'])
    file_entry = {
        'id': row['id'],
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
        'id': row['id'],
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
        'id': row['id'],
        'date': str(row['date']),
        'url': row['url'],
        'content': row['content']
    }
    session['http'].append(http_entry)

def save_sessions(sessions, output_path):
    sessions_df = pd.DataFrame(sessions, columns=['id_sesion', 'user', 'pc', 'logon', 'logoff', 'logs', 'devices', 'files', 'emails', 'http', 'anomaly'])
    sessions_df.to_csv(output_path, index=False)
    print(f"✅ Sesiones guardadas en: {output_path}")

def manage_sessions_and_extract(file_path, output_sessions_csv, max_sessions):

    df = load_logs(file_path)
    sessions = []
    active_sessions = {}
    session_count = 0

    for _, row in df.iterrows():
        if session_count >= max_sessions:
            break

        user = row['user']
        pc = row['pc']

        if user not in active_sessions:
            active_sessions[user] = {}

        if row['activity'] == 'Logon':
            if pc not in active_sessions[user]:
                active_sessions[user][pc] = start_session(row)
            add_log(active_sessions[user][pc], row)
            update_anomaly(active_sessions[user][pc], row)

        elif row['activity'] == 'Logoff':
            if pc in active_sessions[user]:
                add_log(active_sessions[user][pc], row)
                active_sessions[user][pc]['logoff'] = row['date']
                update_anomaly(active_sessions[user][pc], row)
                sessions.append(active_sessions[user][pc])
                if len(sessions) % 1000 == 0:
                    print(f"💾 {len(sessions)} sesiones guardadas...")
                session_count += 1
                del active_sessions[user][pc]

        elif row['activity'] in ['Connect', 'Disconnect']:
            if pc in active_sessions[user]:
                add_device(active_sessions[user][pc], row)

        elif row['log_type'] == 'file':
            if pc in active_sessions[user]:
                add_file(active_sessions[user][pc], row)

        elif row['log_type'] == 'email':
            if pc in active_sessions[user]:
                add_email(active_sessions[user][pc], row)

        elif row['log_type'] == 'http':
            if pc in active_sessions[user]:
                add_http(active_sessions[user][pc], row)

    # Guardar solo las columnas necesarias (sin 'id_sesion')
    columnas_sin_id = ['user', 'pc', 'logon', 'logoff', 'logs', 'devices', 'files', 'emails', 'http', 'anomaly']
    pd.DataFrame(sessions)[columnas_sin_id].to_csv(output_sessions_csv, index=False)
    print(f"✅ Sesiones guardadas sin 'id_sesion' en: {output_sessions_csv}")

    print(f"✅ Guardados {len(sessions)} sesiones en {output_sessions_csv}")


    
def extraer_ids_de_sesiones(path_sesiones, path_salida_ids):
    df = pd.read_csv(path_sesiones)
    ids_unicos = set()

    columnas_con_ids = ['logs', 'devices', 'files', 'emails', 'http']

    for columna in columnas_con_ids:
        if columna not in df.columns:
            continue
        for fila in df[columna].dropna():
            try:
                registros = ast.literal_eval(fila)
                if isinstance(registros, list):
                    for r in registros:
                        if isinstance(r, dict) and 'id' in r:
                            ids_unicos.add(r['id'])
            except Exception as e:
                print(f"Error en columna {columna}: {e}")
                continue

    df_ids = pd.DataFrame(sorted(ids_unicos), columns=['id'])
    df_ids.to_csv(path_salida_ids, index=False)
    print(f"✅ Guardados {len(df_ids)} IDs únicos en: {path_salida_ids}")
    

def main():
    manage_sessions_and_extract(
        file_path="dataset_bueno/dataset_complete_after_validation.csv",
        output_sessions_csv="test/sesiones_test.csv",
        max_sessions=57637
    )
    
    extraer_ids_de_sesiones(
        path_sesiones='test/sesiones_test.csv',
        path_salida_ids='test/test.csv'
    )

if __name__ == "__main__":
    main()