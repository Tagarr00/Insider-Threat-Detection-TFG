# vectorizer.py
import json

from datetime import datetime


def load_ldap_dict(path="ldap_dict.json"):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        print("⚠️ Error cargando ldap_dict.json, se usará diccionario vacío.")
        return {"role": {}, "department": {}, "team": {}, "functional_unit": {}, "supervisor": {}}



ldap_dict = load_ldap_dict()
    
def vectorize_session(session):
    """Genera un feature_vector numérico desde una sesión con métricas ya calculadas."""


    # Extraer valores de hora y minuto de logon y logoff
    logon_time = session['logon']
    logoff_time = session['logoff']
    logon_hour = logon_time.hour
    logon_minute = logon_time.minute
    logoff_hour = logoff_time.hour
    logoff_minute = logoff_time.minute

    # Extraer números de user y pc
    user_token = int(session['user'][-3:]) if session['user'][-4:].isdigit() else 0
    pc_token = int(session['pc'].split('-')[-1]) if session['pc'].startswith('PC-') else 0

    # NUEVOS CAMPOS CATEGÓRICOS TRANSFORMADOS

    
    role_token = ldap_dict["role"].get(session.get('role'), 0)
    func_unit_token = ldap_dict["functional_unit"].get(session.get('functional_unit'), 0)
    department_token = ldap_dict["department"].get(session.get('department'), 0)
    team_token = ldap_dict["team"].get(session.get('team'), 0)
    supervisor_token = ldap_dict["supervisor"].get(session.get('supervisor'), 0)


    # Generar vector final


    # === FLAGS BINARIOS (0 o 1) ===
    vector_hybrid = [
        session.get('devices_activity_out_hours', 0),     # continua
        session.get('panic_email_count', 0),              # flag
        session.get('job_external', 0),                   # continua
        session.get('resignation_email', 0),              # flag
        session.get('threat_mail', 0),                    # flag

        session.get('email_corporate_not_user', 0),       # flag
        session.get('http_leak_flag', 0),                 # flag
        session.get('http_malware_page_flag', 0),         # flag
        session.get('file_virus_threat', 0),              # flag
        session.get('job_search', 0),                     # continua

        session.get('angry_email_count', 0),              # continua
        session.get('events_out_hours', 0),               # continua
        session.get('files_exe', 0),                      # continua
        session.get('very_negative_vader_count', 0),      # continua
        session.get('device_in', 0),                      # continua
    ]

    #return continuous_features + binary_flags ##NTRENAMIENTO
    return vector_hybrid + [session.get('anomaly', 0)] #SOLO PARA VALDIACION Y PRUEBAS




