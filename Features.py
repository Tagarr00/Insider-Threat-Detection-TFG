import pandas as pd
import numpy as np
import nltk
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

nltk.download('vader_lexicon')


###################################################################LDAP#####################################################################################

# Ruta al archivo LDAP
LDAP_FILE = 'dataset/LDAP/2009-12.csv'

# === Cargar el archivo LDAP una sola vez al inicio ===
try:
    ldap_df = pd.read_csv(LDAP_FILE)
    ldap_df.fillna('', inplace=True)  # Rellenar valores NaN con cadenas vacías
except Exception as e:
    print(f"Error cargando el archivo LDAP: {e}")
    ldap_df = pd.DataFrame()


# === Función para cargar la información del usuario desde LDAP ===
def calculate_ldap_info(session):
    """
    Asigna la información del usuario desde el archivo LDAP a la sesión.
    """
    user = session['user']
    user_info = ldap_df[ldap_df['user_id'] == user]

    if not user_info.empty:
        session['name'] = user_info.iloc[0].get('employee_name', '')
        session['email'] = user_info.iloc[0].get('email', '')
        session['role'] = user_info.iloc[0].get('role', '')
        session['functional_unit'] = user_info.iloc[0].get('functional_unit', '')
        session['department'] = user_info.iloc[0].get('department', '')
        session['team'] = user_info.iloc[0].get('team', '')
        session['supervisor'] = user_info.iloc[0].get('supervisor', '')
    else:
        # Campos vacíos si el usuario no se encuentra
        session['name'] = ''
        session['email'] = ''
        session['role'] = ''
        session['functional_unit'] = ''
        session['department'] = ''
        session['team'] = ''
        session['supervisor'] = ''
    return session
#############################################################################################################################################################


###################################################################ocean#####################################################################################
# Ruta al archivo de OCEAN
OCEAN_FILE = 'dataset/psychometric.csv'

# === Cargar el archivo OCEAN una sola vez al inicio ===
try:
    ocean_df = pd.read_csv(OCEAN_FILE)
    ocean_df.fillna('', inplace=True)  # Rellenar NaNs con vacío
except Exception as e:
    print(f"Error cargando el archivo OCEAN: {e}")
    ocean_df = pd.DataFrame()

# === Función para cargar las métricas OCEAN por usuario ===
def calculate_ocean_scores(session):
    user = session['user']
    user_info = ocean_df[ocean_df['user_id'] == user]

    if not user_info.empty:
        session['O'] = user_info.iloc[0].get('O', '')
        session['C'] = user_info.iloc[0].get('C', '')
        session['E'] = user_info.iloc[0].get('E', '')
        session['A'] = user_info.iloc[0].get('A', '')
        session['N'] = user_info.iloc[0].get('N', '')
    else:
        session['O'] = ''
        session['C'] = ''
        session['E'] = ''
        session['A'] = ''
        session['N'] = ''

    return session


#############################################################################################################################################################


############################################################### EMAIL NLP##############################################################################################


# === Inicialización de modelos ===
#bert_model = SentenceTransformer('all-MiniLM-L6-v2')
#pca_model = PCA(n_components=2)
vader_analyzer = SentimentIntensityAnalyzer()
# TINYBEERT
emotion_classifier = pipeline('text-classification', model='AdamCodd/tinybert-emotion-balanced')
#Dislbert
#emotion_classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)

# === Palabras clave ===
IMPORTANCE_WORDS_MAIL = set([ ### meh para futuro?
    "confidential", "top-secret", "classified", "restricted", "sensitive", 
    "private", "internal-use-only", "non-disclosure", "nda", "exclusive", 
    "privileged",  "secret"
])

RISK_WORDS = set([ #igual
    "hacked", "backdoor", "keylogger", "trojan", "stealth", "undetectable",
    "spyware", "keystroke", "phishing", "malware", "ransomware", "exploit", "hacking",
    "cracked", "bypass", "obfuscation", "rootkit", "botnet", "payload", "virus", "worm",
    "ddos", "cyberattack", "leak", "breach", "intrusion", "compromised", "vulnerability"
])

#####
'''
JOBRISK_WORDS = set([
    "resume", "job", "growth", "opening", "experience",
    "skills", "required"
])
'''

JOBRISK_WORDS = set([
    "resume", "job", "opening", "recruiter", "part-time", "apply"
])



RESIGNATION_WORDS = set([
    "resign", "resignation", "quit", "leaving", "termination", "retire", "retirement"
])

EMOTIONAL_ANGRY_WORDS = set([
    "angry", "outraged", "exacerbated", "suffer", "irreplaceable",
    "fed up","no gratitude",
    "i will leave", "take me seriously", "company will suffer"
])

PANIC_WORDS = set([
    "terminated", "rescind","terminated funding","project cancelled","downsizing project","funding terminated",
    "budget rescind","reallocate project","notice termination","budget reallocate"

])

threat_words = set([
    "no one is irreplaceable",
    "this access shouldn't be restricted",
    "i’ll take a backup just in case",
    "they don’t pay me enough for this",
    "i could automate all of this and leave",
    "i have root, they don't even know",
    "i know where everything is stored",
    "i'm logging everything now",
    "let’s see how they handle a real incident",
    "this system is wide open",
    "they think they're safe",
    "i copied the logs just in case",
    "why should i even warn them?",
    "i could crash this in one line"
])



def process_email_activity(session):
    emails = session['emails']
    if not emails:
        return session

    anger_total = fear_total = sadness_total = 0.0
    negative_emotion_score = 0.0
    negative_emotion_count = 0
    vader_scores = []
    bert_vectors = []
    contents = []
    max_size=0

    def count_words(text, word_set):
        text = text.lower()
        return sum(text.count(word) for word in word_set)


    for email in emails:
        content = email.get('content', '').strip()
        content_lower = content.lower()

        # === Guardamos texto para embeddings y stats
        contents.append(content)

        # === VADER
        vader = vader_analyzer.polarity_scores(content)['compound']
        vader_scores.append(vader)
        if vader < -0.9:
            session['very_negative_vader_count'] += 1

        # === TinyBERT emociones
        try:
            emotion = emotion_classifier([content])[0]
            label = emotion['label'].lower()
            score = emotion['score']
        except Exception as e:
            label = ''
            score = 0.0

        if label in ['anger', 'fear', 'sadness']:
            negative_emotion_count += 1
            negative_emotion_score += score
        if label == 'anger':
            anger_total += score
        elif label == 'fear':
            fear_total += score
        elif label == 'sadness':
            sadness_total += score

        # === Palabras clave
        jobrisk_count = count_words(content_lower, JOBRISK_WORDS) #PQ LA USAMOS 2 VECES(no doble conteo) -- eficiiencia
        
        session['risk_words_count'] += count_words(content_lower, RISK_WORDS)
        session['job_risk_words_count'] += jobrisk_count
        session['importance_word_count'] += count_words(content_lower, IMPORTANCE_WORDS_MAIL)
        session['resignation_word_count'] += count_words(content_lower, RESIGNATION_WORDS)

        # === Hora
        hour = pd.to_datetime(email['date']).hour
        if hour < 7 or hour >= 19:
            session['email_unusual_time_count'] += 1

        # === BCC externo
        session['bcc_total_count'] += len(email['bcc'])
        session['bcc_external_count'] += sum(1 for addr in email['bcc'] if not addr.endswith('@dtaa.com'))

        # === Destinatarios completamente externos
        if not any(addr.endswith('@dtaa.com') for addr in email['to'] + email['cc'] + email['bcc']):
            session['email_external_to_count'] += 1

        # === Remitente
        if email['from'] == session.get('email', ''):
            session['email_count_sent_user'] += 1
        else:
            session['email_count_sent_external_account'] += 1
            if email['from'].endswith('@dtaa.com'):
                session['email_corporate_not_user'] += 1

        # === Adjuntos
        session['email_attachment_count'] += email['attachments']
        session['email_attachment_size_total'] += email['size']
        if email['size'] > max_size:
            max_size = email['size']

        ######################## AÑADIR METRICAS DE ANOMALIA EN UN MAIL################################
        # === Nueva métrica: Job risk + destinatario externo
        if jobrisk_count >= 5:
            if any(not addr.endswith('@dtaa.com') for addr in email['to']):
                if session['role'] != "HumanResourceSpecialist":
                    session['job_external'] += 1

        # === Nueva métrica: correo con tono emocionalmente alterado (enfadado)
        angry_word_count = count_words(content_lower, EMOTIONAL_ANGRY_WORDS)

        if  angry_word_count > 10: #(score > 0.75 and label == 'anger') or
            session['angry_email_count'] += 1
            
        if score > 0.75 and label == 'sadness':
            session['sad_email_count'] += 1

        # === Nueva métrica: correos con puntuación alta de miedo
        if score > 0.75 and label == 'fear':
            session['fear_email_count'] += 1
            
         # === Nueva métrica: correo de pánico masivo
        panic_word_count = count_words(content_lower, PANIC_WORDS)
        if panic_word_count > 6:
            session['panic_email_count'] += 1
            
        if count_words(content_lower, RESIGNATION_WORDS) > 6:
            session["resignation_email"] += 1
            
        if count_words(content_lower, threat_words) > 3:
            session["threat_mail"] += 1
        
            
            ##quizxas ratio dd e mails con correo de user y correo externo usado
            #otra metrica de si un mail excede el giga pues sumar uno


        ###########################################################################################33#

    session['emails_total'] = len(emails)

    # === Embeddings y PCA al final
    if contents:
        #bert_vectors = bert_model.encode(contents, show_progress_bar=False)
        #reduced = pca_model.fit_transform(bert_vectors) if len(bert_vectors) > 1 else [[0.0, 0.0]]
        #session['bert_pca1'] = float(np.mean([vec[0] for vec in reduced]))
        #session['bert_pca2'] = float(np.mean([vec[1] for vec in reduced]))
        if session['emails_total'] > 0:
            session['external_email_ratio'] = session['email_external_to_count'] / session['emails_total']
        else:
            session['external_email_ratio'] = 0.0



        if session['emails_total'] > 0:
            session['negative_email_ratio'] = (
                session['angry_email_count'] +
                session['sad_email_count'] +
                session['fear_email_count']
            ) / session['emails_total']
        else:
            session['negative_email_ratio'] = 0.0


        # Medias finales
        session['vader_sentiment'] = float(np.mean(vader_scores))
        n = len(contents)
        #session['negative_emotion_score'] = negative_emotion_score / n

        session['max_email_size'] = max_size
        if session['emails_total'] > 0:
            session['email_unusual_time_ratio'] = session['email_unusual_time_count'] / session['emails_total']
        else:
            session['email_unusual_time_ratio'] = 0.0

        session['panic_email_count'] = int(session['panic_email_count'] > 0)
        session['resignation_email'] = int(session['resignation_email'] > 0)
        #session['job_external'] = int(session['job_external'] > 0)
        session['threat_mail'] = int(session['threat_mail'] > 0)
        session['email_corporate_not_user'] = int(session['email_corporate_not_user'] > 0)
        
         # === Flag: ratio de correos muy enfadados
        if session['emails_total'] > 0:
            session['angry_email_ratio'] = session['angry_email_count'] / session['emails_total']
        else:
            session['angry_email_ratio'] = 0.0

        session['angry_email_flag'] = int(session['angry_email_ratio'] > 0.7)
        session['vader_sentiment_flag'] = int(session['vader_sentiment'] <-0.9)



    return session



#############################################################################################################################################################

###########################################################  FILE NLP ##################################################################################################
# === Palabras clave específicas para archivos ===
importance_words_FILE = set([
    "confidential", "top", "secret", "classified", "restricted", "sensitive", 
    "privileged", "proprietary", "internal-use-only", "protected", 
    "exclusive", "secure", "private", "non-disclosure", "nda"
])

risk_words_FILE = set([
    "malware", "keylogging", "password", "undetectable", "surveillance", "hidden", "captured",  "illegal", "monitor"
])

job_risk_words_FILE = set([
    "resume", "cv", "curriculum", "cover letter", "benefits", 
    "contract", "recruiter", "job offer", 
    "interview", "hiring", "vacancy", 
    "application", 
    "resignation", "quit", "leave", "unemployment", "new role"
])
def process_file_activity(session):
    files = session['files']
    if not files:
        return session

    def count_words(text, word_set):
        text = text.lower()
        return sum(text.count(word) for word in word_set)

    filenames_seen = set()

    for file in files:
        filename = file.get('filename', '').lower()
        content = file.get('content', '').lower()

        # Total y únicos
        filenames_seen.add(filename)

        # .exe y .zip
        if filename.endswith('.exe'):
            session['files_exe'] += 1

        # Hora
        if 'date' in file:
            hour = pd.to_datetime(file['date']).hour
            if hour < 7 or hour >= 19:
                session['files_out_hours'] += 1
                
        risk_score = count_words(content, risk_words_FILE)

        # Análisis de contenido
        if content:
            session['importance_file'] += count_words(content, importance_words_FILE)
            session['risk_file'] += risk_score
            session['job_risk_file'] += count_words(content, job_risk_words_FILE)


         # === Metrica de archivo potencialmente malicioso 
        if risk_score >= 8 and filename.endswith('.exe'):
            session['file_virus_threat'] += 1

    

    session['files_total'] = len(files)
    session['files_unique'] = len(filenames_seen)
    session['file_virus_threat'] = int(session['file_virus_threat'] > 0)
    
    # === Ratio de archivos .exe
    if session['files_total'] > 0:
        exe_ratio = session['files_exe'] / session['files_total']
    else:
        exe_ratio = 0.0

    session['files_exe_ratio'] = exe_ratio
    session['high_exe_ratio_flag'] = int(exe_ratio > 0.7)
    

    return session



#############################################################################################################################################################

################################################################# HTTP NLP############################################################################################

leak_url_keywords = set([
    "spy",
    "sabatoge",
    "forgery",
    "blackmail",
    "clandestine",
    "covert",
    "confidential",
    "top-secret",
    "restricted",
    "subterfuge",
    "evade",
    "overthrow",
    "conspiracy"
])

malware_keywords = set([
    "malware", "keylogging",  "undetectable", "surveillance",  "password", "illegal", "monitor"
])

malware_domains =(['keylogger'])

suspicious_domains =  set(['wikileaks.org'])

jobs_domains = set(['linkedin', 'job-hunt', 'jobhuntersbible', 'careerbuilder', 'simplyhired', 'indeed', 'hotjobs' , 'monster', 'craiglist'])


job_keywords = set([
    "job", "resume", "degree", "experience", "skills", "required", "growth", 
    ])


def process_http_activity(session):
    http_access = session['http']

    if not http_access:
        return session

    def count_words(text, word_set):
        text = text.lower()
        return sum(text.count(word) for word in word_set)
    
    def has_any_keyword(text, word_set):
        text = text.lower()
        return any(word in text for word in word_set)

    for http in http_access:
        date = pd.to_datetime(http['date'])
        if date.hour < 7 or date.hour >= 19:
            session['http_out_hours'] += 1

        url = http.get('url', '').lower()
        content = http.get('content', '').lower()
        
        if has_any_keyword(url, suspicious_domains):
            session['leak_url_count'] += 1

        if has_any_keyword(url, malware_domains):
            session['malware_url_count'] += 1

        if has_any_keyword(url, jobs_domains):
            session['job_url_count'] += 1

        session['leak_httpcontent_count'] += count_words(content, leak_url_keywords)
        session['malware_httpcontent_count'] += count_words(content, malware_keywords)
        session['job_httpcontent_count'] += count_words(content, job_keywords)

        # Búsqueda de empleo: dominio + contenido
        found_job_domain = has_any_keyword(url, jobs_domains)            
        content_job_matches = count_words(content, job_keywords)
        if found_job_domain and content_job_matches >= 3:
            if session['role'] != "HumanResourceSpecialist":
                session["job_search"] += 1
         
        content_malware = count_words(content, malware_keywords)
        if content_malware >= 7 or ((count_words(url, malware_domains)) > 0) :
            session["http_malware_page"] += 1

    session['http_malware_page_flag'] = int(session['http_malware_page'] > 0)
    session['http_leak_flag'] = int(session['leak_url_count'] > 0)
    #session['job_search_flag'] = int(session['job_search'] > 0)
    session['job_search_flag'] = int(session['job_search'] >= 3)

    session['http_total'] = len(http_access)

    return session


#############################################################################################################################################################

################################################################### METRICAS MALICIOSAS ##########################################################################################
def possible_threat(session):

    # RATIO ANOMALO DE DEVICES
    threat_score = 0
    
    if session.get('device_event_ratio', 0) > 0.8:
        threat_score += 30  
    elif session.get('device_event_ratio', 0) > 0.5:
        threat_score += 20
        
    device_oo_ratio = session.get('device_oo_ratio', 0)
    if device_oo_ratio > 0.8:
        threat_score += 40
    elif device_oo_ratio > 0.5:
        threat_score += 30
    elif device_oo_ratio > 0.3:
        threat_score += 20
        
    device_in = session.get('device_in', 0)
    if device_in > 8:
        threat_score += 40
    elif device_in >= 5:
        threat_score += 30
    
    devices_out_hours = session.get('devices_activity_out_hours', 0)
    if devices_out_hours > 5:
        threat_score += 25
    elif devices_out_hours > 2:
        threat_score += 15
    session['device_ratio_anomaly'] = threat_score

    
    # Posible LEAK 
    threat_score2 = 0

    leak_urls = session.get('leak_url_count', 0)
    leak_content = session.get('leak_httpcontent_count', 0)
    http_oo_ratio = session.get('http_oo_ratio', 0)
    device_oo_ratio = session.get('device_oo_ratio', 0)
    events_oo_ratio = session.get('events_oo_ratio', 0)

    # Detecta si hay alguna actividad relacionada con fugas

    if leak_urls > 0:
        threat_score2 += 40
        if http_oo_ratio > 0.3:
            threat_score2 += 10
        if device_oo_ratio > 0.3:
            threat_score2 += 20
        if events_oo_ratio > 0.6:
            threat_score2 += 10

    session['leak_threat'] = threat_score2

     # BUSQUEDA DE EMPLEO
    job_search_threat = 0

    job_emails = session.get('job_external', 0)
    job_search = session.get('job_search', 0)
    
        # URLs relacionadas con job hunting
    if job_search >= 6:
        job_search_threat += 50
    if job_search >= 4:
        job_search_threat += 25
    elif job_search >= 2:
        job_search_threat += 15

    # Emails con contenido de job y enviados a externos
    if job_emails >= 2:
        job_search_threat += 35
    elif job_emails >= 1:
        job_search_threat += 25

    session['job_search_threat'] = job_search_threat
    if job_search >= 2:
        session['job_flag_search'] = 1
    if job_emails > 0:
        session['job_flag_email'] = 1

    # Malicia dejando empleo
    resignation_danger = 0
    resignation_email = session.get('resignation_email', 0)
    device_in = session.get('device_in', 0)

    if resignation_email > 0:
        resignation_danger += 15
        if device_in >= 4:
            resignation_danger += 30
        if session.get('device_event_ratio', 0) > 0.3:
            resignation_danger += 30

    session['resignation_danger'] = resignation_danger

    # ADMIN ENFADADO CON POSIBLE AMENAZA
    virus_threat = 0

    angry_emails = session.get('angry_email_count', 0)
    virus_files = session.get('file_virus_threat', 0)
    malware_page = session.get('http_malware_page', 0)
    
            
    if virus_files > 0 and malware_page > 1:
            virus_threat += 50
            if session['role'] == "ITAdmin":
                virus_threat += 50
            if angry_emails > 0:    
                virus_threat += 20
            
    session['potential_virus_threat'] = virus_threat

    return session


############################################################ LOGON Y LOG OFF#################################################################################################

def calculate_working_hours_ratios(session):
    start = session['logon']
    end = session['logoff']

    total_minutes = (end - start).total_seconds() / 60
    if total_minutes <= 0:
        session['working_hours_ratio'] = 0.0
        session['working_out_hours_ratio'] = 0.0
        session['is_working_hours'] = 0
        return session

    working_minutes = 0
    current_day = start.normalize()

    while current_day <= end.normalize():
        day_start = current_day + pd.Timedelta(hours=7)
        day_end = current_day + pd.Timedelta(hours=19)

        overlap_start = max(start, day_start)
        overlap_end = min(end, day_end)

        if overlap_start < overlap_end:
            working_minutes += (overlap_end - overlap_start).total_seconds() / 60

        current_day += pd.Timedelta(days=1)

    ratio = working_minutes / total_minutes
    session['working_hours_ratio'] = round(ratio, 3)
    session['working_out_hours_ratio'] = round(1 - ratio, 3)
    session['is_working_hours'] = 1 if ratio >= 0.7 else 0

    return session




################################################   devices   ######################################################################################################

def process_device_activity(session):
    devices = session['devices']

    for event in devices:
        if event['activity'] == 'Connect':
            session['device_in'] += 1
            hour = pd.to_datetime(event['date']).hour
            if hour < 7 or hour >= 19:
                session['devices_activity_out_hours'] += 1
        elif event['activity'] == 'Disconnect':
            session['device_out'] += 1

    session['device_out_hours_flag'] = int(session['devices_activity_out_hours'] > 2)
    session['device_lots_flag'] = int(session['device_in'] >= 6)



    return session


######################################################################   EVENTOS   #######################################################################################
    
# === Función para calcular eventos y su conteo ===
def calculate_activity_metrics(session):
    events = []
    for _ in range(session.get('device_in', 0)):
        events.append('device_in')
    for _ in range(session.get('emails_total', 0)): 
        events.append('email')
    for _ in range(session.get('files_total', 0)):
        events.append('file')
    for _ in range(session.get('http_total', 0)):
        events.append('http')

    session['events'] = events
    total_events = len(events)
    if total_events == 0:
        total_events = 1
    session['event_count'] = total_events

    # Ratios por tipo
    session['device_event_ratio'] = session.get('device_in', 0) / total_events
    session['email_event_ratio'] = session.get('emails_total', 0) / total_events
    session['file_event_ratio'] = session.get('files_total', 0) / total_events
    session['http_event_ratio'] = session.get('http_total', 0) / total_events

    # Eventos fuera de horario
    session['events_out_hours'] = (
        session.get('devices_activity_out_hours', 0) +
        session.get('email_unusual_time_count', 0) +
        session.get('files_out_hours', 0) +
        session.get('http_out_hours', 0)
    )
    
    session['events_in_hours'] = session['event_count'] - session['events_out_hours']

    session['device_oo_ratio'] = session.get('devices_activity_out_hours', 0) / total_events
    session['email_oo_ratio'] = session.get('email_unusual_time_count', 0) / total_events
    session['file_oo_ratio'] = session.get('files_out_hours', 0) / total_events
    session['http_oo_ratio'] = session.get('http_out_hours', 0) / total_events
    session['events_oo_ratio'] = session['events_out_hours'] / total_events
    
    # === NUEVO MANEJO DE RATIOS CUANDO NO HAY EVENTOS ===
    if total_events == 0:
        logon_hour = session['logon'].hour
        if 7 <= logon_hour < 19:
            session['events_in_hours_ratio'] = 1
            session['events_out_hours_ratio'] = 0
        else:
            session['events_in_hours_ratio'] = 0
            session['events_out_hours_ratio'] = 1
    else:
        session['events_in_hours_ratio'] = session['events_in_hours'] / session['event_count']
        session['events_out_hours_ratio'] = session['events_out_hours'] / session['event_count']
        
    session['device_ratio_flag'] = int(session['device_event_ratio'] > 0.95)
    session['job_search_and_email'] = int(session.get('job_search_flag', 0) == 1 and session.get('job_external', 0) == 1)
    # === Flag por alta proporción de eventos fuera de horario
    session['events_out_hours_flag'] = int(session['events_out_hours_ratio'] > 0.8)

    return session


# === Función principal de métricas ===
def feature(session):
    #Primero las metricas extra
    session = calculate_ldap_info(session)
    session = calculate_ocean_scores(session)
    #Trabaja en horario laboral o no y cuanto?
    session = calculate_working_hours_ratios(session)

    #metricas por cada tipo de actividad 
    session = process_device_activity(session)
    session = process_email_activity(session)
    session = process_file_activity(session)
    session = process_http_activity(session)

    #metrica de eventos generales en la sesion y si estan fuera de horario
    session = calculate_activity_metrics(session)
    
    # Aqui vamos a añadir metricas que puedan aporatarnos escenarios maliciosos combinando metricas anteriores
    #session = possible_threat(session)

    
    
    return session
