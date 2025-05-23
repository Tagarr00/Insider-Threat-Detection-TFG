import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# === Directorios y rutas ===
model_folder = "iforestHNP"
model_path = os.path.join("model_def", model_folder)
output_folder = os.path.join("graficos", model_folder)
os.makedirs(output_folder, exist_ok=True)

# === Cargar datos ===
X = np.load(os.path.join(model_path, "X_all.npy"))
y = np.load(os.path.join(model_path, "y_all.npy"))
uniques, counts = np.unique(y, return_counts=True)
print(f"Valores en y: {dict(zip(uniques, counts))}")
df = pd.read_csv("train_vectors.csv")

# === Leer nombres de features ===
with open(os.path.join("model_def", "feature_names.txt")) as f:
    all_features = [line.strip() for line in f]

# === Diccionario de listas de features ===
feature_sets = {
    "total": [
        'devices_activity_out_hours', 'panic_email_count', 'job_external',
        'resignation_email', 'threat_mail', 'email_corporate_not_user',
        'http_leak_flag', 'http_malware_page_flag', 'file_virus_threat',
        'job_search', 'angry_email_count', 'events_out_hours', 'files_exe',
        'very_negative_vader_count', 'device_in'
    ],
    "scenario_1": [
        'devices_activity_out_hours', 'http_leak_flag', 'events_out_hours', 'device_in'
    ],
    "scenario_2": [
        'job_external', 'resignation_email', 'job_search', 'device_in'
    ],
    "scenario_3": [
        'panic_email_count', 'threat_mail', 'http_malware_page_flag',
        'file_virus_threat', 'angry_email_count', 'files_exe', 'device_in'
    ],
    "sentiment": [
        'panic_email_count', 'job_external', 'resignation_email', 'threat_mail',
        'angry_email_count', 'files_exe', 'very_negative_vader_count'
    ],
    "flags": [
        'panic_email_count', 'resignation_email', 'threat_mail',
        'email_corporate_not_user', 'http_leak_flag',
        'http_malware_page_flag', 'file_virus_threat'
    ],
    "continuous": [
        'devices_activity_out_hours', 'job_external', 'job_search',
        'angry_email_count', 'events_out_hours', 'files_exe',
        'very_negative_vader_count', 'device_in'
    ]
}

# === Gráficos PCA ===
for name, selected_features in feature_sets.items():
    indices = [all_features.index(f) for f in selected_features]
    X_selected = X[:, indices]

    print(f"📊 Generando gráficos para: {name} con {len(selected_features)} features")

    # PCA 2D
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_selected)
    plt.figure(figsize=(8, 6))
    # Primero anomalía (rojo) y luego normal (azul) encima
    plt.scatter(X_pca_2d[y == 1, 0], X_pca_2d[y == 1, 1], alpha=0.7, label="Anomalía", s=10, color='red')
    plt.scatter(X_pca_2d[y == 0, 0], X_pca_2d[y == 0, 1], alpha=0.5, label="Normal", s=10)
    plt.legend()
    plt.title(f"PCA 2D - {name}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{name}_pca_2d.png"))
    plt.close()

    # PCA 3D
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_selected)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca_3d[y == 1, 0], X_pca_3d[y == 1, 1], X_pca_3d[y == 1, 2], alpha=0.7, label="Anomalía", s=10, color='red')
    ax.scatter(X_pca_3d[y == 0, 0], X_pca_3d[y == 0, 1], X_pca_3d[y == 0, 2], alpha=0.5, label="Normal", s=10)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    ax.set_title(f"PCA 3D - {name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{name}_pca_3d.png"))
    plt.close()


