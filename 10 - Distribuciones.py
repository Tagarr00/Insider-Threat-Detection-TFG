import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os

# Cargar CSV
df = pd.read_csv("train_vectors.csv") 

# Columnas elegidas para análisis
columnas = [
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

# Crear carpeta para guardar los gráficos
output_folder = "distribuciones_metricas"
os.makedirs(output_folder, exist_ok=True)

# Crear carpeta para guardar los gráficos
output_folder = "distribuciones_metricas"
os.makedirs(output_folder, exist_ok=True)

# === Análisis por columna ===
for col in columnas:
    data = df[col].dropna()
    if data.var() == 0:
        print(f"⚠️ {col} tiene varianza 0. Saltando.")
        continue

    mean = data.mean()
    std = data.std()
    min_val = data.min()
    max_val = data.max()
    
    print(f"\n📈 Métrica: {col}")
    print(f"  Media: {mean:.2f}, Desviación estándar: {std:.2f}")
    print(f"  Mínimo: {min_val:.2f}, Máximo: {max_val:.2f}")
    
    bin_width = (data.max() - data.min()) / 30
    x = np.linspace(data.min(), data.max(), 100)
    p = norm.pdf(x, mean, std) * len(data) * bin_width

    # Gráfico 1: Histograma con curva normal
    plt.figure()
    plt.hist(data, bins=30, density=False, alpha=0.6, label='Datos')
    plt.plot(x, p, 'k', linewidth=2, label='Distribución normal')
    plt.title(f'Distribución de {col}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f"{col}_normal.png"))
    plt.close()

    # Gráfico 2: Histograma con escala logarítmica
    plt.figure()
    plt.hist(data, bins=30, density=False, alpha=0.6, log=True, label='Datos')
    plt.title(f'Distribución logarítmica de {col}')
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f"{col}_log.png"))
    plt.close()

    # Gráfico 3: Boxplot con ajuste dinámico del eje x
    plt.figure()
    plt.boxplot(data, vert=False, flierprops=dict(marker='o', color='red', markersize=4))

    rango = data.max() - data.min()
    if rango > 10 * data.std():
        plt.xlim(data.quantile(0.01), data.quantile(0.99))
    else:
        plt.xlim(data.min() - 1, data.max() + 1)

    plt.title(f'Boxplot de {col}')
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f"{col}_boxplot.png"))
    plt.close()
