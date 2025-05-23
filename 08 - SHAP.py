import shap
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import os

# === Elegir modelo: 'lof' o 'iforestHNP' ===
model_folder = "lof"  # iforestHNP
model_name = "lof_model.joblib" if model_folder == "lof" else "iforest_model.joblib"

# === Rutas ===
base_path = os.path.join("model_def", model_folder)
model_path = os.path.join(base_path, model_name)
X_path = os.path.join(base_path, "X_all.npy")
feature_names_path = os.path.join("model_def", "feature_names.txt")
output_folder = os.path.join("XAI_SHAP", model_folder)
os.makedirs(output_folder, exist_ok=True)

# === Cargar modelo y datos ===
model = load(model_path)
pesos_totales = np.array([
    1.25, 8.0, 2.0, 8.0, 8.0,
    8.0, 8.0, 8.0, 8.0, 2.0,
    3.0, 1.25, 0.5, 0.0, 1.25
])

X = np.load(X_path)
X_scaled = X * pesos_totales

with open(feature_names_path) as f:
    feature_names = [line.strip() for line in f]

print(f"🔍 X shape: {X.shape}")
print(f"🔍 Num feature names: {len(feature_names)}")

# === Función de predicción invertida para que SHAP entienda mejor las anomalías
def predict_fn(X_input):
    scores = model.decision_function(X_input)
    return scores.reshape(-1, 1)

# === Inicializar SHAP KernelExplainer con muestra aleatoria
print(f"💡 Inicializando SHAP explainer para {model_folder}...")
background = shap.sample(X_scaled, 100, random_state=42)
explainer = shap.KernelExplainer(predict_fn, background)
X_sample = X_scaled[:1000]
shap_values = explainer.shap_values(X_sample)



# === Guardar gráfico con todas las features visibles
plt.figure()
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=15)
plt.tight_layout()
output_path = os.path.join(output_folder, "shap_summary_plot.png")
plt.savefig(output_path, dpi=300)
print(f"📸 SHAP summary plot guardado en '{output_path}'")
