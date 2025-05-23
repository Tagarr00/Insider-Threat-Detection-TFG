# 🕵️‍♂️ Insider Threat Detection - TFG 🚨

Este proyecto es el resultado de un Trabajo de Fin de Grado (TFG) centrado en la **detección de amenazas internas** (insider threats) mediante **modelos de aprendizaje no supervisado** y **técnicas de NLP (procesamiento del lenguaje natural)**.

💻 Desarrollado en Python, este sistema analiza sesiones de usuario en un entorno corporativo simulado utilizando el dataset **CERT r4.2**, combinando métricas de comportamiento, actividad digital, análisis emocional y modelos como **Isolation Forest (IF)** y **Local Outlier Factor (LOF)**.

---

## 📦 Estructura general del proyecto

📁 dataset/
📁 src/
├── main.py
├── features.py
├── vectorizer.py
├── model_training.py
├── model_prediction.py
├── explain_lime.py
📁 results/
📄 requirements.txt
📄 README.md

yaml
Copiar
Editar

---

## ⚙️ Requisitos

- Python 3.10+
- Recomendado: entorno virtual (virtualenv o conda)

Instala las dependencias con:

```bash
pip install -r requirements.txt
🚀 Instrucciones básicas
1️⃣ Descargar el dataset CERT
Antes de ejecutar cualquier script, descarga el dataset CERT r4.2 desde el siguiente enlace:
🔗 Insider Threat Test Dataset - CERT

Descomprime el contenido y colócalo dentro de la carpeta dataset/.

2️⃣ Generar sesiones y dividir el dataset
Una vez tengas los datos originales:

Ejecuta el script de creación de sesiones (respetando la lógica descrita en el TFG).

Divide el conjunto en train, validation y test según el número de sesiones (70%/15%/15%).

Ejemplo de ejecución:

bash
Copiar
Editar
python src/main.py
3️⃣ Entrena el modelo
bash
Copiar
Editar
python src/model_training.py
4️⃣ Realiza predicciones y explicaciones
bash
Copiar
Editar
python src/model_prediction.py
python src/explain_lime.py
🧠 Tecnologías utilizadas
🐍 Python, Scikit-learn, PyOD

🤖 NLP: Transformers, VADER, BERT embeddings

📊 SHAP y LIME para interpretabilidad

📁 Dataset CERT r4.2

📄 Licencia
Este proyecto ha sido desarrollado con fines académicos como parte de un Trabajo de Fin de Grado (TFG).
Se permite su uso con fines educativos y de investigación.
👨‍🎓 Autor: [Tu Nombre]
