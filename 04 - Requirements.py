import os
import subprocess
import sys

def install_requirements():
    try:
        print("📦 Verificando e instalando dependencias...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements/requirements.txt"])
    except Exception as e:
        print(f"❌ Error instalando dependencias: {e}")
        sys.exit(1)

install_requirements()
