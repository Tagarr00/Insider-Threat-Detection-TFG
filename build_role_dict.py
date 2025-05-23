# build_ldap_dict.py
import pandas as pd
import json

def build_ldap_dict(csv_path, output_path="ldap_dict.json"):
    df = pd.read_csv(csv_path)
    ldap_dict = {}

    # Crear diccionario para cada campo si existe en el CSV
    for field in ['role', 'department', 'team', 'functional_unit', 'supervisor']:
        if field in df.columns:
            unique_values = df[field].dropna().unique()
            value_dict = {value: idx + 1 for idx, value in enumerate(sorted(unique_values))}
            ldap_dict[field] = value_dict
        else:
            print(f"⚠️ Campo '{field}' no encontrado en el CSV.")

    with open(output_path, 'w') as f:
        json.dump(ldap_dict, f, indent=4)

    print(f"✅ Diccionario LDAP guardado en: {output_path}")

if __name__ == "__main__":
    build_ldap_dict("dataset/LDAP/2009-12.csv")
