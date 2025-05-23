import pandas as pd

def dividir_dataset_por_ids(
    path_dataset_completo='dataset_bueno/dataset_complete_after_validation.csv',
    path_ids_train='test/test.csv',
    path_salida_train_completo='test/test_complete.csv',
    path_salida_restante='dataset_bueno/errores_dataset.csv'
):
    print("📥 Cargando dataset completo...")
    df_completo = pd.read_csv(path_dataset_completo)
    print(f"🔢 Registros en dataset completo: {len(df_completo)}")

    print("📥 Cargando IDs del train...")
    df_ids = pd.read_csv(path_ids_train)
    ids_train = set(df_ids['id'])  # nos aseguramos de usar un set para eficiencia

    print("🔍 Separando registros que están en train...")
    df_train = df_completo[df_completo['id'].isin(ids_train)]
    df_restante = df_completo[~df_completo['id'].isin(ids_train)]

    print(f"✅ Registros en train_complete: {len(df_train)}")
    print(f"✅ Registros restantes para validation/test: {len(df_restante)}")

    print("💾 Guardando archivos...")
    df_train.to_csv(path_salida_train_completo, index=False)
    df_restante.to_csv(path_salida_restante, index=False)

    print("🎉 División completada.")

if __name__ == "__main__":
    dividir_dataset_por_ids()
