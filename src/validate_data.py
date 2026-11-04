
from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parent.parent / "data"

datasets = {}
ids = {}

for name in ["train", "validation", "test"]:
    df = pd.read_csv(base / f"{name}.csv")
    patient_ids = pd.read_csv(
        base / f"{name}_ids.csv"
    )["patient_id"]

    assert list(df.columns) == [
        "edad", "sexo", "imc", "diabetes"
    ], f"Columnas inesperadas: {name}"

    assert not df.isna().any().any(), (
        f"Hay nulos en {name}"
    )

    assert set(df["diabetes"].unique()) == {0, 1}, (
        f"Falta una clase en {name}"
    )

    assert len(df) == len(patient_ids), (
        f"IDs desalineados en {name}"
    )

    datasets[name] = df
    ids[name] = set(patient_ids)

    print(
        name,
        "registros:", len(df),
        "diabetes:", round(df["diabetes"].mean() * 100, 2),
        "%"
    )

assert ids["train"].isdisjoint(ids["validation"])
assert ids["train"].isdisjoint(ids["test"])
assert ids["validation"].isdisjoint(ids["test"])

print("Validación correcta: sin nulos ni pacientes repetidos.")
