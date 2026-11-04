
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Rutas del proyecto
BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "synthea" / "output" / "csv"
OUT = BASE / "data"
OUT.mkdir(exist_ok=True)

# Fecha de referencia del conjunto de datos.
# Synthea debe haberse ejecutado con la misma fecha.
REFERENCE_DATE = pd.Timestamp("2026-11-02")

print("Leyendo archivos de Synthea...")

patients = pd.read_csv(
    RAW / "patients.csv",
    dtype={"Id": str}
)

conditions = pd.read_csv(
    RAW / "conditions.csv",
    dtype={"PATIENT": str, "CODE": str}
)

observations = pd.read_csv(
    RAW / "observations.csv",
    dtype={"PATIENT": str, "CODE": str},
    low_memory=False
)

# 1. Datos demográficos
df = patients[["Id", "BIRTHDATE", "GENDER"]].copy()
df = df.rename(columns={"Id": "patient_id"})

birthdates = pd.to_datetime(
    df["BIRTHDATE"], errors="coerce"
)

df["edad"] = (
    REFERENCE_DATE.year - birthdates.dt.year
    - (
        (REFERENCE_DATE.month < birthdates.dt.month)
        | (
            (REFERENCE_DATE.month == birthdates.dt.month)
            & (REFERENCE_DATE.day < birthdates.dt.day)
        )
    ).astype(int)
)

# Codificación simple del sexo registrado.
df["sexo"] = df["GENDER"].map({"F": 0, "M": 1})

# 2. Identificar diagnósticos de diabetes
# Solo tipos 1 y 2 reconocidos en la descripción.
description = conditions["DESCRIPTION"].fillna("")

has_diabetes = description.str.contains(
    r"\b(?:type\s*[12]|type\s*(?:i|ii))\s+diabetes"
    r"|diabetes\s+mellitus",
    case=False,
    regex=True
)

# Se excluye gestacional y prediabetes.
excluded = description.str.contains(
    r"gestational|prediabet|pre-diabet",
    case=False,
    regex=True
)

diabetes_ids = set(
    conditions.loc[
        has_diabetes & ~excluded, "PATIENT"
    ].dropna()
)

df["diabetes"] = (
    df["patient_id"].isin(diabetes_ids).astype(int)
)

# 3. Obtener IMC de las observaciones
# Código LOINC 39156-5: Body mass index
bmi = observations.loc[
    observations["CODE"].eq("39156-5"),
    ["PATIENT", "DATE", "VALUE"]
].copy()

bmi["DATE"] = pd.to_datetime(
    bmi["DATE"], errors="coerce", utc=True
)

bmi["imc"] = pd.to_numeric(
    bmi["VALUE"], errors="coerce"
)

# Evitar observaciones posteriores a la fecha
# de referencia del conjunto.
cutoff = REFERENCE_DATE.tz_localize("UTC")

bmi = bmi.loc[
    (bmi["DATE"] <= cutoff)
    & bmi["imc"].between(10, 80)
]

# Seleccionar el IMC más reciente de cada paciente
bmi = (
    bmi.sort_values("DATE")
       .drop_duplicates("PATIENT", keep="last")
)

df = df.merge(
    bmi[["PATIENT", "imc"]],
    left_on="patient_id",
    right_on="PATIENT",
    how="left"
)

# 4. Limpieza
df = df.dropna(subset=["edad", "sexo", "imc"])
df = df[df["edad"].between(18, 100)].copy()
df = df.drop_duplicates(subset=["patient_id"])

# Variables predictoras y etiqueta
features = ["edad", "sexo", "imc", "diabetes"]

print("Pacientes utilizables:", len(df))
print("Distribución de etiquetas:")
print(df["diabetes"].value_counts())

# Verificar que haya suficientes registros por clase
counts = df["diabetes"].value_counts()

if len(counts) != 2 or counts.min() < 20:
    raise ValueError(
        "No hay suficientes registros de ambas clases. "
        "Revisar diagnósticos o generar más pacientes."
    )

# 5. Separar 70% entrenamiento, 15% validación,
# 15% prueba, conservando la proporción de clases.
train, temp = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["diabetes"]
)

validation, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=temp["diabetes"]
)

# 6. Exportar sin identificadores personales.
train[features].to_csv(
    OUT / "train.csv", index=False
)
validation[features].to_csv(
    OUT / "validation.csv", index=False
)
test[features].to_csv(
    OUT / "test.csv", index=False
)

# Guardar los IDs aparte SOLO para auditoría
# experimental; no se pasan al modelo.
for name, subset in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:
    subset[["patient_id"]].to_csv(
        OUT / f"{name}_ids.csv",
        index=False
    )

print("\nArchivos generados:")
for name, subset in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:
    print(
        name,
        len(subset),
        "pacientes",
        "| diabetes:",
        subset["diabetes"].sum()
    )

print("\nProceso terminado.")
