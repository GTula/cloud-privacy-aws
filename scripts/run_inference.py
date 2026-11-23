
import json
from pathlib import Path

import boto3
import pandas as pd

REGION = "us-east-1"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
N_DISPLAY = 10


def load_endpoints():
    path = RESULTS_DIR / "endpoints.json"
    if not path.exists():
        raise FileNotFoundError(
            "endpoints.json no encontrado — ejecutar deploy_models.py primero."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def invoke_endpoint(runtime_client, endpoint_name, instances):
    payload = json.dumps({"instances": instances})
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=payload.encode("utf-8"),
    )
    result = json.loads(response["Body"].read())
    return result["predictions"]


def main():
    endpoints = load_endpoints()
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    all_instances = test_df[["edad", "sexo", "imc"]].values.tolist()
    sample_instances = all_instances[:N_DISPLAY]
    sample_actuals = test_df["diabetes"].iloc[:N_DISPLAY].tolist()

    runtime = boto3.client("sagemaker-runtime", region_name=REGION)

    print("Consultando endpoints...")
    conv_all = invoke_endpoint(runtime, endpoints["conventional"], all_instances)
    priv_all = invoke_endpoint(runtime, endpoints["private"], all_instances)

    # Tabla comparativa con las primeras N muestras
    header = f"{'Edad':>5}  {'Sexo':>4}  {'IMC':>6}  {'Real':>5}  {'Conv':>8}  {'DP (ε≈5)':>10}"
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("PREDICCIONES — CONVENCIONAL vs. PRIVACIDAD DIFERENCIAL")
    print(sep)
    print(header)
    print("-" * len(header))

    for i, (edad, sexo, imc) in enumerate(sample_instances):
        print(
            f"{edad:>5.0f}  {sexo:>4.0f}  {imc:>6.1f}  "
            f"{sample_actuals[i]:>5}  "
            f"{conv_all[i]:>8.4f}  "
            f"{priv_all[i]:>10.4f}"
        )

    print(sep)

    # Guardar predicciones completas
    conv_out = test_df.copy()
    conv_out["prediction"] = conv_all
    conv_path = RESULTS_DIR / "predictions_conventional.csv"
    conv_out.to_csv(conv_path, index=False)

    priv_out = test_df.copy()
    priv_out["prediction"] = priv_all
    priv_path = RESULTS_DIR / "predictions_private.csv"
    priv_out.to_csv(priv_path, index=False)

    print(f"\nPredicciones guardadas:")
    print(f"  {conv_path}")
    print(f"  {priv_path}")


if __name__ == "__main__":
    main()
