
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "training"))

from model import DiabetesModel

FEATURES = ["edad", "sexo", "imc"]
TARGET = "diabetes"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"


def download_data_from_s3():
    """Descarga train.csv y test.csv desde S3 si no están disponibles localmente."""
    bucket = os.environ.get("BUCKET_NAME")
    if not bucket:
        raise FileNotFoundError(
            "No se encontraron los archivos de datos localmente.\n"
            "Opciones:\n"
            "  1. Ejecutar src/preprocess.py para generarlos desde Synthea.\n"
            "  2. Setear BUCKET_NAME y ejecutar este script (descargará de S3).\n"
            f"  3. Copiar train.csv y test.csv a {DATA_DIR}/"
        )

    import boto3
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name="us-east-1")
    for split in ["train", "test"]:
        key = f"data/processed/{split}.csv"
        dest = DATA_DIR / f"{split}.csv"
        print(f"Descargando s3://{bucket}/{key}...")
        s3.download_file(bucket, key, str(dest))


def load_model_with_stats(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = DiabetesModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = np.array(
        [checkpoint["mean"][f] for f in FEATURES], dtype=np.float32
    )
    std = np.array(
        [checkpoint["std"][f] for f in FEATURES], dtype=np.float32
    )
    return model, mean, std


def compute_per_sample_loss(model, mean, std, df):
    x_raw = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)

    x_norm = (x_raw - mean) / std
    x_tensor = torch.tensor(x_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    with torch.no_grad():
        logits = model(x_tensor)
        losses = criterion(logits, y_tensor).numpy()

    return losses


def tpr_at_fixed_fpr(fpr_arr, tpr_arr, target_fpr):
    idx = np.searchsorted(fpr_arr, target_fpr, side="right")
    if idx == 0:
        return 0.0
    if idx >= len(tpr_arr):
        return float(tpr_arr[-1])
    fpr_lo, fpr_hi = fpr_arr[idx - 1], fpr_arr[idx]
    tpr_lo, tpr_hi = tpr_arr[idx - 1], tpr_arr[idx]
    if fpr_hi == fpr_lo:
        return float(tpr_hi)
    t = (target_fpr - fpr_lo) / (fpr_hi - fpr_lo)
    return float(tpr_lo + t * (tpr_hi - tpr_lo))


def run_attack(model, mean, std, train_df, test_df, model_name):
    train_losses = compute_per_sample_loss(model, mean, std, train_df)
    test_losses = compute_per_sample_loss(model, mean, std, test_df)

    all_losses = np.concatenate([train_losses, test_losses])
    all_labels = np.concatenate([
        np.ones(len(train_df), dtype=int),
        np.zeros(len(test_df), dtype=int),
    ])

    # Score: menor pérdida → mayor probabilidad de ser miembro
    scores = -all_losses

    auc = float(roc_auc_score(all_labels, scores))
    advantage = float(2.0 * (auc - 0.5))

    fpr_arr, tpr_arr, _ = roc_curve(all_labels, scores)
    tpr_01 = tpr_at_fixed_fpr(fpr_arr, tpr_arr, target_fpr=0.1)

    return {
        "model": model_name,
        "attack_auc": auc,
        "advantage": advantage,
        "tpr_at_fpr_0.1": tpr_01,
        "train_mean_loss": float(train_losses.mean()),
        "test_mean_loss": float(test_losses.mean()),
        "loss_gap": float(test_losses.mean() - train_losses.mean()),
    }


def print_results_table(results):
    conv = results["conventional"]
    priv = results["private"]

    rows = [
        ("AUC-ROC del ataque",    "attack_auc"),
        ("Ventaja del atacante",  "advantage"),
        ("TPR @ FPR=0.1",         "tpr_at_fpr_0.1"),
        ("Pérdida media (train)", "train_mean_loss"),
        ("Pérdida media (test)",  "test_mean_loss"),
        ("Brecha de pérdida",     "loss_gap"),
    ]

    sep = "=" * 72
    print(f"\n{sep}")
    print("ATAQUE DE INFERENCIA DE PERTENENCIA — BASADO EN PÉRDIDA")
    print(sep)
    print(f"{'Métrica':<26}{'Convencional':>16}{'Privado (DP)':>16}{'Δ (priv-conv)':>14}")
    print("-" * 72)
    for label, key in rows:
        c = conv[key]
        p = priv[key]
        print(f"{label:<26}{c:>16.4f}{p:>16.4f}{(p - c):>+14.4f}")
    print(sep)
    print("\nINTERPRETACIÓN")
    print("  AUC ~ 0.5 → privacidad fuerte (el ataque adivina al azar)")
    print("  AUC ~ 1.0 → el modelo memorizó datos de entrenamiento")
    print(f"\n  AUC convencional: {conv['attack_auc']:.4f}")
    print(f"  AUC privado (DP): {priv['attack_auc']:.4f}")
    if priv['attack_auc'] < conv['attack_auc']:
        reduction = (conv['attack_auc'] - priv['attack_auc']) / max(conv['attack_auc'] - 0.5, 1e-9)
        print(f"  → DP reduce la ventaja del atacante en {reduction * 100:.1f}%")


def main():
    if not (DATA_DIR / "train.csv").exists() or not (DATA_DIR / "test.csv").exists():
        download_data_from_s3()

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    results = {}
    for mode in ["conventional", "private"]:
        model_path = RESULTS_DIR / mode / "model.pth"
        print(f"Cargando modelo {mode}...")
        model, mean, std = load_model_with_stats(model_path)
        results[mode] = run_attack(model, mean, std, train_df, test_df, mode)

    print_results_table(results)

    out_path = RESULTS_DIR / "membership_attack.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados en {out_path}")


if __name__ == "__main__":
    main()
