
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from opacus import PrivacyEngine

from model import DiabetesModel


FEATURES = ["edad", "sexo", "imc"]
TARGET = "diabetes"


def load_csv(path):
    df = pd.read_csv(path)

    required = FEATURES + [TARGET]
    if not set(required).issubset(df.columns):
        raise ValueError(f"Faltan columnas en {path}")

    if df[required].isna().any().any():
        raise ValueError(f"Hay valores nulos en {path}")

    return df


def prepare_data(data_dir):
    train = load_csv(data_dir / "train.csv")
    val = load_csv(data_dir / "validation.csv")
    test = load_csv(data_dir / "test.csv")

    # Calcular estadísticas SOLO con train.
    mean = train[FEATURES].mean()
    std = train[FEATURES].std(ddof=0).replace(0, 1)

    def convert(df):
        x = ((df[FEATURES] - mean) / std).to_numpy(
            dtype=np.float32
        )
        y = df[TARGET].to_numpy(dtype=np.float32)

        return (
            torch.tensor(x),
            torch.tensor(y)
        )

    return (
        convert(train),
        convert(val),
        convert(test),
        mean.to_dict(),
        std.to_dict(),
        len(train)
    )


@torch.no_grad()
def evaluate(model, data, criterion):
    model.eval()
    x, y = data

    logits = model(x)
    loss = criterion(logits, y).item()
    probs = torch.sigmoid(logits).numpy()
    labels = (probs >= 0.5).astype(int)

    real = y.numpy().astype(int)

    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy_score(real, labels)),
        "f1": float(f1_score(
            real, labels, zero_division=0
        )),
    }

    if len(np.unique(real)) == 2:
        metrics["auc"] = float(
            roc_auc_score(real, probs)
        )
    else:
        metrics["auc"] = None

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["conventional", "private"],
        required=True
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--epsilon", type=float, default=5.0
    )
    parser.add_argument(
        "--delta", type=float, default=1e-5
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("SM_CHANNEL_TRAINING", "data")
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get(
            "SM_MODEL_DIR", "results/model"
        )
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs y batch-size deben ser positivos")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train, val, test, mean, std, n_train = prepare_data(
        data_dir
    )

    if args.mode == "private" and not (
        0 < args.delta < 1 / n_train
    ):
        raise ValueError(
            "Elegir delta > 0 y menor a 1/N_train"
        )

    train_loader = DataLoader(
        TensorDataset(*train),
        batch_size=args.batch_size,
        shuffle=True
    )

    model = DiabetesModel()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr
    )
    criterion = nn.BCEWithLogitsLoss()

    privacy_engine = None

    if args.mode == "private":
        privacy_engine = PrivacyEngine()

        model, optimizer, train_loader = (
            privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=args.epsilon,
                target_delta=args.delta,
                epochs=args.epochs,
                max_grad_norm=args.max_grad_norm,
            )
        )

    print(f"Modo: {args.mode}", flush=True)
    print(f"Pacientes de entrenamiento: {n_train}", flush=True)

    start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0.0
        batches = 0

        for x, y in train_loader:
            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        # Evaluar sobre datos separados del entrenamiento.
        val_metrics = evaluate(model, val, criterion)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={total_loss / max(batches, 1):.4f} | "
            f"val_accuracy={val_metrics['accuracy']:.4f}",
            flush=True
        )

    duration = time.perf_counter() - start

    # Evaluación final. No utilizar test para elegir
    # hiperparámetros después de observar resultados.
    val_metrics = evaluate(model, val, criterion)
    test_metrics = evaluate(model, test, criterion)

    epsilon_spent = None
    noise_multiplier = None

    if privacy_engine is not None:
        epsilon_spent = float(
            privacy_engine.get_epsilon(args.delta)
        )
        noise_multiplier = float(
            optimizer.noise_multiplier
        )

    # Opacus envuelve la red en un módulo adicional.
    original_model = (
        model._module
        if privacy_engine is not None
        else model
    )

    checkpoint = {
        "model_state_dict": original_model.state_dict(),
        "features": FEATURES,
        "mean": mean,
        "std": std,
        "mode": args.mode,
    }

    torch.save(
        checkpoint,
        model_dir / "model.pth"
    )

    metrics = {
        "mode": args.mode,
        "training_samples": n_train,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "training_seconds": duration,
        "validation": val_metrics,
        "test": test_metrics,
        "epsilon": epsilon_spent,
        "delta": (
            args.delta if privacy_engine else None
        ),
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": (
            args.max_grad_norm if privacy_engine else None
        )
    }

    with open(
        model_dir / "metrics.json", "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    print("\nRESULTADOS FINALES", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(
        f"Modelo guardado en {model_dir}",
        flush=True
    )


if __name__ == "__main__":
    main()
