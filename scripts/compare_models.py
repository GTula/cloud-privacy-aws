
import json
from pathlib import Path

# Carpeta principal del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


def load_metrics(mode):
    path = RESULTS_DIR / mode / "metrics.json"

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {path}\n"
            f"Descargá metrics.json del entrenamiento {mode}."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_value(value, decimals=4):
    if value is None:
        return "N/A"

    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"

    return str(value)


def compare_models():
    conventional = load_metrics("conventional")
    private = load_metrics("private")

    print("\n" + "=" * 65)
    print("COMPARACIÓN DE MODELOS - CLOUD PRIVACY")
    print("=" * 65)

    print(
        f"{'Métrica':<25}"
        f"{'Convencional':<20}"
        f"{'Privado':<20}"
    )
    print("-" * 65)

    # Métricas del conjunto de prueba
    metrics = [
        ("Accuracy", "accuracy"),
        ("F1-score", "f1"),
        ("AUC-ROC", "auc"),
        ("Loss", "loss"),
    ]

    for label, key in metrics:
        normal_value = conventional["test"].get(key)
        private_value = private["test"].get(key)

        print(
            f"{label:<25}"
            f"{format_value(normal_value):<20}"
            f"{format_value(private_value):<20}"
        )

    # Tiempo de entrenamiento
    print(
        f"{'Tiempo (segundos)':<25}"
        f"{format_value(conventional.get('training_seconds'), 2):<20}"
        f"{format_value(private.get('training_seconds'), 2):<20}"
    )

    # Parámetros de privacidad
    print("-" * 65)

    print(
        f"{'Epsilon (ε)':<25}"
        f"{'N/A':<20}"
        f"{format_value(private.get('epsilon')):<20}"
    )

    print(
        f"{'Delta (δ)':<25}"
        f"{'N/A':<20}"
        f"{format_value(private.get('delta'), 8):<20}"
    )

    print(
        f"{'Noise multiplier':<25}"
        f"{'N/A':<20}"
        f"{format_value(private.get('noise_multiplier')):<20}"
    )

    print("=" * 65)

    # Diferencias entre modelos
    normal_acc = conventional["test"]["accuracy"]
    private_acc = private["test"]["accuracy"]

    accuracy_diff = (private_acc - normal_acc) * 100

    normal_time = conventional["training_seconds"]
    private_time = private["training_seconds"]

    print("\nANÁLISIS DE RESULTADOS")

    print(
        f"Diferencia de accuracy: "
        f"{accuracy_diff:+.2f} puntos porcentuales"
    )

    if normal_time > 0:
        time_increase = (
            (private_time - normal_time) / normal_time
        ) * 100

        print(
            f"Diferencia de tiempo: "
            f"{time_increase:+.2f}%"
        )

    print("\nCONCLUSIÓN")

    if accuracy_diff < 0:
        print(
            "El modelo privado obtuvo menor precisión "
            "que el convencional."
        )
    elif accuracy_diff > 0:
        print(
            "El modelo privado obtuvo mayor precisión "
            "que el convencional."
        )
    else:
        print(
            "Ambos modelos obtuvieron la misma precisión."
        )

    print(
        "La evaluación de privacidad requiere analizar "
        "también el ataque de inferencia de pertenencia."
    )


if __name__ == "__main__":
    compare_models()
