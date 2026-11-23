
import json
import os

import numpy as np
import torch

from model import DiabetesModel

FEATURES = ["edad", "sexo", "imc"]


def model_fn(model_dir):
    checkpoint = torch.load(
        os.path.join(model_dir, "model.pth"),
        map_location="cpu"
    )
    model = DiabetesModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    model._mean = np.array(
        [checkpoint["mean"][f] for f in FEATURES], dtype=np.float32
    )
    model._std = np.array(
        [checkpoint["std"][f] for f in FEATURES], dtype=np.float32
    )
    return model


def input_fn(request_body, content_type):
    if content_type != "application/json":
        raise ValueError(f"Content type no soportado: {content_type}")

    body = json.loads(request_body)
    return np.array(body["instances"], dtype=np.float32)


def predict_fn(data, model):
    x_norm = (data - model._mean) / model._std
    x_tensor = torch.tensor(x_norm, dtype=torch.float32)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.sigmoid(logits).numpy()

    return np.atleast_1d(probs)


def output_fn(prediction, accept):
    if accept not in ("application/json", "*/*"):
        raise ValueError(f"Accept type no soportado: {accept}")

    return json.dumps({"predictions": prediction.tolist()}), "application/json"
