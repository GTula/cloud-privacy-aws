
import argparse
import os

import boto3
import sagemaker

from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    required=True,
    choices=["conventional", "private"]
)
args = parser.parse_args()

REGION = "us-east-1"

BUCKET = os.environ["BUCKET_NAME"]
ROLE = os.environ["SAGEMAKER_ROLE_ARN"]

boto_session = boto3.Session(region_name=REGION)
sm_session = sagemaker.Session(
    boto_session=boto_session
)

# Imagen CPU oficial para SageMaker.
# Verificar compatibilidad de Opacus antes del job.
IMAGE_URI = (
    f"763104351884.dkr.ecr.{REGION}.amazonaws.com/"
    "pytorch-training:"
    "2.10.0-cpu-py313-ubuntu22.04-sagemaker"
)

estimator = PyTorch(
    entry_point="train.py",
    source_dir="training",
    role=ROLE,
    image_uri=IMAGE_URI,
    framework_version="2.10.0",
    py_version="py313",
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size=20,
    max_run=1800,
    output_path=f"s3://{BUCKET}/models/{args.mode}",
    base_job_name=f"cloud-privacy-{args.mode}",
    sagemaker_session=sm_session,
    hyperparameters={
        "mode": args.mode,
        "epochs": 10,
        "batch-size": 64,
        "lr": 0.05,
        "epsilon": 5.0,
        "delta": 1e-5,
        "max-grad-norm": 1.0
    }
)

input_data = TrainingInput(
    s3_data=f"s3://{BUCKET}/data/processed/",
    input_mode="File",
    content_type="text/csv"
)

print(f"Iniciando entrenamiento {args.mode}...")

estimator.fit(
    {"training": input_data},
    wait=True,
    logs=True
)

print("Entrenamiento finalizado.")
print("Job:", estimator.latest_training_job.name)
print("Modelo:", estimator.model_data)
