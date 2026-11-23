
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel


REGION = "us-east-1"
BUCKET = os.environ["BUCKET_NAME"]
ROLE = os.environ["SAGEMAKER_ROLE_ARN"]

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
INFERENCE_DIR = BASE_DIR / "inference"
TRAINING_DIR = BASE_DIR / "training"

IMAGE_URI = (
    f"763104351884.dkr.ecr.{REGION}.amazonaws.com/"
    "pytorch-inference:2.1.0-cpu-py310-ubuntu20.04-sagemaker"
)


def build_model_tar(mode, work_dir):
    """
    Estructura dentro del tar.gz:
      model.pth
      code/inference.py
      code/model.py
    """
    code_dir = os.path.join(work_dir, f"code-{mode}")
    os.makedirs(code_dir, exist_ok=True)

    shutil.copy(
        str(INFERENCE_DIR / "inference.py"),
        os.path.join(code_dir, "inference.py")
    )
    shutil.copy(
        str(TRAINING_DIR / "model.py"),
        os.path.join(code_dir, "model.py")
    )

    tar_path = os.path.join(work_dir, f"model-{mode}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(
            str(RESULTS_DIR / mode / "model.pth"),
            arcname="model.pth"
        )
        tar.add(code_dir, arcname="code")

    return tar_path


def upload_to_s3(local_path, s3_key):
    s3 = boto3.client("s3", region_name=REGION)
    s3.upload_file(local_path, BUCKET, s3_key)
    return f"s3://{BUCKET}/{s3_key}"


def deploy_model(mode, s3_model_uri, sm_session):
    endpoint_name = f"diabetes-{mode}"

    pytorch_model = PyTorchModel(
        model_data=s3_model_uri,
        role=ROLE,
        entry_point="inference.py",
        framework_version="2.1.0",
        py_version="py310",
        image_uri=IMAGE_URI,
        sagemaker_session=sm_session,
    )

    pytorch_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",
        endpoint_name=endpoint_name,
        wait=True,
    )

    return endpoint_name


def main():
    boto_sess = boto3.Session(region_name=REGION)
    sm_session = sagemaker.Session(boto_session=boto_sess)

    endpoints = {}

    with tempfile.TemporaryDirectory() as work_dir:
        for mode in ["conventional", "private"]:
            print(f"\n[{mode}] Construyendo tar.gz...")
            tar_path = build_model_tar(mode, work_dir)

            # Verificar estructura antes de subir
            print(f"[{mode}] Contenido del tar.gz:")
            with tarfile.open(tar_path, "r:gz") as tar:
                for member in tar.getmembers():
                    print(f"  {member.name}")

            s3_key = f"inference/{mode}/model.tar.gz"
            print(f"[{mode}] Subiendo a s3://{BUCKET}/{s3_key}...")
            s3_uri = upload_to_s3(tar_path, s3_key)

            print(f"[{mode}] Desplegando endpoint diabetes-{mode} (~5-8 min)...")
            endpoint_name = deploy_model(mode, s3_uri, sm_session)
            endpoints[mode] = endpoint_name
            print(f"[{mode}] Endpoint activo: {endpoint_name}")

    endpoints_path = RESULTS_DIR / "endpoints.json"
    with open(endpoints_path, "w", encoding="utf-8") as f:
        json.dump(endpoints, f, indent=2)

    print(f"\nEndpoints guardados en {endpoints_path}")
    print(json.dumps(endpoints, indent=2))


if __name__ == "__main__":
    main()
