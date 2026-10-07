from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def _train_model():
    import subprocess

    result = subprocess.run(
        ["/opt/airflow/venv-ml/bin/python", "-m", "tp_mlops2.train"],
        capture_output=True,
        text=True,
        cwd="/opt/airflow",
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"train.py falló con código {result.returncode}")


def _upload_to_datalake():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "/opt/airflow/scripts/upload_to_datalake.py"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"upload_to_datalake.py falló con código {result.returncode}")


default_args = {
    "owner": "tp_mlops2",
    "retries": 1,
}

with DAG(
    dag_id="train_demanda_electrica",
    description="Reentrena el modelo de demanda eléctrica y lo publica en el Data Lake",
    default_args=default_args,
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops2", "demanda-electrica"],
) as dag:
    train_task = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
    )

    upload_task = PythonOperator(
        task_id="upload_to_datalake",
        python_callable=_upload_to_datalake,
    )

    train_task >> upload_task
