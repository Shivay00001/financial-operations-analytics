"""Financial Operations Analytics — pipeline entrypoint.

Runs the end-to-end analytics pipeline:
  1. data_generator.py  (only if the CSV dataset is missing)
  2. revenue_forecast.py (XGBoost 30-day revenue forecast)
  3. churn_prediction.py  (RandomForest churn classifier)

Matches the Dockerfile CMD: `python main.py`.
"""
import os
import subprocess
import sys

DATA_FILES = ["customers.csv", "orders.csv", "order_items.csv", "payments.csv"]


def run(script: str) -> None:
    print(f"\n===== Running {script} =====", flush=True)
    subprocess.run([sys.executable, script], check=True)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    if not all(os.path.exists(f) for f in DATA_FILES):
        run("data_generator.py")
    else:
        print("Dataset CSVs already present — skipping data generation.")

    run("revenue_forecast.py")
    run("churn_prediction.py")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
