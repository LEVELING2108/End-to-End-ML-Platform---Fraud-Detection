"""
Automated Model Promotion Script.

This script:
1. Identifies the current 'champion' model.
2. Identifies the latest 'challenger' (newest version).
3. Compares their metrics (e.g., F1-score).
4. Automatically promotes the challenger to champion if it performs better.
"""
import mlflow
from mlflow.tracking import MlflowClient

# Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "FraudDetectionModel"
ALIAS_CHAMPION = "champion"

def promote_model():
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    print(f"--- Starting Promotion Logic for {MODEL_NAME} ---")
    
    # 1. Get the current champion
    try:
        champion_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS_CHAMPION)
        champion_run = client.get_run(champion_version.run_id)
        champion_f1 = champion_run.data.metrics.get("f1", 0)
        print(f"Current Champion: Version {champion_version.version} (F1: {champion_f1:.4f})")
    except Exception:
        print("No current champion found. Promoting the latest version.")
        champion_f1 = -1
        champion_version = None

    # 2. Get the latest version (Challenger)
    latest_versions = client.get_latest_versions(MODEL_NAME)
    challenger_version = latest_versions[0] # The list is usually sorted by version descending
    
    if champion_version and challenger_version.version == champion_version.version:
        print("Latest version is already the Champion. Nothing to do.")
        return

    challenger_run = client.get_run(challenger_version.run_id)
    challenger_f1 = challenger_run.data.metrics.get("f1", 0)
    print(f"Latest Challenger: Version {challenger_version.version} (F1: {challenger_f1:.4f})")

    # 3. Promotion Logic: Promote if F1 is better (or if no champion exists)
    if challenger_f1 >= champion_f1:
        print(f"SUCCESS: Version {challenger_version.version} outperformed or matched the champion!")
        print(f"Promoting Version {challenger_version.version} to '{ALIAS_CHAMPION}'...")
        
        # Set the alias to the new version (MLflow automatically moves it)
        client.set_registered_model_alias(MODEL_NAME, ALIAS_CHAMPION, challenger_version.version)
        
        print(f"--- Promotion Complete: Version {challenger_version.version} is now the Champion ---")
    else:
        print(f"REJECTED: Challenger (F1: {challenger_f1:.4f}) did not beat Champion (F1: {champion_f1:.4f}).")
        print("Maintaining current champion.")

if __name__ == "__main__":
    promote_model()
