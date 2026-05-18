"""
Self-Healing Orchestrator.

This script acts as the 'brain' of the platform:
1. Monitors data for drift.
2. If drift is detected, triggers advanced retraining.
3. Automatically promotes the new model if it beats the current champion.
"""
import os
import pandas as pd
import subprocess
import time
from src.monitoring import DriftMonitor
import mlflow
from mlflow.tracking import MlflowClient

# Configuration
REFERENCE_DATA = "data/train.csv"
PRODUCTION_DATA = "data/test.csv"  # In real life, this would be new live data
DRIFT_THRESHOLD = 0.20 # 20% of features drifted triggers retraining
MODEL_NAME = "FraudDetectionModel"
ALIAS_CHAMPION = "champion"

def run_command(command, description):
    """Utility to run shell commands and log output."""
    print(f"\n[ORCHESTRATOR] {description}...")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"SUCCESS: {description}")
        return True
    else:
        print(f"ERROR: {description}")
        print(result.stderr)
        return False

def check_for_drift():
    """Check if the data has drifted enough to warrant retraining."""
    if not os.path.exists(REFERENCE_DATA) or not os.path.exists(PRODUCTION_DATA):
        print("Data files missing. Cannot check for drift.")
        return False
        
    ref_df = pd.read_csv(REFERENCE_DATA)
    prod_df = pd.read_csv(PRODUCTION_DATA)
    
    # We'll simulate drift for this demonstration if needed, 
    # but let's use the real monitor logic
    monitor = DriftMonitor(ref_df, feature_columns=['amount', 'hour', 'day_of_week'])
    
    # Let's simulate a 'drifted' production set for this demo 
    # by slightly inflating amounts in the test set
    drifted_prod = prod_df.copy()
    drifted_prod['amount'] = drifted_prod['amount'] * 1.5
    
    result = monitor.check_drift(drifted_prod, threshold=DRIFT_THRESHOLD)
    print(f"Drift Analysis: {result['drift_share']:.1%} of features drifted.")
    
    return result['alert']

def heal_system():
    """The self-healing loop."""
    print("="*60)
    print(f"SYSTEM SCAN: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 1. MONITOR
    drift_detected = check_for_drift()
    
    if drift_detected:
        print("🚨 ALERT: Significant data drift detected! Initiating Self-Healing...")
        
        # 2. RETRAIN (Advanced)
        if run_command("python src/train_advanced.py", "Retraining model with new data"):
            
            # 3. PROMOTE
            if run_command("python src/promote_model.py", "Evaluating and promoting new model"):
                print("✅ SYSTEM HEALED: New champion model is live.")
            else:
                print("⚠️ HEALING INCOMPLETE: Retraining succeeded but promotion failed (performance gate).")
        else:
            print("❌ HEALING FAILED: Retraining error.")
            
    else:
        print("✅ SYSTEM HEALTHY: No significant drift detected.")

if __name__ == "__main__":
    heal_system()
