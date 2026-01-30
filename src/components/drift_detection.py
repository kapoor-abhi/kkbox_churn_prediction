import pandas as pd
import numpy as np
import os
import sys
import json
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# -------------------------- IMPORTS (FIXED: OLDER VERSION COMPATIBLE) --------------------------
from evidently import Report
# FIX: Use 'evidently.presets' instead of 'evidently.metric_preset'
from evidently.presets import DataDriftPreset 
# -----------------------------------------------------------------------------------------------

def run_drift_analysis():
    # -------------------------- Step 1: Load Real Project Data --------------------------
    ref_path = "data/featured/master_features.parquet"
    curr_path = "data/live_traffic.jsonl"
    output_path = "data/monitoring/drift_report.html" 
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Load Reference (Training) Data
    if not os.path.exists(ref_path):
        print(f"❌ Error: Reference data {ref_path} not found.")
        return

    print(f"--- Loading Reference Data from {ref_path} ---")
    reference_data = pd.read_parquet(ref_path)
    
    # Drop target if present in reference (ground truth)
    if 'is_churn' in reference_data.columns:
        reference_data = reference_data.drop(columns=['is_churn'])

    # 2. Load Current (Live) Data
    print(f"--- Loading Current Data from {curr_path} ---")
    if not os.path.exists(curr_path):
        print("⚠️ No live traffic logs found. Skipping drift check.")
        return

    current_data_list = []
    try:
        with open(curr_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # Flatten the input dictionary
                    flat_record = record['input']
                    # Add the prediction probability if needed
                    flat_record['prediction_prob'] = record['prediction_prob']
                    current_data_list.append(flat_record)
                except Exception:
                    continue
        
        if not current_data_list:
            print("⚠️ Live traffic log is empty. Skipping.")
            return

        current_data = pd.DataFrame(current_data_list)
        
        # Ensure columns match (Intersection of columns)
        common_cols = [c for c in reference_data.columns if c in current_data.columns]
        reference_data = reference_data[common_cols]
        current_data = current_data[common_cols]
        
        print(f"✅ Loaded {len(current_data)} live requests for analysis.")

    except Exception as e:
        print(f"❌ Error loading live logs: {e}")
        return

    # -------------------------- Step 2: Generate Report --------------------------
    print("--- Generating Evidently Drift Report ---")
    
    # FIX: Initialize Report using the import from 'evidently.presets'
    report = Report(metrics=[DataDriftPreset()]) 

    # FIX: Assign the result to a variable (older API style)
    data_drift_report = report.run(current_data=current_data, reference_data=reference_data)

    # -------------------------- Step 3: Save Report --------------------------
    try:
        # FIX: Robust saving method handling both API versions
        try:
            # Attempt 1: Call save_html on the result object
            data_drift_report.save_html(output_path)
        except AttributeError:
            # Attempt 2: Fallback for versions where run() updates in-place
            report.save_html(output_path)
            
        print(f"✅ Data Drift Report saved to {output_path}")

        # --- Check for Drift Programmatically ---
        # Note: The structure of as_dict() may vary between versions. 
        # We wrap this in a try/except to prevent crashing if the dictionary structure is different.
        try:
            results = report.as_dict()
            # Depending on version, 'metrics' might be a list or direct dict keys. 
            # This logic attempts to find drift share generally.
            drift_share = results['metrics'][0]['result']['drift_share']
            drift_detected = results['metrics'][0]['result']['dataset_drift']
            
            print(f"📊 Drift Share: {drift_share:.2%}")
            if drift_detected:
                print("🚨 ALERT: Significant Data Drift Detected!")
            else:
                print("🟢 System Status: Healthy (No Drift)")
        except Exception:
             print("ℹ️  Report saved, but programmatic drift check skipped (API version mismatch).")

    except Exception as e:
        print(f"❌ An error occurred saving the report: {e}")
        return

if __name__ == "__main__":
    run_drift_analysis()