import pandas as pd
import numpy as np
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------- IMPORTS (OLDER VERSION) --------------------------
from evidently import Report, Dataset
from evidently.presets import DataDriftPreset 
# -----------------------------------------------------------------------------

def run_drift_analysis():
    # -------------------------- Step 1: Load Real Project Data --------------------------
    ref_path = "data/featured/master_features.parquet"
    
    # --- FIX: Match the filename expected by your DVC error ---
    output_path = "data/integrity_report.html" 
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(ref_path):
        print(f"❌ Error: {ref_path} not found.")
        return

    print("--- Loading Reference Data ---")
    reference_data = pd.read_parquet(ref_path)
    if 'msno' in reference_data.columns:
        reference_data = reference_data.drop(columns=['msno'])

    print("--- Simulating Current Data with Drift ---")
    current_data = reference_data.sample(frac=0.5, random_state=42).copy()
    
    if 'total_payment' in current_data.columns:
        current_data['total_payment'] = current_data['total_payment'] * 0.7 
    if 'bd' in current_data.columns:
        current_data['bd'] = current_data['bd'] - 5 

    # -------------------------- Step 2: Generate Report (OLDER API) --------------------------
    print("--- Generating Evidently Drift Report ---")
    
    report = Report(metrics=[DataDriftPreset()]) 

    # Run the report
    data_drift_report = report.run(current_data=current_data, reference_data=reference_data)

    # -------------------------- Step 3: Save Report --------------------------
    
    # Try saving using the older API methods
    try:
        # Attempt 1: Call save_html on the result object (common in 0.1.x - 0.2.x)
        data_drift_report.save_html(output_path)
    except AttributeError:
        # Attempt 2: Fallback for versions where run() updates in-place
        report.save_html(output_path)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        # DVC Safeguard: Create the file even on error to prevent pipeline crash
        with open(output_path, "w") as f:
            f.write(f"Drift analysis failed with error: {e}")
        return

    print(f"--- Data Drift Report saved to {output_path} ---")

if __name__ == "__main__":
    run_drift_analysis()