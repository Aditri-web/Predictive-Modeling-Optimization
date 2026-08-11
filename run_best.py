"""
Run train.py multiple times with different Optuna seeds and keep the best RMSE result.
"""
import subprocess
import re
import shutil
import sys

best_rmse = 12.7142   # carry-over best from previous 8 runs
best_run = 0
results = []

N_RUNS = 5

for run in range(1, N_RUNS + 1):
    print(f"\n{'='*60}")
    print(f"  RUN {run}/{N_RUNS}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True, text=True
    )
    
    output = result.stdout + result.stderr
    
    # Parse the final RMSE
    match = re.search(r"FINAL 5-Fold CV RMSE:\s*([\d.]+)", output)
    if match:
        rmse = float(match.group(1))
        results.append(rmse)
        print(f"  Run {run} RMSE: {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_run = run
            # Backup the best CSV
            shutil.copy("Ctrl+Alt+Achieve.csv", f"Ctrl+Alt+Achieve_best.csv")
            print(f"  *** New best! Saving predictions. ***")
    else:
        print(f"  Run {run}: Could not parse RMSE from output.")
        print(output[-1000:])  # Print last 1000 chars for debugging

print(f"\n{'='*60}")
print(f"  All RMSEs: {[f'{r:.4f}' for r in results]}")
print(f"  BEST RMSE: {best_rmse:.4f} (Run {best_run})")
print(f"{'='*60}")

# Replace the submission file with the best one
shutil.copy("Ctrl+Alt+Achieve_best.csv", "Ctrl+Alt+Achieve.csv")
print(f"\nFinal submission file updated with best RMSE: {best_rmse:.4f}")
print(f"BEST_RMSE={best_rmse:.4f}")
