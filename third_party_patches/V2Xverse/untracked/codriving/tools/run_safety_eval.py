"""Wrapper: run inference_e2e with and without safety constraints, compare results."""
import sys, os, subprocess, json, numpy as np
from pathlib import Path

V2X_ROOT = str(Path(__file__).parent.parent.parent)
COOP_ROOT = '/raid/xuyifan/jiqiuyu'

# Step 1: Run baseline (already done, ADE=0.592, FDE=1.353)
# Step 2: We need to modify the inference loop to add safety constraint
# For now, use the existing results and compute safety metrics separately

def main():
    # Load the baseline prediction results
    # The inference_e2e.py outputs ADE/FDE but doesn't save per-sample predictions
    # We need to run it in a mode that saves predictions
    
    print("V2Xverse Safety Constraint Evaluation")
    print("=" * 50)
    print("Baseline (CoDriving): ADE=0.592, FDE=1.353")
    print()
    print("To integrate safety constraints, we need to:")
    print("1. Save per-sample predictions from baseline")
    print("2. Apply safety constraint to modify predictions")
    print("3. Re-compute ADE/FDE with modified predictions")
    print()
    print("Running modified inference with --save_npy flag...")
    
    # Run with save_npy to get per-sample outputs
    cmd = [
        sys.executable, 'codriving/tools/inference_e2e.py',
        '--config-file', './codriving/hypes_yaml/codriving/end2end_codriving_local.yaml',
        '--out-dir', './eval_output_safety',
        '--model_dir', './checkpoints/codriving/perception',
        '--planner_resume', './checkpoints/codriving/planner/codriving_planner.ckpt',
        '--save_npy',
    ]
    
    env = dict(os.environ)
    env['PYTHONPATH'] = V2X_ROOT + ':' + env.get('PYTHONPATH', '')
    env['CUDA_VISIBLE_DEVICES'] = '0'
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=V2X_ROOT)
    print("stdout:", result.stdout[-500:] if result.stdout else "")
    print("stderr:", result.stderr[-500:] if result.stderr else "")

if __name__ == '__main__':
    main()
