"""
Run steps 01-07, 09 and 10 of the pipeline for one region.

  python run_all.py                  # download/preprocess + all analyses
  python run_all.py --skip-download  # analyses only (data already in processed/)
  python run_all.py --only 03        # one step
  python run_all.py --region MOs     # same pipeline on a control region (own folders: processed_MOs/ ...)
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ['01_download_preprocess.py', '02_dataset_behavior.py', '03_single_unit_glm.py',
         '04_decoding.py', '05_subspace.py', '06_location.py', '07_imposter_null.py', '09_followup.py', '10_reversal_history.py']

ap = argparse.ArgumentParser()
ap.add_argument('--skip-download', action='store_true')
ap.add_argument('--only', default=None, help='step number, e.g. 03')
ap.add_argument('--region', default='ZI', help='Beryl acronym, e.g. ZI (default), MOs, RSPd, VPM, SCm')
args = ap.parse_args()
env = dict(os.environ, ANALYSIS_REGION=args.region)
print(f'Region: {args.region}')

steps = STEPS[1:] if args.skip_download else STEPS
if args.only:
    steps = [s for s in STEPS if s.startswith(args.only)]
for s in steps:
    t0 = time.time()
    print(f'\n========== {s} ==========', flush=True)
    r = subprocess.run([sys.executable, str(HERE / s)], cwd=HERE, env=env)
    if r.returncode != 0:
        sys.exit(f'{s} failed (exit code {r.returncode}). Fix the error above, then rerun with --only {s[:2]} --region {args.region}')
    print(f'{s} finished in {(time.time() - t0) / 60:.1f} min', flush=True)
print('\nAll done. Quick-look figures in figures/, tables and summary_*.txt in results/.')
