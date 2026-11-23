"""
Simulation validation of the analysis pipeline.

Creates two simulated datasets in the SAME format as the real processed data and runs the analysis steps
(run_all.py --skip-download) on each, inside validation/sim_<scenario>/ (completely separate from the real data folders):

  choice_present : 25% of units carry choice signals independent of movement; choice and speed are confounded
  null_confound  : NO unit carries choice information; speed is strongly confounded with choice

Expected: in null_confound, unmatched decoding is falsely above chance, while the GLM and
movement-matched decoding stay at chance -> the controls are necessary and sufficient.

Run:  python validation/run_validation.py      (about 10 min)
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for scenario in ('choice_present', 'null_confound'):
    d = HERE / f'sim_{scenario}'
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    for f in ROOT.glob('*.py'):
        shutil.copy(f, d / f.name)
    cfg = (d / 'config.py').read_text()
    cfg = cfg.replace('N_PERM = 2000', 'N_PERM = 300').replace('N_PERM_DECODING = 200', 'N_PERM_DECODING = 30') \
             .replace('N_SUBSAMPLE = 50', 'N_SUBSAMPLE = 10')
    (d / 'config.py').write_text(cfg)
    imp = (d / '07_imposter_null.py').read_text()
    (d / '07_imposter_null.py').write_text(imp.replace('N_ROUNDS_GLM = 200', 'N_ROUNDS_GLM = 50')
                                           .replace('N_ROUNDS_DEC = 100', 'N_ROUNDS_DEC = 10'))
    subprocess.run([sys.executable, str(HERE / 'make_synthetic.py'), str(d), scenario], check=True)
    print(f'\n######## scenario: {scenario} ########')
    subprocess.run([sys.executable, str(d / 'run_all.py'), '--skip-download'], cwd=d, check=True)
print('\nValidation results: validation/sim_*/results/summary_*.txt and figures/')
