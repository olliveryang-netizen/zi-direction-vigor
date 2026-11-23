"""Synthetic processed data with planted effects, in the same format as 01_download_preprocess.py.

Usage: python make_synthetic.py <folder> [choice_present | null_confound]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

proj = Path(sys.argv[1])
scenario = sys.argv[2] if len(sys.argv) > 2 else 'choice_present'
sys.path.insert(0, str(proj))
import config as C  # noqa: E402

rng = np.random.default_rng(7)
edges = np.round(np.arange(C.EDGES[0], C.EDGES[1] + C.BIN / 2, C.BIN), 6)
cent = (edges[:-1] + edges[1:]) / 2
types = ['choice', 'speed_pos', 'speed_neg', 'mixed', 'prior', 'none']
# choice_present: planted choice units; null_confound: NO choice units, strong choice-speed confound
probs = [0.15, 0.2, 0.1, 0.1, 0.1, 0.35] if scenario == 'choice_present' else [0.0, 0.35, 0.15, 0.0, 0.1, 0.4]
n_ins = 32
subjects = [f'M{i:02d}' for i in range(24)]

for k in range(n_ins):
    pid = f'pid{k:03d}'
    out = C.PROCESSED_DIR / pid
    out.mkdir(parents=True, exist_ok=True)
    n = int(rng.integers(300, 700))
    blocks = np.repeat(rng.choice([0.2, 0.8], n // 50 + 2), 50)[:n]
    blocks[:90] = 0.5
    contr = rng.choice([0, 0.0625, 0.125, 0.25, 1.0], n)
    side = np.where(rng.random(n) < blocks, -1, 1)          # probabilityLeft -> left stimulus (-)
    sc = np.where(contr == 0, 0, side * contr)
    # psychometric: P(choice=+1) decreases with signed contrast, shifted by block prior
    logit = -8 * sc + 2.0 * (blocks - 0.5)
    choice = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), 1, -1)
    correct = np.where(sc == 0, rng.random(n) < 0.5, choice == -np.sign(sc)).astype(int)
    so = np.cumsum(rng.uniform(2.0, 3.0, n))
    rt = np.exp(rng.normal(np.log(0.18), 0.35, n))
    fm = so + rt
    conf = rng.uniform(0.5, 2.0) if scenario == 'choice_present' else rng.uniform(1.5, 3.0)                             # choice->speed confound strength
    speed = np.exp(rng.normal(np.log(4), 0.35, n)) * (1 + 0.25 * conf * (choice == 1))
    dur = np.exp(rng.normal(np.log(0.3), 0.3, n))
    trials = pd.DataFrame(dict(
        stimOn_times=so, firstMovement_times=fm, choice=choice, feedbackType=np.where(correct == 1, 1, -1),
        contrastLeft=np.where(sc < 0, contr, np.where(sc == 0, 0, np.nan)),
        contrastRight=np.where(sc > 0, contr, np.nan), probabilityLeft=blocks,
        response_times=fm + dur, feedback_times=fm + dur + 0.01, trial_number=np.arange(n),
        signed_contrast=sc, correct=correct, rt=rt, move_end=fm + dur, peak_speed=speed,
        mean_speed=speed * 0.6, wheel_disp=choice * speed * 0.1, move_dur=dur))
    for w in C.WINDOWS:
        trials[f'me_whisker_{w}'] = 0.5 * speed + rng.normal(0, 1, n)
        trials[f'me_body_{w}'] = 0.3 * speed + rng.normal(0, 1, n)

    nu = int(rng.integers(3, 21))
    utype = rng.choice(types, nu, p=probs)
    y_ap = rng.normal(-2.2e-3, 3e-4, nu)
    # location gradient: positive speed units anterior, negative posterior
    y_ap[utype == 'speed_pos'] += 2.5e-4
    y_ap[utype == 'speed_neg'] -= 2.5e-4
    units = pd.DataFrame(dict(unit_index=np.arange(nu), x=rng.normal(-1e-3, 2e-4, nu), y=y_ap,
                              z=rng.normal(-4.5e-3, 2e-4, nu), firing_rate=rng.uniform(5, 30, nu),
                              eid=f'eid{k:03d}', pid=pid, subject=subjects[k % len(subjects)], lab='lab',
                              gt_type=utype))

    sz = (np.log(speed) - np.log(speed).mean()) / np.log(speed).std()
    fm_counts = np.zeros((nu, n, len(cent)), np.uint16)
    so_counts = np.zeros_like(fm_counts)
    ramp = np.clip((cent + 0.25) / 0.25, 0, 1) * (cent < 0.15)       # choice: -0.25..0 ramp
    mov = np.exp(-((cent - 0.1) / 0.12) ** 2)                         # movement profile
    for i, ty in enumerate(utype):
        base = rng.uniform(5, 20)
        r = np.full((n, len(cent)), base)
        if ty in ('choice', 'mixed'):
            r += 8 * (choice[:, None] == 1) * ramp[None, :]
        if ty in ('speed_pos', 'mixed'):
            r += 0.5 * base * mov[None, :] + 5 * sz[:, None] * mov[None, :]
        if ty == 'speed_neg':
            r += 0.5 * base * mov[None, :] - 4 * sz[:, None] * mov[None, :]
        fm_counts[i] = rng.poisson(np.clip(r, 0.1, None) * C.BIN)
        rs = np.full((n, len(cent)), base)
        if ty == 'prior':
            rs += 10 * (blocks[:, None] - 0.5)
        so_counts[i] = rng.poisson(np.clip(rs, 0.1, None) * C.BIN)
    wv = (choice * speed)[:, None] * mov[None, :]
    trials.to_parquet(out / 'trials.parquet')
    units.to_parquet(out / 'units.parquet')
    np.savez_compressed(out / 'binned.npz', fm_counts=fm_counts, so_counts=so_counts, edges=edges,
                        wheel_vel_fm=wv.astype(np.float32))
    (out / 'meta.json').write_text(json.dumps(dict(eid=f'eid{k:03d}', pid=pid, subject=subjects[k % len(subjects)],
                                                   lab='lab', date='2020-01-01', probe_name='probe00',
                                                   n_units=nu, n_trials=n, motion_energy={'whisker': True, 'body': True})))
print('synthetic insertions written:', n_ins)
