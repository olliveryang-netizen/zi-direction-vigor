"""
Step 12 - Directional body-movement control with video pose estimation (paper Figure 6F).

Video motion energy only measures HOW MUCH the animal moves, not WHICH WAY. If left and right
choices are accompanied by different paw / head movements or posture, a "direction" signal could
partly reflect those movements (or the sensory feedback they generate). Here we use IBL pose
estimation (left camera; Lightning Pose if available, otherwise DLC) to measure, in the premove
window (-160..0 ms from movement onset), the position AND velocity (x, y) of both forepaws and the
nose tip, and ask whether direction information survives controlling for them.

Per insertion (premove window):
  1  pose-only decoding: how well do paw/nose features alone predict the upcoming direction?
  2  GLM: fraction of choice units with the step-03 design vs design + pose features (same trials)
  3  neural decoding before vs after regressing the pose features out of every unit
     (speed + block matched, minus imposter chance) -> retention = residualised / original
Residualisation is conservative: it also removes neural variance that merely co-varies with pose
through the choice itself, so retention < 1 is expected even for a signal that does not depend on
pose. Interpret retention by COMPARING regions, and use the GLM (choice effect over and above pose)
as the primary per-unit measure.

Run once per region (downloads pose files the first time, stored in ibl_cache/):
    python 12_pose_control.py
    ANALYSIS_REGION=MOs python 12_pose_control.py
    ANALYSIS_REGION=VPM python 12_pose_control.py
The last run prints the comparison across all regions processed so far.

Outputs: processed/<pid>/pose.parquet, results/pose_sessions.csv, results/summary_12_pose.txt,
         figures/step12_pose_control.(png|pdf), results_comparison/summary_12_pose_regions.txt
"""
import argparse
import importlib
import json
import logging

import numpy as np
import pandas as pd
from scipy import stats

import common as K
import config as C

glm = importlib.import_module('03_single_unit_glm')
fu = importlib.import_module('09_followup')
log = logging.getLogger('12')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logging.getLogger('fontTools').setLevel(logging.WARNING)

TRACKERS = ('lightningPose', 'dlc')
PARTS = ('paw_l', 'paw_r', 'nose_tip')
WIN = C.WINDOWS['premove']
MIN_FINITE = 0.7          # fraction of trials with pose features required for an insertion
N_SUB = 5
N_ROUNDS_IMP = 5
N_BOOT = 2000


# ----------------------------------------------------------------------------
# pose extraction
# ----------------------------------------------------------------------------
def window_nanmean(times, values, ev, t0, t1):
    fin = np.isfinite(values)
    cs = np.concatenate([[0], np.cumsum(np.where(fin, values, 0))])
    cn = np.concatenate([[0], np.cumsum(fin)])
    a = np.searchsorted(times, ev + t0)
    b = np.searchsorted(times, ev + t1)
    n = cn[b] - cn[a]
    out = np.full(len(ev), np.nan)
    ok = n > 0
    out[ok] = (cs[b[ok]] - cs[a[ok]]) / n[ok]
    return out


def extract_pose(one, eid, trials):
    from brainbox.io.one import SessionLoader
    sl = SessionLoader(one=one, eid=eid)
    df, used = None, None
    for tracker in TRACKERS:
        try:
            sl.load_pose(views=['left'], tracker=tracker, likelihood_thr=0.9)
            df, used = sl.pose['leftCamera'], tracker
            break
        except Exception as e:
            log.info('  %s not available: %r', tracker, e)
    if df is None:
        return None, None
    times = df['times'].to_numpy(float)
    ev = trials['firstMovement_times'].to_numpy(float)
    feats = {}
    for part in PARTS:
        cx, cy = f'{part}_x', f'{part}_y'
        if cx not in df or cy not in df:
            continue
        for ax_, col in (('x', cx), ('y', cy)):
            pos = df[col].to_numpy(float)
            vel = np.gradient(pos, times)
            feats[f'{part}_{ax_}'] = window_nanmean(times, pos, ev, *WIN[1:])
            feats[f'{part}_v{ax_}'] = window_nanmean(times, vel, ev, *WIN[1:])
    if not feats:
        log.warning('  no paw/nose columns found in %s pose; available: %s', used, list(df.columns)[:30])
    return pd.DataFrame(feats), used


def download_all(one):
    paths = K.list_insertions()
    for k, path in enumerate(paths):
        out = path / 'pose.parquet'
        if out.exists():
            continue
        meta = json.loads((path / 'meta.json').read_text())
        trials = pd.read_parquet(path / 'trials.parquet')
        log.info('[%d/%d] pose for %s %s', k + 1, len(paths), meta['subject'], meta['eid'][:8])
        try:
            P, used = extract_pose(one, meta['eid'], trials)
        except Exception as e:
            log.warning('  failed: %r', e)
            P, used = None, None
        if P is None or P.empty:
            pd.DataFrame().to_parquet(out)          # mark as unavailable
            continue
        P.attrs = {}
        P.to_parquet(out)
        (path / 'pose_tracker.txt').write_text(used)


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------
def load_pose_sessions():
    sess = []
    for path in K.list_insertions():
        f = path / 'pose.parquet'
        if not f.exists():
            continue
        P = pd.read_parquet(f)
        if P.empty:
            continue
        meta, t, u, b = K.load_insertion(path)
        ok = K.valid_trials(t)
        Pv = P.loc[ok].reset_index(drop=True)
        fin = Pv.notna().mean(1) >= 0.5
        if fin.mean() < MIN_FINITE or len(u) < C.MIN_UNITS_POPULATION:
            continue
        keep = fin.to_numpy()
        t = t[ok].reset_index(drop=True)[keep].reset_index(drop=True)
        Pv = Pv[keep].reset_index(drop=True)
        Pv = Pv.fillna(Pv.mean())
        Pv = Pv.loc[:, Pv.std() > 0]
        Z = ((Pv - Pv.mean()) / Pv.std()).to_numpy()
        X = np.sqrt(K.window_counts(b, 'premove')[:, ok][:, keep].T)
        tr = (path / 'pose_tracker.txt').read_text() if (path / 'pose_tracker.txt').exists() else '?'
        sess.append(dict(pid=meta['pid'], subject=meta['subject'], t=t, X=X, P=Z, tracker=tr))
    return sess


def residualise(X, P):
    A = np.column_stack([np.ones(len(P)), P])
    beta, *_ = np.linalg.lstsq(A, X, rcond=None)
    return X - A @ beta


def dec(X, t_beh, n, rng, n_sub):
    y = t_beh['choice'].to_numpy()[:n]
    return fu.matched_acc(X[:n], y, fu.strata(t_beh.iloc[:n]), rng, n_sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-download', action='store_true')
    args = ap.parse_args()
    rng = np.random.default_rng(C.RANDOM_SEED + 12)
    plt = K.setup_matplotlib()
    if not args.no_download:
        download_all(C.get_one())
    sess = load_pose_sessions()
    n_all = len(K.list_insertions())
    print(f'{C.REGION}: {len(sess)} insertions with usable pose (of {n_all})')
    if len(sess) < 5:
        print('too few insertions with pose data')
        return
    for s in sess:
        s['Xr'] = residualise(s['X'], s['P'])

    # GLM base vs + pose
    glm_cnt = {'base': [0, 0], 'pose': [0, 0], 'pose_term': [0, 0]}
    for s in sess:
        Xd, names, idx = K.design_matrix(s['t'], 'premove')
        Y = s['X']
        F, *_r = glm.nested_F(Xd, Y, idx['choice'])
        p = stats.f.sf(F, 1, Xd.shape[0] - Xd.shape[1])
        glm_cnt['base'][0] += int(np.nansum(p < C.ALPHA))
        glm_cnt['base'][1] += Y.shape[1]
        Xp = np.column_stack([Xd, s['P']])
        F, *_r = glm.nested_F(Xp, Y, idx['choice'])
        p = stats.f.sf(F, 1, Xp.shape[0] - Xp.shape[1])
        glm_cnt['pose'][0] += int(np.nansum(p < C.ALPHA))
        glm_cnt['pose'][1] += Y.shape[1]
        pc = list(range(Xd.shape[1], Xp.shape[1]))
        F, *_r = glm.nested_F(Xp, Y, pc)
        p = stats.f.sf(F, len(pc), Xp.shape[0] - Xp.shape[1])
        glm_cnt['pose_term'][0] += int(np.nansum(p < C.ALPHA))
        glm_cnt['pose_term'][1] += Y.shape[1]

    # decoding: neural original / residualised / pose-only, each minus imposter chance
    rows = []
    for s in sess:
        rows.append(dict(pid=s['pid'], subject=s['subject'], tracker=s['tracker'], n_units=s['X'].shape[1],
                         n_trials=len(s['t']), n_pose_features=s['P'].shape[1],
                         acc_neural=dec(s['X'], s['t'], len(s['t']), rng, N_SUB),
                         acc_resid=dec(s['Xr'], s['t'], len(s['t']), rng, N_SUB),
                         acc_pose=dec(s['P'], s['t'], len(s['t']), rng, N_SUB)))
    df = pd.DataFrame(rows)
    imp = {k: np.zeros((N_ROUNDS_IMP, len(sess))) for k in ('neural', 'resid', 'pose')}
    for r in range(N_ROUNDS_IMP):
        perm = fu.derangement(len(sess), rng)
        for i, s in enumerate(sess):
            tb = sess[perm[i]]['t']
            n = min(len(s['t']), len(tb))
            imp['neural'][r, i] = dec(s['X'], tb, n, rng, 1)
            imp['resid'][r, i] = dec(s['Xr'], tb, n, rng, 1)
            # pose belongs to behaviour: pair session i's pose with session j's labels
            imp['pose'][r, i] = dec(s['P'], tb, n, rng, 1)
        log.info('imposter round %d/%d', r + 1, N_ROUNDS_IMP)
    for k in imp:
        df[f'chance_{k}'] = imp[k].mean(0)
        df[f'above_{k}'] = df[f'acc_{k}'] - df[f'chance_{k}']
    df.to_csv(C.RESULTS_DIR / 'pose_sessions.csv', index=False)

    def ci(x):
        x = np.asarray(x, float)
        b = [rng.choice(x, len(x)).mean() for _ in range(N_BOOT)]
        return np.percentile(b, [2.5, 97.5])

    ret = df['above_resid'].mean() / df['above_neural'].mean() if df['above_neural'].mean() > 0 else np.nan
    bret = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(df), len(df))
        a, b_ = df['above_resid'].iloc[i].mean(), df['above_neural'].iloc[i].mean()
        bret.append(a / b_ if b_ > 0 else np.nan)
    L = [f'=== Pose control ({C.REGION}; {len(df)} insertions; trackers: {df["tracker"].value_counts().to_dict()}) ===',
         f"pose features per insertion: median {df['n_pose_features'].median():.0f} (paw/nose x, y, vx, vy in premove window)"]
    for k, lab in (('pose', 'pose features alone'), ('neural', 'neural (original)'), ('resid', 'neural, pose regressed out')):
        x = df[f'above_{k}']
        lo, hi = ci(x)
        L.append(f'{lab:28s}: {x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] above imposter chance; '
                 f'Wilcoxon p={stats.wilcoxon(x, alternative="greater").pvalue:.3g}')
    L.append(f'retention (resid / original): {ret:.2f} (bootstrap 95% CI {np.nanpercentile(bret, 2.5):.2f}-{np.nanpercentile(bret, 97.5):.2f})')
    L.append(f"GLM choice units: base {glm_cnt['base'][0] / glm_cnt['base'][1]:.1%} -> + pose {glm_cnt['pose'][0] / glm_cnt['pose'][1]:.1%}; "
             f"units with pose effects {glm_cnt['pose_term'][0] / glm_cnt['pose_term'][1]:.1%} [parametric p<0.05]")
    C.write_summary('summary_12_pose', L)
    pd.DataFrame([dict(region=C.REGION, n=len(df), retention=ret, ret_lo=np.nanpercentile(bret, 2.5),
                       ret_hi=np.nanpercentile(bret, 97.5), glm_base=glm_cnt['base'][0] / glm_cnt['base'][1],
                       glm_pose=glm_cnt['pose'][0] / glm_cnt['pose'][1],
                       **{f'above_{k}': df[f'above_{k}'].mean() for k in ('pose', 'neural', 'resid')})]) \
        .to_csv(C.RESULTS_DIR / 'pose_region_summary.csv', index=False)

    fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.4))
    fig.subplots_adjust(wspace=0.5)
    ax = axs[0]
    for _, r in df.iterrows():
        ax.plot([0, 1, 2], [r['above_pose'], r['above_neural'], r['above_resid']], color='0.8', lw=0.5)
    ax.plot([0, 1, 2], [df['above_pose'].mean(), df['above_neural'].mean(), df['above_resid'].mean()],
            color=K.COL_CHOICE, marker='o', lw=1.5)
    ax.axhline(0, color='k', ls=':', lw=0.8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['pose\nonly', 'neural', 'neural\n- pose'])
    ax.set(ylabel='Direction decoding\n(accuracy - imposter chance)', title=f'A  {C.REGION}, premove')
    ax = axs[1]
    ax.bar([0, 1], [glm_cnt['base'][0] / glm_cnt['base'][1], glm_cnt['pose'][0] / glm_cnt['pose'][1]],
           color=['0.6', K.COL_CHOICE], width=0.6)
    ax.axhline(C.ALPHA, color='k', ls=':', lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['base GLM', '+ pose'])
    ax.set(ylabel='Choice units (p<0.05)', title='B  GLM')
    C.save_fig(fig, 'step12_pose_control')

    # comparison across regions
    out = C.PROJECT_DIR / 'results_comparison'
    out.mkdir(exist_ok=True)
    regs = []
    for d in [C.PROJECT_DIR / 'results'] + sorted(C.PROJECT_DIR.glob('results_*')):
        f = d / 'pose_region_summary.csv'
        g = d / 'pose_sessions.csv'
        if f.exists() and g.exists():
            regs.append((pd.read_csv(f).iloc[0], pd.read_csv(g)))
    if len(regs) >= 2:
        L2 = ['=== Pose control across regions (premove direction decoding, minus imposter chance) ===']
        boots = {}
        for r, g in regs:
            L2.append(f"{r['region']:5s}: pose-only {r['above_pose']:+.3f} | neural {r['above_neural']:+.3f} -> "
                      f"pose regressed out {r['above_resid']:+.3f}; retention {r['retention']:.2f} "
                      f"[{r['ret_lo']:.2f}, {r['ret_hi']:.2f}]; GLM choice units {r['glm_base']:.1%} -> {r['glm_pose']:.1%} ({int(r['n'])} insertions)")
            b = []
            for _ in range(N_BOOT):
                i = rng.integers(0, len(g), len(g))
                d_ = g['above_neural'].iloc[i].mean()
                b.append(g['above_resid'].iloc[i].mean() / d_ if d_ > 0 else np.nan)
            boots[r['region']] = np.array(b)
        if 'ZI' in boots:
            for reg, b in boots.items():
                if reg == 'ZI':
                    continue
                d = boots['ZI'] - b
                d = d[np.isfinite(d)]
                p = 2 * min((d >= 0).mean(), (d <= 0).mean())
                L2.append(f'retention ZI - {reg}: {np.median(d):+.2f} (95% CI {np.percentile(d, 2.5):+.2f} to '
                          f'{np.percentile(d, 97.5):+.2f}; bootstrap p={max(p, 1 / len(d)):.3g})')
        text = '\n'.join(L2)
        (out / 'summary_12_pose_regions.txt').write_text(text, encoding='utf-8')
        print(text)


if __name__ == '__main__':
    main()
