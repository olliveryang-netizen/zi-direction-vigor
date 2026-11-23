"""
Step 13 - Which comes first: direction information in neurons or in the body? (paper Figures 4C-F and S4)

Paw / nose kinematics can predict the upcoming direction before the wheel moves (step 12). Here we
resolve the TIMING:
in sliding windows (60 ms wide, 20 ms steps, -300..+200 ms from wheel-movement onset) we decode the
upcoming direction from
   (1) neural activity,
   (2) pose features alone (x, y, vx, vy of both forepaws and the nose tip, left camera),
   (3) neural activity with the pose features of the SAME window regressed out,
all with trials matched for wheel speed and block prior, minus imposter-session chance.

Latencies: half-maximum latency of each curve and first significant window (Wilcoxon across
insertions, FDR over windows, 2 consecutive windows). Neural vs pose latency is compared with a
PAIRED bootstrap over insertions (same insertions resampled for both curves).
   neural earlier than pose  -> direction signal precedes the body movements it could reflect
   pose earlier than neural  -> neural signal could follow body movement

Requires step 12 (pose files) for each region. Run once:   python 13_pose_timecourse.py
Outputs: processed*/<pid>/pose_tc.npz, results_comparison/pose_timecourse.csv,
         results_comparison/summary_13_pose_timecourse.txt, figures_comparison/step13_pose_timecourse.(png|pdf)
"""
import importlib
import json
import logging

import numpy as np
import pandas as pd
from scipy import stats

import common as K
import config as C

fu = importlib.import_module('09_followup')
rev = importlib.import_module('10_reversal_history')
pc = importlib.import_module('12_pose_control')
log = logging.getLogger('13')
logging.getLogger('fontTools').setLevel(logging.WARNING)

WIDTH, STEP, RANGE = 0.06, 0.02, (-0.30, 0.20)
STARTS = np.round(np.arange(RANGE[0], RANGE[1] - WIDTH + 1e-9, STEP), 3)
CENTERS = STARTS + WIDTH / 2
N_SUB = 3
N_ROUNDS_IMP = 5
N_BOOT = 2000
OUT_R = C.PROJECT_DIR / 'results_comparison'
OUT_F = C.PROJECT_DIR / 'figures_comparison'


def region_processed():
    out = {'ZI': C.PROJECT_DIR / 'processed'}
    for p in sorted(C.PROJECT_DIR.glob('processed_*')):
        out[p.name.split('_', 1)[1]] = p
    return {r: p for r, p in out.items() if p.exists()}


def pose_timecourse(one, eid, trials, tracker):
    """features x trials x windows (pose aligned to wheel-movement onset)."""
    from brainbox.io.one import SessionLoader
    sl = SessionLoader(one=one, eid=eid)
    sl.load_pose(views=['left'], tracker=tracker, likelihood_thr=0.9)
    df = sl.pose['leftCamera']
    times = df['times'].to_numpy(float)
    ev = trials['firstMovement_times'].to_numpy(float)
    feats, names = [], []
    for part in pc.PARTS:
        for ax_ in ('x', 'y'):
            col = f'{part}_{ax_}'
            if col not in df:
                continue
            pos = df[col].to_numpy(float)
            vel = np.gradient(pos, times)
            for kind, sig in (('', pos), ('v', vel)):
                feats.append(np.stack([pc.window_nanmean(times, sig, ev, s, s + WIDTH) for s in STARTS], 1))
                names.append(f'{part}_{kind}{ax_}')
    return np.stack(feats), names


def ensure_pose_tc(one, pdir):
    for path in sorted(q for q in pdir.iterdir() if (q / 'meta.json').exists()):
        out = path / 'pose_tc.npz'
        f = path / 'pose.parquet'
        if out.exists() or not f.exists() or pd.read_parquet(f).empty:
            continue
        meta = json.loads((path / 'meta.json').read_text())
        tracker = (path / 'pose_tracker.txt').read_text() if (path / 'pose_tracker.txt').exists() else 'lightningPose'
        try:
            F, names = pose_timecourse(one, meta['eid'], pd.read_parquet(path / 'trials.parquet'), tracker)
            np.savez_compressed(out, F=F.astype(np.float32), names=np.array(names))
            log.info('pose time course saved for %s', meta['pid'][:8])
        except Exception as e:
            log.warning('pose time course failed for %s: %r', meta['pid'][:8], e)


def load_region(pdir):
    sess = []
    for path in sorted(q for q in pdir.iterdir() if (q / 'pose_tc.npz').exists()):
        meta = json.loads((path / 'meta.json').read_text())
        _, t, u, b = K.load_insertion(path)
        if len(u) < C.MIN_UNITS_POPULATION:
            continue
        ok = K.valid_trials(t)
        F = np.load(path / 'pose_tc.npz')['F'][:, ok, :]                       # feat x trials x windows
        if np.mean(np.isfinite(F)) < pc.MIN_FINITE:
            continue
        t = t[ok].reset_index(drop=True)
        Xw, Pw = [], []
        for j, s in enumerate(STARTS):
            Xw.append(np.sqrt(K.window_counts(b, ('firstMove', s, s + WIDTH))[:, ok].T))
            P = F[:, :, j].T.copy()
            good = np.mean(np.isfinite(P), 0) >= 0.5
            P = P[:, good]
            P = np.where(np.isfinite(P), P, np.nanmean(P, 0))
            sd = P.std(0)
            Pw.append((P[:, sd > 0] - P[:, sd > 0].mean(0)) / sd[sd > 0])
        sess.append(dict(pid=meta['pid'], subject=meta['subject'], t=t, X=Xw, P=Pw,
                         R=[pc.residualise(x, p) if p.shape[1] else x for x, p in zip(Xw, Pw)]))
    return sess


def curves(sess, rng):
    kinds = ('neural', 'pose', 'resid')
    real = {k: np.zeros((len(sess), len(STARTS))) for k in kinds}
    imp = {k: np.zeros((len(sess), len(STARTS))) for k in kinds}
    src = {'neural': 'X', 'pose': 'P', 'resid': 'R'}
    for i, s in enumerate(sess):
        for k in kinds:
            real[k][i] = [pc.dec(m, s['t'], len(s['t']), rng, N_SUB) if m.shape[1] else np.nan for m in s[src[k]]]
    for r in range(N_ROUNDS_IMP):
        perm = fu.derangement(len(sess), rng)
        for i, s in enumerate(sess):
            tb = sess[perm[i]]['t']
            n = min(len(s['t']), len(tb))
            for k in kinds:
                imp[k][i] += np.array([pc.dec(m, tb, n, rng, 1) if m.shape[1] else np.nan for m in s[src[k]]]) / N_ROUNDS_IMP
        print(f'    imposter round {r + 1}/{N_ROUNDS_IMP}')
    return {k: real[k] - imp[k] for k in kinds}


def main():
    OUT_R.mkdir(exist_ok=True)
    OUT_F.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    rng = np.random.default_rng(C.RANDOM_SEED + 13)
    plt = K.setup_matplotlib()
    regions = region_processed()
    one = None
    res, rows = {}, []
    for reg, pdir in regions.items():
        need = [q for q in pdir.iterdir() if (q / 'pose.parquet').exists() and not (q / 'pose_tc.npz').exists()]
        if need:
            one = one or C.get_one()
            ensure_pose_tc(one, pdir)
        sess = load_region(pdir)
        if len(sess) < 5:
            continue
        print(f'{reg}: {len(sess)} insertions')
        res[reg] = curves(sess, rng)
        for k, m in res[reg].items():
            for j, c in enumerate(CENTERS):
                rows.append(dict(region=reg, kind=k, t_center_ms=c * 1000, mean=np.nanmean(m[:, j]),
                                 sem=K.sem(m[:, j])))
    pd.DataFrame(rows).to_csv(OUT_R / 'pose_timecourse.csv', index=False)
    if not res:
        print('No region with pose time courses. Run 12_pose_control.py first.')
        return

    cms = CENTERS * 1000
    L = ['=== Timing of direction information: neural vs body (pose), relative to wheel-movement onset ===',
         f'windows {WIDTH * 1000:.0f} ms, step {STEP * 1000:.0f} ms; values are accuracy minus imposter chance']
    lat_rows = []
    for reg, m in res.items():
        L.append(f'--- {reg} ({len(m["neural"])} insertions)')
        lat = {}
        for k, lab in (('neural', 'neural'), ('pose', 'pose only'), ('resid', 'neural - pose')):
            hm = rev.halfmax_latency(cms, np.nanmean(m[k], 0))
            on, _q = rev.fdr_onset(cms, np.nan_to_num(m[k]))
            lat[k] = hm
            L.append(f'  {lab:14s}: half-max {hm:.0f} ms; first significant window {on:.0f} ms; '
                     f'value at {cms[0]:.0f} ms {np.nanmean(m[k][:, 0]):+.3f}, at {cms[np.argmin(np.abs(cms + 50))]:.0f} ms '
                     f'{np.nanmean(m[k][:, np.argmin(np.abs(cms + 50))]):+.3f}')
        bs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(m['neural']), len(m['neural']))
            bs.append(rev.halfmax_latency(cms, np.nanmean(m['neural'][i], 0)) - rev.halfmax_latency(cms, np.nanmean(m['pose'][i], 0)))
        bs = np.array(bs)
        bs = bs[np.isfinite(bs)]
        p = 2 * min((bs >= 0).mean(), (bs <= 0).mean())
        L.append(f'  neural - pose half-max latency: {np.median(bs):+.0f} ms (paired bootstrap 95% CI '
                 f'{np.percentile(bs, 2.5):+.0f} to {np.percentile(bs, 97.5):+.0f}; p={max(p, 1 / len(bs)):.3g}); '
                 f'negative = neural earlier')
        lat_rows.append(dict(region=reg, neural=lat['neural'], pose=lat['pose'], resid=lat['resid'],
                             diff=np.median(bs), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    text = '\n'.join(L)
    (OUT_R / 'summary_13_pose_timecourse.txt').write_text(text, encoding='utf-8')
    print(text)

    n = len(res)
    fig, axs = plt.subplots(n, 3, figsize=(7.2, 1.9 * n + 0.4), squeeze=False)
    fig.subplots_adjust(hspace=0.75, wspace=0.5)
    sty = {'neural': (K.COL_CHOICE, '-', 'neural'), 'pose': ('#8a5a44', '-', 'pose only'),
           'resid': (K.COL_CHOICE, '--', 'neural - pose')}
    for row, (reg, m) in enumerate(res.items()):
        for k, (col, ls, lab) in sty.items():
            mu = np.nanmean(m[k], 0)
            se = np.nanstd(m[k], 0, ddof=1) / np.sqrt(np.sum(np.isfinite(m[k]), 0))
            axs[row, 0].plot(cms, mu, color=col, ls=ls, lw=1.2, label=lab)
            axs[row, 0].fill_between(cms, mu - se, mu + se, color=col, alpha=0.15, lw=0)
            if k != 'resid':
                axs[row, 1].plot(cms, mu / np.nanmax(mu), color=col, lw=1.2, label=lab)
        for a in axs[row, :2]:
            a.axvline(0, color='k', lw=0.6)
            a.axhline(0, color='k', ls=':', lw=0.6)
            a.set_xlabel('Time from wheel movement (ms)')
        axs[row, 0].set(ylabel='Accuracy - chance', title=f'{reg}: direction information')
        axs[row, 1].set(ylabel='Normalised', title=f'{reg}: timing (normalised)')
        if row == 0:
            axs[row, 0].legend(frameon=False, fontsize=5.5)
            axs[row, 1].legend(frameon=False, fontsize=5.5)
        axs[row, 2].axis('off')
    ax = axs[0, 2]
    ax.axis('on')
    lr = pd.DataFrame(lat_rows)
    ax.errorbar(np.arange(len(lr)), lr['diff'], yerr=[lr['diff'] - lr['lo'], lr['hi'] - lr['diff']], fmt='o', ms=3,
                color=K.COL_CHOICE, capsize=2)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(np.arange(len(lr)))
    ax.set_xticklabels(lr['region'])
    ax.set(ylabel='Neural - pose latency (ms)', title='Neural earlier (<0)?')
    fig.savefig(OUT_F / 'step13_pose_timecourse.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_F / 'step13_pose_timecourse.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
