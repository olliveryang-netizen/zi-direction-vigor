"""
Step 15 - Additional statistical checks (direct between-region tests, clustering, cross-validation).

Uses existing processed data and results only. Run after steps 01-13 exist for ZI, MOs and VPM:

    python 15_revision_checks.py              # all checks
    python 15_revision_checks.py --skip-pose  # skip A (the slowest part)

A  Neural - pose latency compared BETWEEN regions (ZI vs VPM, ZI vs MOs, MOs vs VPM).
   The per-insertion time courses of step 13 are recomputed with the same random seed and region order
   (so the within-region values reproduce summary_13), saved to
   results_comparison/pose_timecourse_insertions.npz, and the difference of (neural - pose) half-max
   latencies between two regions is bootstrapped by resampling insertions within each region.
B  Choice-independent vigor R2 (100-300 ms after movement onset) compared between regions:
   Mann-Whitney across insertions, bootstrap CI of the difference, and the same at the level of mice.
C  Unit fractions compared between regions with a cluster bootstrap over MICE (units of a mouse are
   resampled together), and with Mann-Whitney tests on per-mouse fractions. Replaces the unit-level
   Fisher test, which treats units as independent.
D  Overlap of direction and speed coding within ZI (pre-movement window): permutation test that shuffles
   speed labels among the units of the same insertion (keeps each insertion's rate of speed units).
E  Speed-matched direction decoding with temporally BLOCKED cross-validation (contiguous blocks of trials
   as test folds) compared with the standard stratified random folds, for real and imposter pairings.
F  Insertions used by step 12 but not by step 13, with the reason.

Output: results_comparison/summary_15_revision_checks.txt
"""
import argparse
import importlib
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score

import common as K
import config as C

warnings.filterwarnings('ignore')
dec = importlib.import_module('04_decoding')
fu = importlib.import_module('09_followup')
rev = importlib.import_module('10_reversal_history')

REGIONS = ('ZI', 'MOs', 'VPM')
OUT_R = C.PROJECT_DIR / 'results_comparison'
N_BOOT = 10000
N_SUB_REAL = 20
N_ROUNDS_IMP = 20
N_SUB_IMP = 5


def results_dir(reg):
    return C.PROJECT_DIR / ('results' if reg == 'ZI' else f'results_{reg}')


def processed_dir(reg):
    return C.PROJECT_DIR / ('processed' if reg == 'ZI' else f'processed_{reg}')


def p_two_sided(bs):
    bs = np.asarray(bs, float)
    bs = bs[np.isfinite(bs)]
    return max(2 * min((bs >= 0).mean(), (bs <= 0).mean()), 1 / len(bs))


def ci(bs):
    bs = np.asarray(bs, float)
    bs = bs[np.isfinite(bs)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


# ----------------------------------------------------------------------------
# A  neural - pose latency between regions
# ----------------------------------------------------------------------------
def pose_curves():
    """Per-insertion curves of step 13, reproduced with the same seed and order (cached)."""
    f = OUT_R / 'pose_timecourse_insertions.npz'
    pt = importlib.import_module('13_pose_timecourse')
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return {k: z[k].item() for k in z.files}, pt
    rng = np.random.default_rng(C.RANDOM_SEED + 13)
    res = {}
    for reg, pdir in pt.region_processed().items():
        sess = pt.load_region(pdir)
        if len(sess) < 5:
            continue
        print(f'  {reg}: {len(sess)} insertions (recomputing time courses)')
        res[reg] = pt.curves(sess, rng)
        res[reg]['pid'] = np.array([s['pid'] for s in sess])
    # the within-region paired bootstrap of step 13, continued with the same generator
    cms = pt.CENTERS * 1000
    for reg, m in res.items():
        bs = []
        for _ in range(pt.N_BOOT):
            i = rng.integers(0, len(m['neural']), len(m['neural']))
            bs.append(rev.halfmax_latency(cms, np.nanmean(m['neural'][i], 0))
                      - rev.halfmax_latency(cms, np.nanmean(m['pose'][i], 0)))
        m['paired_boot'] = np.array(bs)
    np.savez(f, **{reg: m for reg, m in res.items()})
    return res, pt


def check_A(L):
    res, pt = pose_curves()
    cms = pt.CENTERS * 1000
    L.append('--- A  Neural - pose half-max latency, compared between regions')

    def lag(m, i=None):
        n = m['neural'] if i is None else m['neural'][i]
        p = m['pose'] if i is None else m['pose'][i]
        return rev.halfmax_latency(cms, np.nanmean(n, 0)) - rev.halfmax_latency(cms, np.nanmean(p, 0))

    for reg in REGIONS:
        if reg in res:
            bs = res[reg]['paired_boot']
            lo, hi = ci(bs)
            L.append(f'  {reg:4s}: {np.nanmedian(bs):+.0f} ms (95% CI {lo:+.0f} to {hi:+.0f}; '
                     f'p={p_two_sided(bs):.3g}; {len(res[reg]["neural"])} insertions)  [should match summary_13]')
    rng = np.random.default_rng(C.RANDOM_SEED + 1501)
    for a, b in (('ZI', 'VPM'), ('ZI', 'MOs'), ('MOs', 'VPM')):
        if a not in res or b not in res:
            continue
        ma, mb = res[a], res[b]
        na, nb = len(ma['neural']), len(mb['neural'])
        bs = np.array([lag(ma, rng.integers(0, na, na)) - lag(mb, rng.integers(0, nb, nb)) for _ in range(N_BOOT)])
        lo, hi = ci(bs)
        L.append(f'  {a} - {b}: {lag(ma) - lag(mb):+.0f} ms (bootstrap 95% CI {lo:+.0f} to {hi:+.0f}; '
                 f'p={p_two_sided(bs):.3g})')


# ----------------------------------------------------------------------------
# B  vigor R2 between regions
# ----------------------------------------------------------------------------
def check_B(L):
    L.append('--- B  Choice-independent vigor R2 (mean over windows centred 100-300 ms), between regions')
    per, per_mouse = {}, {}
    for reg in REGIONS:
        f = results_dir(reg) / 'decoding_timecourse.csv'
        if not f.exists():
            continue
        tc = pd.read_csv(f)
        w = tc[(tc['t_center'] >= 0.10) & (tc['t_center'] <= 0.30)]
        s = w.groupby('pid')['r2_speed_resid'].mean().dropna()
        per[reg] = s.to_numpy()
        per_mouse[reg] = w.groupby(['subject', 'pid'])['r2_speed_resid'].mean().groupby('subject').mean().to_numpy()
        L.append(f'  {reg:4s}: {s.mean():+.3f} ({len(s)} insertions, {len(per_mouse[reg])} mice)')
    rng = np.random.default_rng(C.RANDOM_SEED + 1502)
    for a, b in (('ZI', 'MOs'), ('ZI', 'VPM')):
        if a not in per or b not in per:
            continue
        x, y = per[a], per[b]
        bs = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(N_BOOT)]
        lo, hi = ci(bs)
        p_mw = stats.mannwhitneyu(x, y).pvalue
        p_mm = stats.mannwhitneyu(per_mouse[a], per_mouse[b]).pvalue
        L.append(f'  {a} - {b}: {x.mean() - y.mean():+.3f} (bootstrap 95% CI {lo:+.3f} to {hi:+.3f}); '
                 f'Mann-Whitney p={p_mw:.3g} (insertions), p={p_mm:.3g} (mice)')


# ----------------------------------------------------------------------------
# C  unit fractions between regions, clustered by mouse
# ----------------------------------------------------------------------------
def sig_units(reg, window, term):
    g = pd.read_csv(results_dir(reg) / 'glm_units.csv')
    g = g[(g['window'] == window) & (g['term'] == term)]
    return pd.DataFrame({'subject': g['subject'].to_numpy(), 'sig': (g['p_perm'] < C.ALPHA).to_numpy()})


def mouse_boot_fraction(df, rng):
    by = {s: v['sig'].to_numpy() for s, v in df.groupby('subject')}
    mice = list(by)
    out = np.empty(N_BOOT)
    for k in range(N_BOOT):
        pick = rng.integers(0, len(mice), len(mice))
        v = np.concatenate([by[mice[i]] for i in pick])
        out[k] = v.mean()
    return out


def check_C(L):
    L.append('--- C  Unit fractions between regions: cluster bootstrap over mice, and per-mouse Mann-Whitney')
    rng = np.random.default_rng(C.RANDOM_SEED + 1503)
    for label, window, term in (('direction units (pre-movement)', 'premove', 'choice'),
                                ('speed units (movement)', 'move', 'speed')):
        d = {r: sig_units(r, window, term) for r in REGIONS if (results_dir(r) / 'glm_units.csv').exists()}
        boots = {r: mouse_boot_fraction(v, rng) for r, v in d.items()}
        L.append(f'  {label}: ' + ', '.join(f'{r} {v["sig"].mean():.1%} ({v["subject"].nunique()} mice)'
                                           for r, v in d.items()))
        for a, b in (('ZI', 'MOs'), ('ZI', 'VPM')):
            if a not in d or b not in d:
                continue
            diff = d[a]['sig'].mean() - d[b]['sig'].mean()
            bs = boots[a] - boots[b]
            lo, hi = ci(bs)
            fa = d[a].groupby('subject')['sig'].mean()
            fb = d[b].groupby('subject')['sig'].mean()
            p_fisher = stats.fisher_exact([[d[a]['sig'].sum(), (~d[a]['sig']).sum()],
                                           [d[b]['sig'].sum(), (~d[b]['sig']).sum()]])[1]
            L.append(f'    {a} - {b}: {diff * 100:+.1f} points (mouse-bootstrap 95% CI {lo * 100:+.1f} to '
                     f'{hi * 100:+.1f}; p={p_two_sided(bs):.3g}); per-mouse Mann-Whitney p='
                     f'{stats.mannwhitneyu(fa, fb).pvalue:.3g}; unit-level Fisher p={p_fisher:.3g} (for reference)')


# ----------------------------------------------------------------------------
# D  overlap of direction and speed coding within ZI
# ----------------------------------------------------------------------------
def check_D(L):
    g = pd.read_csv(results_dir('ZI') / 'glm_units.csv')
    g = g[g['window'] == 'premove']
    w = g.pivot_table(index=['pid', 'unit_index'], columns='term', values='p_perm').reset_index()
    ch = (w['choice'] < C.ALPHA).to_numpy()
    sp = (w['speed'] < C.ALPHA).to_numpy()
    pid = w['pid'].to_numpy()
    obs = int((ch & sp).sum())
    rng = np.random.default_rng(C.RANDOM_SEED + 1504)
    groups = [np.where(pid == p)[0] for p in np.unique(pid)]
    null = np.empty(N_BOOT, int)
    for k in range(N_BOOT):
        sp2 = sp.copy()
        for i in groups:
            sp2[i] = sp[rng.permutation(i)]
        null[k] = (ch & sp2).sum()
    p = (1 + (null >= obs).sum()) / (1 + N_BOOT)
    L.append('--- D  ZI pre-movement overlap of direction and speed units (labels shuffled within insertions)')
    L.append(f'  both {obs} (direction {ch.sum()}, speed {sp.sum()}, units {len(ch)}); '
             f'expected {null.mean():.1f} (95% range {np.percentile(null, 2.5):.0f}-{np.percentile(null, 97.5):.0f}); '
             f'p={p:.3g} (one-sided)')


# ----------------------------------------------------------------------------
# E  blocked cross-validation
# ----------------------------------------------------------------------------
def cv_blocked(X, y):
    if min((y == 1).sum(), (y == -1).sum()) < dec.N_FOLD:
        return np.nan
    pred = np.empty_like(y)
    for te in np.array_split(np.arange(len(y)), dec.N_FOLD):
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(np.unique(y[tr])) < 2:
            return np.nan
        pred[te] = dec.clf().fit(X[tr], y[tr]).predict(X[te])
    return balanced_accuracy_score(y, pred)


def load_sessions(pdir):
    sess = []
    for path in sorted(q for q in pdir.iterdir() if (q / 'meta.json').exists()):
        meta, t, u, b = K.load_insertion(path)
        ok = K.valid_trials(t)
        if ok.sum() < 100:
            continue
        sess.append(dict(pid=meta['pid'], t=t[ok].reset_index(drop=True), n_units=len(u),
                         X=np.sqrt(K.window_counts(b, 'premove')[:, ok].T)))
    return sess


def pair_accuracy(s_neu, s_beh, n_sub, rng):
    n = min(len(s_neu['t']), len(s_beh['t']))
    t = s_beh['t'].iloc[:n]
    X = s_neu['X'][:n]
    y = t['choice'].to_numpy()
    sbin = pd.qcut(t['peak_speed'], C.N_SPEED_BINS, labels=False, duplicates='drop').to_numpy()
    a_rand, a_blk = [], []
    for _ in range(n_sub):
        idx = np.sort(dec.matched_idx(y, sbin, rng))                         # chronological order
        a_rand.append(dec.cv_balacc(X[idx], y[idx], rng))
        a_blk.append(cv_blocked(X[idx], y[idx]))
    return np.nanmean(a_rand), np.nanmean(a_blk)


def check_E(L):
    L.append('--- E  Speed-matched direction decoding: stratified random folds vs contiguous (blocked) folds')
    rng = np.random.default_rng(C.RANDOM_SEED + 1505)
    for reg in REGIONS:
        pdir = processed_dir(reg)
        if not pdir.exists():
            continue
        sess = load_sessions(pdir)
        use = [i for i, s in enumerate(sess) if s['n_units'] >= C.MIN_UNITS_POPULATION]
        real = np.array([pair_accuracy(sess[i], sess[i], N_SUB_REAL, rng) for i in use])
        imp = np.zeros_like(real)
        for r in range(N_ROUNDS_IMP):
            perm = fu.derangement(len(sess), rng)
            imp += np.array([pair_accuracy(sess[i], sess[perm[i]], N_SUB_IMP, rng) for i in use]) / N_ROUNDS_IMP
        d = real - imp
        line = f'  {reg:4s} ({len(use)} insertions): '
        for j, lab in ((0, 'random folds'), (1, 'blocked folds')):
            p = stats.wilcoxon(d[:, j], alternative='greater').pvalue
            line += f'{lab} real {np.nanmean(real[:, j]):.3f}, imposter {np.nanmean(imp[:, j]):.3f}, ' \
                    f'difference {np.nanmean(d[:, j]):+.3f} (Wilcoxon p={p:.3g}); '
        L.append(line.rstrip('; '))
        print(f'  E done for {reg}')


# ----------------------------------------------------------------------------
# F  insertions in step 12 but not in step 13
# ----------------------------------------------------------------------------
def check_F(L):
    L.append('--- F  Insertions used for the pose control (step 12) but not for the pose time course (step 13)')
    pc = importlib.import_module('12_pose_control')
    for reg in REGIONS:
        f = results_dir(reg) / 'pose_sessions.csv'
        if not f.exists():
            continue
        p12 = set(pd.read_csv(f)['pid'])
        for pid in sorted(p12):
            path = processed_dir(reg) / pid
            if not path.exists():
                cand = [q for q in processed_dir(reg).iterdir()
                        if (q / 'meta.json').exists() and json.loads((q / 'meta.json').read_text())['pid'] == pid]
                if not cand:
                    continue
                path = cand[0]
            tc = path / 'pose_tc.npz'
            if not tc.exists():
                L.append(f'  {reg}: {pid} - no pose time course file (pose could not be loaded for -300..+200 ms)')
                continue
            _, t, _, _ = K.load_insertion(path)
            F = np.load(tc)['F'][:, K.valid_trials(t), :]
            cov = np.mean(np.isfinite(F))
            if cov < pc.MIN_FINITE:
                L.append(f'  {reg}: {pid} - valid pose values {cov:.0%} over -300..+200 ms '
                         f'(criterion {pc.MIN_FINITE:.0%})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-pose', action='store_true', help='skip check A')
    args = ap.parse_args()
    OUT_R.mkdir(exist_ok=True)
    L = ['=== Additional statistical checks ===']
    steps = [('B', check_B), ('C', check_C), ('D', check_D), ('F', check_F), ('E', check_E)]
    if not args.skip_pose:
        steps.insert(0, ('A', check_A))
    for name, fn in steps:
        print(f'check {name} ...', flush=True)
        try:
            fn(L)
        except Exception as e:                                              # noqa: BLE001
            L.append(f'--- {name}  failed: {e!r}')
            print(f'  check {name} failed: {e!r}')
    text = '\n'.join(L)
    (OUT_R / 'summary_15_revision_checks.txt').write_text(text, encoding='utf-8')
    print('\n' + text)


if __name__ == '__main__':
    main()
