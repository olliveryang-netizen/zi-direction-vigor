"""
Step 11 - Unit-count-matched comparison of direction information across regions (paper Figure 6C).

Decoding accuracy grows with the number of units, and regions differ a lot in units per
insertion (ZI has far fewer units per insertion than MOs or VPM). Here every region is subsampled to the SAME number of units
(N = 3, 5, 8, 12) before decoding, so information per neuron can be compared fairly.

For each insertion with >= N units: R random subsets of N units; premove window (-160..0 ms);
choice decoded with trials matched for wheel speed AND block prior (as in step 09).
Chance level: imposter sessions within the same region, same N.
Statistic per insertion: accuracy - imposter chance. Regions compared with bootstrap CIs over
insertions and Mann-Whitney tests.

Run after steps 01-10 exist for at least two regions:   python 11_unit_matched.py
Outputs: results_comparison/unit_matched_decoding.csv, results_comparison/summary_11_unit_matched.txt,
         figures_comparison/step11_unit_matched.(png|pdf)
"""
import importlib
import json

import numpy as np
import pandas as pd
from scipy import stats

import common as K
import config as C

fu = importlib.import_module('09_followup')

N_LIST = (3, 5, 8, 12)
N_SUBSETS = 20                    # random unit subsets per insertion and N
N_SUB = 2                         # matched trial resamples per subset
N_ROUNDS_IMP = 5                  # imposter pairings per region and N
N_SUBSETS_IMP = 5
N_BOOT = 2000
OUT_R = C.PROJECT_DIR / 'results_comparison'
OUT_F = C.PROJECT_DIR / 'figures_comparison'


def region_processed():
    out = {'ZI': C.PROJECT_DIR / 'processed'}
    for p in sorted(C.PROJECT_DIR.glob('processed_*')):
        out[p.name.split('_', 1)[1]] = p
    return {r: p for r, p in out.items() if p.exists() and any(p.iterdir())}


def load_region(pdir):
    sess = []
    for path in sorted(q for q in pdir.iterdir() if (q / 'meta.json').exists()):
        meta = json.loads((path / 'meta.json').read_text())
        _, t, u, b = K.load_insertion(path)
        ok = K.valid_trials(t)
        if ok.sum() < 100:
            continue
        t = t[ok].reset_index(drop=True)
        X = np.sqrt(K.window_counts(b, 'premove')[:, ok].T)          # trials x units
        sess.append(dict(pid=meta['pid'], subject=meta['subject'], t=t, X=X))
    return sess


def subset_acc(X, t_beh, n_tr, N, rng, n_subsets, n_sub):
    y = t_beh['choice'].to_numpy()[:n_tr]
    st = fu.strata(t_beh.iloc[:n_tr])
    accs = []
    for _ in range(n_subsets):
        cols = rng.choice(X.shape[1], N, replace=False)
        accs.append(fu.matched_acc(X[:n_tr][:, cols], y, st, rng, n_sub))
    return np.nanmean(accs)


def boot_ci(x, rng):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    b = [rng.choice(x, len(x)).mean() for _ in range(N_BOOT)]
    return np.percentile(b, [2.5, 97.5])


def main():
    OUT_R.mkdir(exist_ok=True)
    OUT_F.mkdir(exist_ok=True)
    rng = np.random.default_rng(C.RANDOM_SEED + 11)
    plt = K.setup_matplotlib()
    regions = region_processed()
    print('regions:', ', '.join(regions))
    rows = []
    for reg, pdir in regions.items():
        sess = load_region(pdir)
        print(f'{reg}: {len(sess)} insertions')
        for N in N_LIST:
            eligible = [s for s in sess if s['X'].shape[1] >= N]
            if len(eligible) < 5:
                continue
            real = {s['pid']: subset_acc(s['X'], s['t'], len(s['t']), N, rng, N_SUBSETS, N_SUB) for s in eligible}
            imp = {s['pid']: [] for s in eligible}
            for _ in range(N_ROUNDS_IMP):
                perm = fu.derangement(len(eligible), rng)
                for i, s in enumerate(eligible):
                    tb = eligible[perm[i]]['t']
                    n = min(len(s['t']), len(tb))
                    imp[s['pid']].append(subset_acc(s['X'], tb, n, N, rng, N_SUBSETS_IMP, 1))
            for s in eligible:
                rows.append(dict(region=reg, N=N, pid=s['pid'], subject=s['subject'], n_units=s['X'].shape[1],
                                 acc=real[s['pid']], imposter=np.mean(imp[s['pid']]),
                                 above=real[s['pid']] - np.mean(imp[s['pid']])))
            print(f'  N={N}: {len(eligible)} insertions done')
    df = pd.DataFrame(rows)
    df.to_csv(OUT_R / 'unit_matched_decoding.csv', index=False)

    L = ['=== Unit-count-matched direction decoding (premove; speed + block matched; minus imposter chance) ===']
    summ = []
    for (reg, N), g in df.groupby(['region', 'N']):
        lo, hi = boot_ci(g['above'], rng)
        p = stats.wilcoxon(g['above'], alternative='greater').pvalue if len(g) >= 5 else np.nan
        summ.append(dict(region=reg, N=N, n=len(g), mean=g['above'].mean(), lo=lo, hi=hi, p=p))
    summ = pd.DataFrame(summ)
    for N in N_LIST:
        s = summ[summ['N'] == N]
        if s.empty:
            continue
        L.append(f'--- N = {N} units')
        for _, r in s.iterrows():
            L.append(f"  {r['region']:5s}: {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] ({int(r['n'])} insertions, p={r['p']:.2g})")
        if 'ZI' in s['region'].values:
            z = df[(df['region'] == 'ZI') & (df['N'] == N)]['above']
            for reg in s['region']:
                if reg == 'ZI':
                    continue
                o = df[(df['region'] == reg) & (df['N'] == N)]['above']
                d = [rng.choice(z, len(z)).mean() - rng.choice(o, len(o)).mean() for _ in range(N_BOOT)]
                L.append(f'  ZI - {reg}: {z.mean() - o.mean():+.3f} (95% CI {np.percentile(d, 2.5):+.3f} to '
                         f'{np.percentile(d, 97.5):+.3f}; Mann-Whitney p={stats.mannwhitneyu(z, o).pvalue:.2g})')
    text = '\n'.join(L)
    (OUT_R / 'summary_11_unit_matched.txt').write_text(text, encoding='utf-8')
    print(text)

    cols = {'ZI': K.COL_CHOICE, 'MOs': '#2a9d8f', 'VPM': '#888888'}
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    for i, (reg, g) in enumerate(summ.groupby('region')):
        off = (i - 1) * 0.15
        ax.errorbar(g['N'] + off, g['mean'], yerr=[g['mean'] - g['lo'], g['hi'] - g['mean']], marker='o', ms=3,
                    capsize=2, lw=1.2, color=cols.get(reg, None), label=reg)
    ax.axhline(0, color='k', ls=':', lw=0.8)
    ax.set_xticks(N_LIST)
    ax.set(xlabel='Units per insertion (subsampled)', ylabel='Direction decoding\n(accuracy - imposter chance)',
           title='Information per neuron')
    ax.legend(frameon=False, fontsize=6)
    fig.savefig(OUT_F / 'step11_unit_matched.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_F / 'step11_unit_matched.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
