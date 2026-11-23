"""
Step 08 - Compare ZI with control regions (paper Figures 6 and S5).

Run the full pipeline for each region first (run_all.py runs steps 01-07, 09 and 10):
    python run_all.py                  # ZI
    python run_all.py --region MOs     # positive control: secondary motor cortex
    python run_all.py --region VPM     # movement/sensory control: whisker thalamus
then:
    python 08_compare_regions.py

Per region (each metric is read only if the corresponding step was run for that region):
  A  choice (premove) and speed (move) units, GLM with all controls        [step 03]
  B  speed-matched choice decoding minus the imposter-session chance level [steps 04, 07]
  C  correct -> error generalisation                                       [step 04]
  D  choice-independent speed information 100-300 ms after movement onset  [step 04]
  E  change of mind: agreement with FINAL choice before movement onset     [step 09]
     and fraction of choice units preferring the contralateral choice      [step 09]
  F  timing: half-max latency of direction information (re movement onset) and
     crossing time relative to wheel reversal on change-of-mind trials     [step 10]

Outputs: results_comparison/region_comparison.csv, results_comparison/summary_08_regions.txt,
         figures_comparison/step08_region_comparison.(png|pdf)
"""
import importlib

import numpy as np
import pandas as pd
from scipy import stats

import common as K
import config as C

OUT_R = C.PROJECT_DIR / 'results_comparison'
OUT_F = C.PROJECT_DIR / 'figures_comparison'
ORDER = ['ZI', 'MOs', 'SCm', 'MRN', 'APN', 'VPM', 'RSPd', 'VISp']
N_BOOT = 2000


def region_dirs():
    dirs = {'ZI': C.PROJECT_DIR / 'results'}
    for p in sorted(C.PROJECT_DIR.glob('results_*')):
        reg = p.name.split('_', 1)[1]
        if reg != 'comparison':
            dirs[reg] = p
    return dirs


def boot_mean_ci(x, rng):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    b = [rng.choice(x, len(x)).mean() for _ in range(N_BOOT)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def _read(f):
    try:
        return pd.read_csv(f)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def collect(reg, d, rng, rev):
    row = dict(region=reg)
    # A GLM
    f = d / 'glm_summary.csv'
    if f.exists():
        g = pd.read_csv(f).set_index(['window', 'term'])
        row['n_units'] = int(g.loc[('premove', 'choice'), 'n_units'])
        for (w, term) in (('premove', 'choice'), ('move', 'speed')):
            row[f'{term}_frac'] = g.loc[(w, term), 'frac_sig']
            row[f'{term}_lo'], row[f'{term}_hi'] = g.loc[(w, term), 'ci_low'], g.loc[(w, term), 'ci_high']
    # B, C decoding
    f = d / 'decoding_sessions.csv'
    if f.exists():
        dd = pd.read_csv(f)
        row['n_sessions'] = len(dd)
        imp = d / 'imposter_null.csv'
        chance = pd.read_csv(imp)['acc_matched'].mean() if imp.exists() else 0.5
        row['chance_matched'] = chance
        x = dd['acc_matched'] - chance
        row['dec_above_chance'] = x.mean()
        row['dec_lo'], row['dec_hi'] = boot_mean_ci(x, rng)
        row['dec_p'] = stats.wilcoxon(x, alternative='greater').pvalue if len(x) >= 5 else np.nan
        x = dd['acc_corr_to_error'].dropna() - 0.5
        row['err_gen'] = x.mean()
        row['err_lo'], row['err_hi'] = boot_mean_ci(x, rng)
        row['err_p'] = stats.wilcoxon(x, alternative='greater').pvalue if len(x) >= 5 else np.nan
    # D vigour information
    f = d / 'decoding_timecourse.csv'
    if f.exists():
        tc = pd.read_csv(f)
        s = tc[(tc['t_center'] >= 0.10) & (tc['t_center'] <= 0.30)].groupby('pid')['r2_speed_resid'].mean()
        row['vigour_r2'] = s.mean()
        row['vig_lo'], row['vig_hi'] = boot_mean_ci(s, rng)
        row['vig_p'] = stats.wilcoxon(s, alternative='greater').pvalue if len(s) >= 5 else np.nan
    # E change of mind + lateralisation
    f = d / 'followup_com_trials.csv'
    if f.exists():
        c = _read(f)
        c = c[c['window'] == 'premove'] if 'window' in c else c
        if len(c):
            per = c.groupby('pid')['agree_final'].mean()
            row['com_final'] = c['agree_final'].mean()
            row['com_lo'], row['com_hi'] = boot_mean_ci(per, rng)
            row['com_n'] = len(c)
    f = d / 'followup_lateralisation_units.csv'
    if f.exists():
        lu = _read(f)
        sig = lu[lu['p_perm'] < C.ALPHA] if 'p_perm' in lu else lu
        if len(sig):
            row['contra_frac'] = sig['prefers_contra'].mean()
            row['contra_lo'], row['contra_hi'] = K.binom_ci(int(sig['prefers_contra'].sum()), len(sig))
            row['contra_p'] = stats.binomtest(int(sig['prefers_contra'].sum()), len(sig), 0.5).pvalue
    # F timing
    f = d / 'onset_sessions.npz'
    if f.exists():
        z = np.load(f)
        diff = z['real'] - z['imp_mean']
        cen = z['centers'] * 1000
        row['latency'] = rev.halfmax_latency(cen, diff.mean(0))
        bs = np.array([rev.halfmax_latency(cen, diff[rng.integers(0, len(diff), len(diff))].mean(0)) for _ in range(N_BOOT)])
        row['lat_lo'], row['lat_hi'] = np.nanpercentile(bs, [2.5, 97.5])
    f = d / 'reversal_curves.csv'
    if f.exists():
        cv = _read(f)
        taus = np.array([float(c.split('_')[1].replace('ms', '')) for c in cv.columns])
        arr = cv.to_numpy(float)
        if len(arr) >= 5:
            row['reversal_cross'] = rev.crossing(np.nanmean(arr, 0), taus)
            bs = np.array([rev.crossing(np.nanmean(arr[rng.integers(0, len(arr), len(arr))], 0), taus) for _ in range(N_BOOT)])
            row['rev_lo'], row['rev_hi'] = (np.nanpercentile(bs, [2.5, 97.5]) if np.isfinite(bs).any() else (np.nan, np.nan))
            row['rev_frac_noswitch'] = np.mean(~np.isfinite(bs))
    # a latency is only meaningful if the region carries significant direction information
    if not (row.get('dec_p', 1) < C.ALPHA):
        for k in ('latency', 'lat_lo', 'lat_hi'):
            row[k] = np.nan
        row['latency_note'] = 'no significant direction information'
    return row


def bars(ax, x, df, col, lo, hi, color, ref=None, label=None):
    if col not in df:
        return
    v = df[col].to_numpy(float)
    ok = np.isfinite(v)
    ax.bar(x[ok], v[ok], 0.6, color=color, label=label)
    if lo in df and hi in df:
        e = np.vstack([v - df[lo].to_numpy(float), df[hi].to_numpy(float) - v])
        ax.errorbar(x[ok], v[ok], yerr=e[:, ok], fmt='none', ecolor='k', lw=0.7)
    if ref is not None:
        ax.axhline(ref, color='k', ls=':', lw=0.8)


def main():
    OUT_R.mkdir(exist_ok=True)
    OUT_F.mkdir(exist_ok=True)
    plt = K.setup_matplotlib()
    rng = np.random.default_rng(C.RANDOM_SEED + 8)
    rev = importlib.import_module('10_reversal_history')
    rows = [collect(reg, d, rng, rev) for reg, d in region_dirs().items()]
    df = pd.DataFrame([r for r in rows if len(r) > 1])
    if df.empty:
        print('No region results found.')
        return
    df['order'] = df['region'].map({r: i for i, r in enumerate(ORDER)}).fillna(99)
    df = df.sort_values('order').drop(columns='order').reset_index(drop=True)
    df.to_csv(OUT_R / 'region_comparison.csv', index=False)

    def fmt(r, c, lo=None, hi=None, p=None, pct=False, ms=False):
        if c not in r or not np.isfinite(r[c]):
            return 'n/a'
        f = (lambda v: f'{v:.1%}') if pct else (lambda v: f'{v:.0f} ms') if ms else (lambda v: f'{v:+.3f}')
        s = f(r[c])
        if lo and lo in r and np.isfinite(r[lo]):
            s += f' [{f(r[lo])}, {f(r[hi])}]'
        if p and p in r and np.isfinite(r[p]):
            s += f' p={r[p]:.2g}'
        return s

    L = ['=== Region comparison (95% CIs: binomial for unit fractions, bootstrap over sessions otherwise) ===']
    for _, r in df.iterrows():
        r = r.to_dict()
        nu = r.get('n_units', np.nan)
        ns = r.get('n_sessions', np.nan)
        L += [f"--- {r['region']}  ({'?' if not np.isfinite(nu) else int(nu)} units, "
              f"{'?' if not np.isfinite(ns) else int(ns)} decoding sessions)",
              f"  choice units (premove, GLM):          {fmt(r, 'choice_frac', 'choice_lo', 'choice_hi', pct=True)}",
              f"  speed units (move, GLM):              {fmt(r, 'speed_frac', 'speed_lo', 'speed_hi', pct=True)}",
              f"  matched decoding - imposter chance:   {fmt(r, 'dec_above_chance', 'dec_lo', 'dec_hi', 'dec_p')}",
              f"  correct->error generalisation - 0.5:  {fmt(r, 'err_gen', 'err_lo', 'err_hi', 'err_p')}",
              f"  choice-independent vigour R2:         {fmt(r, 'vigour_r2', 'vig_lo', 'vig_hi', 'vig_p')}",
              f"  CoM agreement with final choice:      {fmt(r, 'com_final', 'com_lo', 'com_hi')}",
              f"  choice units preferring contra:       {fmt(r, 'contra_frac', 'contra_lo', 'contra_hi', 'contra_p', pct=True)}",
              f"  half-max latency (re movement onset): {fmt(r, 'latency', 'lat_lo', 'lat_hi', ms=True)}"
              + (f"  ({r['latency_note']})" if isinstance(r.get('latency_note'), str) else ''),
              f"  switch re wheel reversal (CoM):       {fmt(r, 'reversal_cross', 'rev_lo', 'rev_hi', ms=True)}"
              + (f"  (no switch in {r['rev_frac_noswitch']:.0%} of bootstraps)" if 'rev_frac_noswitch' in r and np.isfinite(r['rev_frac_noswitch']) else '')]
    text = '\n'.join(L)
    (OUT_R / 'summary_08_regions.txt').write_text(text, encoding='utf-8')
    print(text)

    x = np.arange(len(df))
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6))
    fig.subplots_adjust(hspace=0.65, wspace=0.55)
    ax = axs[0, 0]
    for off, term, colr in ((-0.2, 'choice', K.COL_CHOICE), (0.2, 'speed', K.COL_SPEED)):
        if f'{term}_frac' in df:
            v = df[f'{term}_frac'].to_numpy(float)
            ax.bar(x + off, v, 0.38, color=colr, label=term)
            ax.errorbar(x + off, v, yerr=[v - df[f'{term}_lo'], df[f'{term}_hi'] - v], fmt='none', ecolor='k', lw=0.6)
    ax.axhline(C.ALPHA, color='0.5', ls=':', lw=0.8)
    ax.set(ylabel='Fraction of units', title='A  GLM (all controls)')
    ax.legend(frameon=False, fontsize=6)
    bars(axs[0, 1], x, df, 'dec_above_chance', 'dec_lo', 'dec_hi', K.COL_CHOICE, ref=0)
    axs[0, 1].set(ylabel='Accuracy - imposter chance', title='B  Speed-matched decoding')
    bars(axs[0, 2], x, df, 'vigour_r2', 'vig_lo', 'vig_hi', K.COL_SPEED, ref=0)
    axs[0, 2].set(ylabel='CV R$^2$ (speed | choice)', title='C  Vigour info (100-300 ms)')
    bars(axs[1, 0], x, df, 'com_final', 'com_lo', 'com_hi', K.COL_CHOICE, ref=0.5)
    axs[1, 0].set(ylabel='Agreement with final choice', title='D  Change of mind (premove)', ylim=(0.2, 0.8))
    bars(axs[1, 1], x, df, 'contra_frac', 'contra_lo', 'contra_hi', K.COL_CHOICE, ref=0.5)
    axs[1, 1].set(ylabel='Choice units preferring contra', title='E  Lateralisation', ylim=(0, 1))
    ax = axs[1, 2]
    for off, c, lo, hi, colr, lab in ((-0.2, 'latency', 'lat_lo', 'lat_hi', K.COL_CHOICE, 're movement onset'),
                                      (0.2, 'reversal_cross', 'rev_lo', 'rev_hi', '0.4', 're wheel reversal')):
        if c in df:
            v = df[c].to_numpy(float)
            ok = np.isfinite(v)
            ax.errorbar(x[ok] + off, v[ok], yerr=np.vstack([v - df[lo], df[hi] - v])[:, ok], fmt='o', ms=3,
                        color=colr, lw=0.8, capsize=2, label=lab)
    ax.axhline(0, color='k', lw=0.6)
    ax.set(ylabel='Lead (ms, negative = before)', title='F  Timing of direction signal')
    ax.legend(frameon=False, fontsize=5.5, loc='upper left')
    for a in axs.flat:
        a.set_xticks(x)
        a.set_xticklabels(df['region'])
        a.set_xlim(-0.6, len(df) - 0.4)
    fig.savefig(OUT_F / 'step08_region_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_F / 'step08_region_comparison.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
