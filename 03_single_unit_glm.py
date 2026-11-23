"""
Step 03 - Single-unit encoding model (paper Figure 2 and Figure S1).

For every ZI unit and analysis window (iti / premove / move) we fit a linear model on
square-root spike counts (variance-stabilised):

  rate ~ choice + stimulus(side, |contrast|) + block prior + speed(linear + log)
         + reaction time + movement duration + video motion energy(whisker, body)
         + slow drift(trial number, trial number^2)

Each term is tested with a nested-model F statistic:
  * speed is tested JOINTLY (2 df: linear + log), so collinearity between the two terms does not hide it
  * stimulus jointly (side + |contrast|), motion energy jointly (whisker + body)
Significance is assessed with Freedman-Lane permutation (residuals of the reduced model are
permuted across trials, N_PERM times, jointly for all units in a session), and corrected with
Benjamini-Hochberg FDR across all units. A pooled permutation null gives the number of
significant units expected by chance for the whole population.

Also: time-resolved encoding (fraction of units with choice / speed effects in sliding windows
around movement onset; parametric F-tests) and example PSTHs.

Outputs:
  results/glm_units.csv          unit x window x term statistics
  results/glm_summary.csv        population fractions, CIs, permutation null, FDR counts, mouse-level tests
  results/glm_timecourse.csv     time-resolved fractions
  results/glm_vif.csv            collinearity diagnostics per session
  results/summary_03_glm.txt     numbers for the Results section
  figures/step03_single_unit_glm.(png|pdf)   quick-look figure
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d

import common as K
import config as C

TERMS = ['choice', 'speed', 'prior', 'stimulus', 'rt', 'dur', 'me']
TC_WIDTH, TC_STEP, TC_RANGE = 0.10, 0.04, (-0.40, 0.40)   # time-resolved encoding windows (firstMove)
MIN_EXAMPLE_RATE = 2.0                                     # Hz, minimum rate for example units


def _qr_resid(X):
    Q, _ = np.linalg.qr(X)
    return lambda Y: Y - Q @ (Q.T @ Y)


def nested_F(X, Y, cols):
    """F statistic (per column of Y) for dropping `cols` from X, plus partial R2 and betas."""
    n, p = X.shape
    keep = [j for j in range(p) if j not in cols]
    res_full = _qr_resid(X)
    res_red = _qr_resid(X[:, keep])
    rss_f = (res_full(Y) ** 2).sum(0)
    rss_r = (res_red(Y) ** 2).sum(0)
    q, dof = len(cols), n - p
    with np.errstate(divide='ignore', invalid='ignore'):
        F = ((rss_r - rss_f) / q) / (rss_f / dof)
        pr2 = (rss_r - rss_f) / rss_r
    return F, pr2, res_full, res_red, keep, dof


def freedman_lane(X, Y, cols, n_perm, rng):
    """Observed F, parametric p, permutation p, and per-permutation F matrix (units x n_perm)."""
    F_obs, pr2, res_full, res_red, keep, dof = nested_F(X, Y, cols)
    p_param = stats.f.sf(F_obs, len(cols), dof)
    R = res_red(Y)                       # residuals of reduced model
    fitted = Y - R
    n = Y.shape[0]
    F_null = np.empty((Y.shape[1], n_perm))
    for k in range(n_perm):
        Ys = fitted + R[rng.permutation(n)]
        rf = (res_full(Ys) ** 2).sum(0)
        rr = (res_red(Ys) ** 2).sum(0)
        with np.errstate(divide='ignore', invalid='ignore'):
            F_null[:, k] = ((rr - rf) / len(cols)) / (rf / dof)
    F_null = np.nan_to_num(F_null, nan=0.0)
    p_perm = (1 + (F_null >= F_obs[:, None]).sum(1)) / (1 + n_perm)
    return F_obs, p_param, p_perm, pr2, F_null


def null_sig_counts(F_null, alpha):
    """For each permutation k, number of units that would be called significant if perm k were the data."""
    # p of perm k for each unit = rank of F_null[:, k] among all perms (descending)
    ranks = (-F_null).argsort(1).argsort(1) + 1          # 1 = largest
    p_k = ranks / (F_null.shape[1] + 1)
    return (p_k < alpha).sum(0)


def unit_betas(X, Y, names):
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    b = dict(zip(names, beta))
    return {'choice': b['choice'], 'speed': b['speed_lin'] + b['speed_log'], 'prior': b['prior']}


def psth(counts, sel, sigma=1.5):
    return gaussian_filter1d(counts[:, sel, :].mean(1) / C.BIN, sigma, axis=-1)


def make_figure(df, summ, tc, rng):
    """Quick-look figure (also used by --figure-only)."""
    plt = K.setup_matplotlib()
    # overlap of choice and speed coding in the premove window
    pm = df[df['window'] == 'premove'].pivot_table(index=['pid', 'unit_index'], columns='term', values='p_perm')
    sc, ss = pm['choice'] < C.ALPHA, pm['speed'] < C.ALPHA
    both, c_only, s_only = int((sc & ss).sum()), int((sc & ~ss).sum()), int((~sc & ss).sum())
    neither = int((~sc & ~ss).sum())
    odds, p_fisher = stats.fisher_exact([[both, c_only], [s_only, neither]])

    # ---------------- quick-look figure ----------------
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(3, 3, hspace=0.75, wspace=0.55)
    term_lab = {'choice': 'Choice', 'speed': 'Speed', 'prior': 'Prior', 'stimulus': 'Stimulus',
                'rt': 'RT', 'dur': 'Duration', 'me': 'Video'}
    term_col = {'choice': K.COL_CHOICE, 'speed': K.COL_SPEED, 'prior': K.COL_PRIOR, 'stimulus': K.COL_STIM}
    for j, w in enumerate(['iti', 'premove', 'move']):
        ax = fig.add_subplot(gs[0, j])
        s = summ[summ['window'] == w].set_index('term').reindex([x for x in TERMS if x in summ['term'].unique()])
        x = np.arange(len(s))
        ax.bar(x, s['frac_sig'], color=[term_col.get(tm, K.COL_GREY) for tm in s.index], width=0.7)
        ax.errorbar(x, s['frac_sig'], yerr=[s['frac_sig'] - s['ci_low'], s['ci_high'] - s['frac_sig']],
                    fmt='none', ecolor='k', lw=0.7)
        ax.scatter(x, s['frac_sig_fdr'], marker='_', s=60, color='k', zorder=3, label='FDR q<0.05')
        for xi, (nm, n95) in enumerate(zip(s['null_frac_mean'], s['null_frac_95'])):
            ax.fill_between([xi - 0.4, xi + 0.4], 0, n95, color='0.85', zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([term_lab[tm] for tm in s.index], fontsize=6, rotation=45, ha='right')
        ax.set(ylim=(0, max(0.6, s['ci_high'].max() + 0.05)), ylabel=f'Fraction of {C.REGION} units' if j == 0 else '',
               title=f"{'ABC'[j]}  {w} window")
        if j == 0:
            ax.legend(frameon=False, fontsize=6, loc='upper right')

    # D overlap
    ax = fig.add_subplot(gs[1, 0])
    tot = both + c_only + s_only + neither
    vals = [c_only, both, s_only]
    ax.bar([0, 1, 2], np.array(vals) / tot, color=[K.COL_CHOICE, '#7a5f80', K.COL_SPEED], width=0.7)
    exp_both = (c_only + both) * (s_only + both) / tot ** 2
    ax.plot([0.6, 1.4], [exp_both] * 2, 'k--', lw=0.8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['choice\nonly', 'both', 'speed\nonly'], fontsize=7)
    ax.set(ylabel='Fraction of units', title=f'D  Mixed selectivity (premove)\nFisher p={p_fisher:.2g}')

    # E per-mouse fractions
    ax = fig.add_subplot(gs[1, 1])
    for xi, (w, term, col) in enumerate([('premove', 'choice', K.COL_CHOICE), ('move', 'speed', K.COL_SPEED),
                                         ('iti', 'prior', K.COL_PRIOR)]):
        g = df[(df['window'] == w) & (df['term'] == term)]
        pmouse = g.groupby('subject')['p_perm'].agg(lambda x: (x < C.ALPHA).mean() if len(x) >= 3 else np.nan).dropna()
        ax.scatter(xi + rng.uniform(-0.15, 0.15, len(pmouse)), pmouse, s=8, color=col, alpha=0.7, lw=0)
        ax.plot([xi - 0.25, xi + 0.25], [pmouse.median()] * 2, 'k', lw=1.2)
        nf = summ.query('window == @w and term == @term')['null_frac_mean'].iloc[0]
        ax.plot([xi - 0.3, xi + 0.3], [nf] * 2, color='0.5', ls=':', lw=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['choice\n(premove)', 'speed\n(move)', 'prior\n(iti)'], fontsize=7)
    ax.set(ylabel='Fraction per mouse', title='E  Consistency across mice')

    # F time-resolved encoding
    ax = fig.add_subplot(gs[1, 2])
    for term, col in (('choice', K.COL_CHOICE), ('speed', K.COL_SPEED), ('prior', K.COL_PRIOR)):
        g = tc[tc['term'] == term]
        ax.plot(g['t_center'] * 1000, g['frac'], color=col, lw=1.3, label=term)
    ax.axvline(0, color='k', lw=0.6)
    ax.axhline(C.ALPHA, color='0.5', ls=':', lw=0.8)
    ax.set(xlabel='Time from movement onset (ms)', ylabel='Fraction of units (p<0.05)',
           title='F  Time-resolved encoding')
    ax.legend(frameon=False, fontsize=6)

    # G/H example units
    edges, cent = K.bin_centers()
    for j, (term, w, title) in enumerate([('choice', 'premove', 'G  Example choice unit'),
                                          ('speed', 'move', 'H  Example speed unit')]):
        g = df[(df['window'] == w) & (df['term'] == term)].sort_values('F', ascending=False)
        if g.empty:
            continue
        counts = None
        for _, best in g.head(100).iterrows():          # strongest unit with a usable firing rate
            meta, t, u, b = K.load_insertion(C.PROCESSED_DIR / best['pid'])
            ok = K.valid_trials(t)
            ui = int(np.where(u['unit_index'].to_numpy() == best['unit_index'])[0][0])
            cc = b['fm_counts'][ui:ui + 1][:, ok, :].astype(float)
            if cc.mean() / C.BIN >= MIN_EXAMPLE_RATE and ok.sum() >= 100:
                counts, t = cc, t[ok].reset_index(drop=True)
                break
        if counts is None:
            continue
        terc = pd.qcut(t['peak_speed'], 3, labels=False).to_numpy()
        ax = fig.add_subplot(gs[2, j])
        for ti, ls in zip(range(3), [':', '--', '-']):
            for ch, col in ((1, K.COL_CHOICE), (-1, '#d1495b')):
                sel = (terc == ti) & (t['choice'].to_numpy() == ch)
                if sel.sum() >= 5:
                    ax.plot(cent * 1000, psth(counts, sel)[0], color=col, ls=ls, lw=1)
        ax.axvline(0, color='k', lw=0.6)
        ax.set(xlabel='Time from movement onset (ms)', ylabel='Firing rate (Hz)', title=title, xlim=(-400, 400))
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=K.COL_CHOICE, lw=1.2, label='choice +1'),
          Line2D([], [], color='#d1495b', lw=1.2, label='choice -1'),
          Line2D([], [], color='k', ls=':', lw=1, label='slow tercile'),
          Line2D([], [], color='k', ls='--', lw=1, label='medium'),
          Line2D([], [], color='k', ls='-', lw=1, label='fast')]
    ax.legend(handles=hs, frameon=False, fontsize=6, loc='lower left', bbox_to_anchor=(0, 0), title='G, H', title_fontsize=6)
    ax.text(0, 0.98, 'Grey band: 95th pct of permutation null\nBlack tick: FDR q<0.05', va='top', fontsize=6)
    C.save_fig(fig, 'step03_single_unit_glm')
    return both, c_only, s_only, neither, odds, p_fisher


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--figure-only', action='store_true', help='redraw the quick-look figure from saved results (no refitting)')
    args = ap.parse_args()
    rng = np.random.default_rng(C.RANDOM_SEED)
    if args.figure_only:
        make_figure(pd.read_csv(C.RESULTS_DIR / 'glm_units.csv'), pd.read_csv(C.RESULTS_DIR / 'glm_summary.csv'),
                    pd.read_csv(C.RESULTS_DIR / 'glm_timecourse.csv'), rng)
        print('Figure redrawn.')
        return
    rows, vifs, tc_rows = [], [], []
    null_counts = {}                      # (window, term) -> summed null counts over sessions
    sens_rows = []

    paths = K.list_insertions()
    for si, path in enumerate(paths):
        meta, t, u, b = K.load_insertion(path)
        ok = K.valid_trials(t)
        t = t[ok].reset_index(drop=True)
        print(f'[{si + 1}/{len(paths)}] {meta["subject"]} {meta["pid"][:8]}: {len(u)} units, {len(t)} trials')
        if len(t) < 50:
            continue
        for w in C.WINDOWS:
            X, names, idx = K.design_matrix(t, w)
            Y = np.sqrt(K.window_counts(b, w)[:, ok].T)          # trials x units
            if w == 'premove':
                vifs.append(dict(pid=meta['pid'], **{f'vif_{k}': v for k, v in K.vif(X, names).items()}))
            betas = unit_betas(X, Y, names)
            for term in TERMS:
                if term not in idx:
                    continue
                F, pp, pperm, pr2, Fnull = freedman_lane(X, Y, idx[term], C.N_PERM, rng)
                nc = null_sig_counts(Fnull, C.ALPHA)
                null_counts[(w, term)] = null_counts.get((w, term), 0) + nc
                for i in range(Y.shape[1]):
                    rows.append(dict(pid=meta['pid'], eid=meta['eid'], subject=meta['subject'],
                                     unit_index=int(u['unit_index'].iloc[i]),
                                     x=u['x'].iloc[i], y=u['y'].iloc[i], z=u['z'].iloc[i],
                                     window=w, term=term, F=F[i], p_param=pp[i], p_perm=pperm[i],
                                     partial_r2=pr2[i], beta=betas[term][i] if term in betas else np.nan,
                                     n_trials=len(t)))
            # sensitivity: model without motion energy (parametric)
            Xn, nn, idn = K.design_matrix(t, w, include_me=False)
            for term in ('choice', 'speed'):
                F, *_r = nested_F(Xn, Y, idn[term])
                pp = stats.f.sf(F, len(idn[term]), Xn.shape[0] - Xn.shape[1])
                sens_rows += [dict(pid=meta['pid'], window=w, term=term, p_param_noME=v) for v in pp]

        # time-resolved encoding (parametric), aligned to movement onset
        starts = np.round(np.arange(TC_RANGE[0], TC_RANGE[1] - TC_WIDTH + 1e-9, TC_STEP), 3)
        for s in starts:
            win = ('firstMove', s, s + TC_WIDTH)
            X, names, idx = K.design_matrix(t, 'premove' if s < 0 else 'move')
            Y = np.sqrt(K.window_counts(b, win)[:, ok].T)
            for term in ('choice', 'speed', 'prior'):
                F, *_r = nested_F(X, Y, idx[term])
                pp = stats.f.sf(F, len(idx[term]), X.shape[0] - X.shape[1])
                tc_rows.append(dict(pid=meta['pid'], t_center=s + TC_WIDTH / 2, term=term,
                                    n_sig=int(np.nansum(pp < C.ALPHA)), n_units=Y.shape[1]))

    df = pd.DataFrame(rows)
    df['q_perm'] = np.nan
    df['q_param'] = np.nan
    for (w, term), g in df.groupby(['window', 'term']):
        df.loc[g.index, 'q_perm'] = K.fdr_bh(g['p_perm'].to_numpy())
        df.loc[g.index, 'q_param'] = K.fdr_bh(g['p_param'].to_numpy())
    df.to_csv(C.RESULTS_DIR / 'glm_units.csv', index=False)
    pd.DataFrame(vifs).to_csv(C.RESULTS_DIR / 'glm_vif.csv', index=False)
    tc = pd.DataFrame(tc_rows).groupby(['t_center', 'term'])[['n_sig', 'n_units']].sum().reset_index()
    tc['frac'] = tc['n_sig'] / tc['n_units']
    tc.to_csv(C.RESULTS_DIR / 'glm_timecourse.csv', index=False)

    # ---------------- population summary ----------------
    summ = []
    for (w, term), g in df.groupby(['window', 'term']):
        n = len(g)
        k = int((g['p_perm'] < C.ALPHA).sum())
        lo, hi = K.binom_ci(k, n)
        nullc = null_counts[(w, term)]
        p_group = (1 + (nullc >= k).sum()) / (1 + len(nullc))
        per_mouse = g.groupby('subject').agg(n=('p_perm', 'size'), k=('p_perm', lambda x: (x < C.ALPHA).sum()))
        per_mouse = per_mouse[per_mouse['n'] >= 3]
        fm = per_mouse['k'] / per_mouse['n']
        null_frac = nullc.mean() / n
        try:
            p_mouse = stats.wilcoxon(fm - null_frac, alternative='greater').pvalue
        except ValueError:
            p_mouse = np.nan
        summ.append(dict(window=w, term=term, n_units=n, n_sig=k, frac_sig=k / n, ci_low=lo, ci_high=hi,
                         null_frac_mean=null_frac, null_frac_95=np.percentile(nullc, 95) / n,
                         p_population_perm=p_group, n_sig_fdr=int((g['q_perm'] < C.ALPHA).sum()),
                         frac_sig_fdr=(g['q_perm'] < C.ALPHA).mean(),
                         n_sig_fdr_param=int((g['q_param'] < C.ALPHA).sum()),
                         frac_sig_fdr_param=(g['q_param'] < C.ALPHA).mean(), n_mice=len(fm),
                         median_frac_per_mouse=fm.median(), n_mice_above_null=int((fm > null_frac).sum()),
                         p_mouse_wilcoxon=p_mouse))
    summ = pd.DataFrame(summ)
    summ.to_csv(C.RESULTS_DIR / 'glm_summary.csv', index=False)

    sens = pd.DataFrame(sens_rows)
    sens.to_csv(C.RESULTS_DIR / 'glm_sensitivity.csv', index=False)
    both, c_only, s_only, neither, odds, p_fisher = make_figure(df, summ, tc, rng)


    # ---------------- summary text ----------------
    def line(w, term):
        r = summ.query('window == @w and term == @term')
        if r.empty:
            return f'{w}/{term}: not available'
        r = r.iloc[0]
        return (f"{w:8s} {term:9s}: {r.n_sig}/{r.n_units} = {r.frac_sig:.1%} [95% CI {r.ci_low:.1%}-{r.ci_high:.1%}] "
                f"(perm null {r.null_frac_mean:.1%}, p_pop={r.p_population_perm:.3g}); FDR perm: {r.n_sig_fdr} ({r.frac_sig_fdr:.1%}), "
                f"FDR param: {r.n_sig_fdr_param} ({r.frac_sig_fdr_param:.1%}); "
                f"mice above null {r.n_mice_above_null}/{r.n_mice}, Wilcoxon p={r.p_mouse_wilcoxon:.3g}")
    lines = ['=== Single-unit GLM (Freedman-Lane permutation, %d perms) ===' % C.N_PERM]
    lines += [line(w, term) for w in C.WINDOWS for term in TERMS]
    lines += ['=== Mixed selectivity (premove, p_perm<0.05) ===',
              f'choice only {c_only}, speed only {s_only}, both {both}, neither {neither}; '
              f'Fisher exact OR={odds:.2f}, p={p_fisher:.3g}']
    if not sens.empty:
        lines.append('=== Sensitivity: model without video motion energy (parametric p<0.05) ===')
        for (w, term), g in sens.groupby(['window', 'term']):
            lines.append(f'{w:8s} {term:7s}: {(g["p_param_noME"] < C.ALPHA).mean():.1%}')
    v = pd.DataFrame(vifs)
    if 'vif_choice' in v:
        lines.append(f"VIF choice median {v['vif_choice'].median():.2f} (max {v['vif_choice'].max():.2f}); "
                     f"stim_side median {v['vif_stim_side'].median():.2f}")
    pk = tc[tc['term'] == 'choice'].sort_values('frac').iloc[-1]
    lines.append(f"time-resolved: choice fraction peaks at {pk.t_center * 1000:.0f} ms from movement onset ({pk.frac:.1%})")
    C.write_summary('summary_03_glm', lines)


if __name__ == '__main__':
    main()
