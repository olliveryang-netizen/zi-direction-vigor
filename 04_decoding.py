"""
Step 04 - Population decoding of choice with movement controls (paper Figure 3).

Per insertion with >= MIN_UNITS_POPULATION ZI units, pre-movement population activity
(sqrt spike counts, -160..0 ms from movement onset) is used to decode the reported choice
(L2 logistic regression, z-scored within training folds, balanced accuracy, 5-fold CV).

  A  Unmatched vs MOVEMENT-MATCHED decoding: in the matched analysis, trials are subsampled so that
     both choices have equal numbers in every peak-speed quantile bin (repeated N_SUBSAMPLE times).
  B  Permutation null for matched decoding: choice labels shuffled WITHIN speed bins (N_PERM_DECODING).
  C  Stimulus control: decoder trained on correct trials (speed-matched) and tested on ERROR trials and on
     0%-contrast trials. A stimulus decoder would score < 0.5 on errors; a choice decoder > 0.5.
  D  Movement generalisation: train on fast trials, test on slow trials (and vice versa), vs within-condition CV.
  E  Time-resolved decoding (100-ms windows, 40-ms steps): choice (matched) and wheel speed
     (log peak speed residualised on choice -> choice-independent movement vigour; ridge regression, CV R2).

Outputs: results/decoding_sessions.csv, results/decoding_timecourse.csv, results/summary_04_decoding.txt,
         figures/step04_decoding.(png|pdf)
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as K
import config as C

warnings.filterwarnings('ignore', category=UserWarning)
N_FOLD = 5
N_SUB_TC = 10
TC_WIDTH, TC_STEP, TC_RANGE = 0.10, 0.04, (-0.40, 0.40)


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))


def cv_balacc(X, y, rng):
    if min((y == 1).sum(), (y == -1).sum()) < N_FOLD:
        return np.nan
    skf = StratifiedKFold(N_FOLD, shuffle=True, random_state=int(rng.integers(1e9)))
    pred = np.empty_like(y)
    for tr, te in skf.split(X, y):
        pred[te] = clf().fit(X[tr], y[tr]).predict(X[te])
    return balanced_accuracy_score(y, pred)


def matched_idx(choice, sbin, rng, pool=None):
    """Indices with equal numbers of each choice inside every speed bin."""
    pool = np.arange(len(choice)) if pool is None else pool
    out = []
    for bb in np.unique(sbin[pool]):
        a = pool[(sbin[pool] == bb) & (choice[pool] == 1)]
        c = pool[(sbin[pool] == bb) & (choice[pool] == -1)]
        m = min(len(a), len(c))
        if m > 0:
            out += list(rng.choice(a, m, replace=False)) + list(rng.choice(c, m, replace=False))
    return np.array(out, int)


def shuffle_within(y, groups, rng):
    y = y.copy()
    for g in np.unique(groups):
        i = np.where(groups == g)[0]
        y[i] = y[rng.permutation(i)]
    return y


def session_decoding(t, b, rng):
    ok = K.valid_trials(t)
    t = t[ok].reset_index(drop=True)
    X = np.sqrt(K.window_counts(b, 'premove')[:, ok].T)
    y = t['choice'].to_numpy()
    sbin = pd.qcut(t['peak_speed'], C.N_SPEED_BINS, labels=False, duplicates='drop').to_numpy()
    res = {'n_trials': len(t), 'n_units': X.shape[1]}

    # A unmatched vs matched
    res['acc_unmatched'] = cv_balacc(X, y, rng)
    accs = []
    for _ in range(C.N_SUBSAMPLE):
        idx = matched_idx(y, sbin, rng)
        accs.append(cv_balacc(X[idx], y[idx], rng))
    res['acc_matched'] = np.nanmean(accs)
    res['n_matched_trials'] = len(idx)

    # B permutation null (labels shuffled within speed bins)
    null = []
    for _ in range(C.N_PERM_DECODING):
        idx = matched_idx(y, sbin, rng)
        ys = shuffle_within(y[idx], sbin[idx], rng)
        null.append(cv_balacc(X[idx], ys, rng))
    null = np.array(null, float)
    res['null_mean'] = np.nanmean(null)
    res['p_matched'] = (1 + np.nansum(null >= res['acc_matched'])) / (1 + np.isfinite(null).sum())

    # C correct -> error / zero-contrast generalisation (training set speed-matched)
    corr = np.where((t['correct'] == 1) & (t['signed_contrast'] != 0))[0]
    err = np.where(t['correct'] == 0)[0]
    zero = np.where(t['signed_contrast'] == 0)[0]
    for name, test in (('error', err), ('zero', zero)):
        if min((y[test] == 1).sum(), (y[test] == -1).sum()) < 5:
            res[f'acc_corr_to_{name}'] = np.nan
            continue
        a = []
        for _ in range(10):
            tr = matched_idx(y, sbin, rng, pool=corr)
            if len(np.unique(y[tr])) < 2:
                continue
            a.append(balanced_accuracy_score(y[test], clf().fit(X[tr], y[tr]).predict(X[test])))
        res[f'acc_corr_to_{name}'] = np.nanmean(a) if a else np.nan
    res['n_error_a'], res['n_error_b'] = int((y[err] == 1).sum()), int((y[err] == -1).sum())

    # D fast <-> slow generalisation
    fast = t['peak_speed'].to_numpy() >= np.median(t['peak_speed'])
    within, cross = [], []
    for _ in range(10):
        for trm, tem in ((fast, ~fast), (~fast, fast)):
            tr = matched_idx(y, np.zeros(len(y)), rng, pool=np.where(trm)[0])
            te = matched_idx(y, np.zeros(len(y)), rng, pool=np.where(tem)[0])
            if len(tr) < 2 * N_FOLD or len(te) < 10:
                continue
            within.append(cv_balacc(X[tr], y[tr], rng))
            cross.append(balanced_accuracy_score(y[te], clf().fit(X[tr], y[tr]).predict(X[te])))
    res['acc_within_speed'] = np.nanmean(within) if within else np.nan
    res['acc_cross_speed'] = np.nanmean(cross) if cross else np.nan
    return res


def session_timecourse(t, b, rng):
    ok = K.valid_trials(t)
    t = t[ok].reset_index(drop=True)
    y = t['choice'].to_numpy()
    sbin = pd.qcut(t['peak_speed'], C.N_SPEED_BINS, labels=False, duplicates='drop').to_numpy()
    ls = np.log(t['peak_speed'].to_numpy())
    resid = ls - np.where(y == 1, ls[y == 1].mean(), ls[y == -1].mean())      # choice-independent vigour
    out = []
    for s in np.round(np.arange(TC_RANGE[0], TC_RANGE[1] - TC_WIDTH + 1e-9, TC_STEP), 3):
        X = np.sqrt(K.window_counts(b, ('firstMove', s, s + TC_WIDTH))[:, ok].T)
        acc = np.nanmean([cv_balacc(X[i], y[i], rng) for i in (matched_idx(y, sbin, rng) for _ in range(N_SUB_TC))])
        pred = np.empty_like(resid)
        for tr, te in KFold(N_FOLD, shuffle=True, random_state=int(rng.integers(1e9))).split(X):
            pred[te] = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X[tr], resid[tr]).predict(X[te])
        out.append(dict(t_center=s + TC_WIDTH / 2, acc_choice_matched=acc, r2_speed_resid=r2_score(resid, pred)))
    return out


def wil(x, mu=0.5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    # one-sided: is performance ABOVE chance?
    return stats.wilcoxon(x - mu, alternative='greater').pvalue if len(x) >= 5 else np.nan


def main():
    rng = np.random.default_rng(C.RANDOM_SEED + 4)
    plt = K.setup_matplotlib()
    rows, tcs = [], []
    paths = K.list_insertions()
    for si, path in enumerate(paths):
        meta, t, u, b = K.load_insertion(path)
        if len(u) < C.MIN_UNITS_POPULATION:
            continue
        print(f'[{si + 1}/{len(paths)}] {meta["subject"]} {meta["pid"][:8]} ({len(u)} units)')
        r = session_decoding(t, b, rng)
        rows.append(dict(pid=meta['pid'], subject=meta['subject'], **r))
        tcs += [dict(pid=meta['pid'], subject=meta['subject'], **d) for d in session_timecourse(t, b, rng)]

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / 'decoding_sessions.csv', index=False)
    tc = pd.DataFrame(tcs)
    tc.to_csv(C.RESULTS_DIR / 'decoding_timecourse.csv', index=False)
    mouse = df.groupby('subject').mean(numeric_only=True)

    # ---------------- figure ----------------
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.8))
    fig.subplots_adjust(hspace=0.7, wspace=0.55)
    ax = axs[0, 0]
    for _, r in df.iterrows():
        ax.plot([0, 1], [r['acc_unmatched'], r['acc_matched']], color='0.7', lw=0.6)
    ax.scatter(np.zeros(len(df)), df['acc_unmatched'], s=8, color='0.4', zorder=3)
    ax.scatter(np.ones(len(df)), df['acc_matched'], s=8, color=K.COL_CHOICE, zorder=3)
    ax.axhline(0.5, color='k', ls=':', lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['all trials', 'speed-\nmatched'])
    ax.set(xlim=(-0.4, 1.4), ylabel='Choice decoding\n(balanced accuracy)', title='A  Movement matching')

    ax = axs[0, 1]
    ax.scatter(df['null_mean'], df['acc_matched'], s=10, c=np.where(df['p_matched'] < 0.05, K.COL_CHOICE, '0.6'))
    lim = [min(0.4, df[['null_mean', 'acc_matched']].min().min()), max(0.9, df['acc_matched'].max() + 0.02)]
    ax.plot(lim, lim, 'k:', lw=0.8)
    ax.set(xlabel='Shuffled-within-speed null', ylabel='Matched accuracy',
           title=f"B  {int((df['p_matched'] < 0.05).sum())}/{len(df)} sessions p<0.05")

    ax = axs[0, 2]
    for xi, (col, lab) in enumerate((('acc_corr_to_error', 'error\ntrials'), ('acc_corr_to_zero', '0%\ncontrast'))):
        v = df[col].dropna()
        ax.scatter(xi + rng.uniform(-0.12, 0.12, len(v)), v, s=8, color=K.COL_CHOICE, alpha=0.7)
        ax.plot([xi - 0.25, xi + 0.25], [v.median()] * 2, 'k', lw=1.2)
    ax.axhline(0.5, color='k', ls=':', lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['error\ntrials', '0%\ncontrast'])
    ax.set(ylabel='Choice accuracy', title='C  Train correct -> test', xlim=(-0.5, 1.5))

    ax = axs[1, 0]
    for _, r in df.iterrows():
        ax.plot([0, 1], [r['acc_within_speed'], r['acc_cross_speed']], color='0.7', lw=0.6)
    ax.scatter(np.zeros(len(df)), df['acc_within_speed'], s=8, color='0.4', zorder=3)
    ax.scatter(np.ones(len(df)), df['acc_cross_speed'], s=8, color=K.COL_CHOICE, zorder=3)
    ax.axhline(0.5, color='k', ls=':', lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['within\nspeed half', 'fast <-> slow'])
    ax.set(xlim=(-0.4, 1.4), ylabel='Balanced accuracy', title='D  Cross-speed generalisation')

    g = tc.groupby('t_center')
    ax = axs[1, 1]
    m, s = g['acc_choice_matched'].mean(), g['acc_choice_matched'].agg(K.sem)
    ax.plot(m.index * 1000, m, color=K.COL_CHOICE)
    ax.fill_between(m.index * 1000, m - s, m + s, color=K.COL_CHOICE, alpha=0.25, lw=0)
    ax.axhline(0.5, color='k', ls=':', lw=0.8)
    ax.axvline(0, color='k', lw=0.6)
    ax.set(xlabel='Time from movement onset (ms)', ylabel='Choice accuracy (matched)', title='E  Choice over time')
    ax = axs[1, 2]
    m, s = g['r2_speed_resid'].mean(), g['r2_speed_resid'].agg(K.sem)
    ax.plot(m.index * 1000, m, color=K.COL_SPEED)
    ax.fill_between(m.index * 1000, m - s, m + s, color=K.COL_SPEED, alpha=0.25, lw=0)
    ax.axhline(0, color='k', ls=':', lw=0.8)
    ax.axvline(0, color='k', lw=0.6)
    ax.set(xlabel='Time from movement onset (ms)', ylabel='CV R$^2$ (speed | choice)', title='F  Movement vigour over time')
    C.save_fig(fig, 'step04_decoding')

    # time-resolved significance (Wilcoxon across sessions, FDR over time)
    pt = g['acc_choice_matched'].apply(lambda x: wil(x)).to_numpy()
    qt = K.fdr_bh(pt)
    sig_t = m.index[qt < C.ALPHA] if np.any(qt < C.ALPHA) else []
    first_sig = (sig_t.min() * 1000 if len(sig_t) else np.nan)
    ps = g['r2_speed_resid'].apply(lambda x: wil(x, 0)).to_numpy()
    qs = K.fdr_bh(ps)
    sig_s = g['r2_speed_resid'].mean().index[qs < C.ALPHA]
    first_sig_s = (sig_s.min() * 1000 if len(sig_s) else np.nan)

    lines = [f'=== Choice decoding from {C.REGION} pre-movement activity (sessions with >= {C.MIN_UNITS_POPULATION} units) ===',
             f"sessions: {len(df)}, mice: {df['subject'].nunique()}",
             f"unmatched: {df['acc_unmatched'].mean():.3f} +/- {K.sem(df['acc_unmatched']):.3f} (Wilcoxon vs .5 p={wil(df['acc_unmatched']):.3g})",
             f"speed-matched: {df['acc_matched'].mean():.3f} +/- {K.sem(df['acc_matched']):.3f} (p={wil(df['acc_matched']):.3g}); "
             f"mouse-level p={wil(mouse['acc_matched']):.3g} (n={len(mouse)} mice)",
             f"matched vs unmatched (paired Wilcoxon): p={stats.wilcoxon(df['acc_unmatched'], df['acc_matched']).pvalue:.3g}",
             f"sessions significant vs within-speed-bin shuffle: {(df['p_matched'] < 0.05).sum()}/{len(df)} "
             f"(binomial p={stats.binomtest(int((df['p_matched'] < 0.05).sum()), len(df), 0.05, alternative='greater').pvalue:.3g})",
             f"correct->error: {df['acc_corr_to_error'].mean():.3f} (n={df['acc_corr_to_error'].notna().sum()}, p={wil(df['acc_corr_to_error']):.3g})",
             f"correct->0% contrast: {df['acc_corr_to_zero'].mean():.3f} (n={df['acc_corr_to_zero'].notna().sum()}, p={wil(df['acc_corr_to_zero']):.3g})",
             f"within-speed: {df['acc_within_speed'].mean():.3f}, cross-speed: {df['acc_cross_speed'].mean():.3f} "
             f"(cross vs .5 p={wil(df['acc_cross_speed']):.3g}; cross vs within p={stats.wilcoxon(df['acc_within_speed'], df['acc_cross_speed']).pvalue:.3g})",
             f"time-resolved: choice decoding first significant (FDR) at window centre {first_sig:.0f} ms; "
             f"choice-independent speed first significant at {first_sig_s:.0f} ms"]
    C.write_summary('summary_04_decoding', lines)


if __name__ == '__main__':
    main()
