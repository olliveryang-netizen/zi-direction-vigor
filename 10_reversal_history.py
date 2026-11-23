"""
Step 10 - Reversal timing, trial history, power-matched feedback control, onset comparison across regions
(paper Figures 3E-F, 4B and S5). Uses existing processed data only.

A  REVERSAL-ALIGNED CHANGE OF MIND
   On change-of-mind (CoM) trials the wheel first moves one way, then reverses to the final choice.
   A time-general direction decoder (trained on consistent trials, 60-ms windows from -100 to +300 ms
   around movement onset) is applied to CoM trials aligned to the moment the wheel REVERSES.
     P(final choice) crosses 0.5 BEFORE the reversal  -> ZI anticipates the new direction (re-decision)
     ... crosses AFTER the reversal                   -> ZI follows the executed movement
   Crossing time: bootstrap over sessions.

B  TRIAL HISTORY
   Previous choice and previous reward (and their interaction) are added to the GLM (iti, premove) and
   used as an extra matching variable (speed x block x previous choice) for time-resolved decoding
   against imposter sessions. Tests whether early (pre-stimulus) choice information is history.

C  POWER-MATCHED FEEDBACK CONTROL
   Fraction of move-window units on feedback-free trials vs equally sized RANDOM subsets of all trials
   (removes the loss-of-power explanation).

D  ONSET COMPARISON ACROSS REGIONS
   Half-maximum latency of (real - imposter) decoding, history-matched, bootstrap over sessions.
   Compares every region with saved results (results/, results_MOs/, ...). Run this script once per region:
       python 10_reversal_history.py                     (ZI)
       ANALYSIS_REGION=MOs python 10_reversal_history.py
   The comparison is printed at the end of whichever run finishes last.

Outputs: results/reversal_*.csv, results/history_*.csv, results/onset_sessions.npz,
         results/summary_10_reversal_history.txt, figures/step10_reversal_history.(png|pdf)
"""
import importlib

import numpy as np
import pandas as pd
from scipy import stats

import common as K
import config as C

glm = importlib.import_module('03_single_unit_glm')
dec = importlib.import_module('04_decoding')
fu = importlib.import_module('09_followup')

WIN_BINS = 3                    # 60-ms decoding windows
TRAIN_RANGE = (-0.10, 0.30)     # consistent-trial training times (s from movement onset)
REV_MAX = 0.30                  # reversal must occur by 300 ms (data extend to +500 ms)
TAU = np.arange(-15, 11)        # bins relative to reversal (-300..+200 ms)
N_BOOT = 2000
N_REP_FB = 20
N_ROUNDS_IMP = 10
N_SUB = 5
N_SUB_IMP = 2


# ----------------------------------------------------------------------------
def add_history(t):
    t = t.copy()
    lut_c = dict(zip(t['trial_number'], t['choice']))
    lut_r = dict(zip(t['trial_number'], t['feedbackType']))
    t['prev_choice'] = [lut_c.get(n - 1, 0) for n in t['trial_number']]
    t['prev_reward'] = [lut_r.get(n - 1, 0) for n in t['trial_number']]
    return t


def history_design(t, window):
    X, names, idx = K.design_matrix(t, window)
    pc, pr = t['prev_choice'].to_numpy(float), t['prev_reward'].to_numpy(float)
    H = np.column_stack([pc, pr, pc * pr])
    p0 = X.shape[1]
    idx = dict(idx, history=[p0, p0 + 1, p0 + 2])
    return np.column_stack([X, H]), names + ['prev_choice', 'prev_reward', 'prev_cxr'], idx


# ----------------------------------------------------------------------------
# A reversal-aligned decoding
# ----------------------------------------------------------------------------
def reversal_analysis(sess):
    _, cent = K.bin_centers()
    iw = (cent > fu.INIT_WIN[0]) & (cent <= fu.INIT_WIN[1])
    train_c = np.where((cent >= TRAIN_RANGE[0]) & (cent <= TRAIN_RANGE[1]))[0]
    curves, wheel_curves, info = [], [], []
    for s in sess:
        if len(s['u']) < C.MIN_UNITS_POPULATION:
            continue
        t, b = s['t'], s['b']
        wv = b['wheel_vel_fm']
        y = t['choice'].to_numpy()
        init_v = wv[:, iw].mean(1)
        strong = np.abs(init_v) > np.percentile(np.abs(init_v), 10)
        sign_map = np.sign(np.sum(y[strong] * np.sign(init_v[strong]))) or 1.0
        init_dir = (sign_map * np.sign(init_v)).astype(int)
        com = strong & (init_dir != y)
        vdir = sign_map * np.sign(wv)
        thr = 0.1 * np.nanmax(np.abs(wv[:, cent > 0]), axis=1)
        rev = np.full(len(t), -1)
        for i in np.where(com)[0]:
            cand = np.where((cent > fu.INIT_WIN[1]) & (cent <= REV_MAX) & (vdir[i] == y[i]) & (np.abs(wv[i]) > thr[i]))[0]
            if len(cand):
                rev[i] = cand[0]
        comv = np.where(com & (rev >= 0))[0]
        info.append(dict(pid=s['meta']['pid'], n_com=int(com.sum()), n_com_with_reversal=len(comv),
                         median_reversal_ms=float(np.median(cent[rev[comv]]) * 1000) if len(comv) else np.nan))
        if len(comv) < 3:
            continue
        cnt = b['fm_counts'].astype(float)
        S = np.sqrt(cnt[:, :, :-2] + cnt[:, :, 1:-1] + cnt[:, :, 2:])      # centre bin k -> index k-1
        train = np.where(strong & ~com)[0]
        cc = train_c[(train_c >= 1) & (train_c <= cnt.shape[2] - 2)]
        Xtr = np.concatenate([S[:, train, c - 1].T for c in cc])
        ytr = np.tile(y[train], len(cc))
        model = dec.clf().fit(Xtr, ytr)
        cls = list(model.classes_)
        P = np.full((len(comv), len(TAU)), np.nan)
        Wd = np.full((len(comv), len(TAU)), np.nan)
        for r, i in enumerate(comv):
            for j, tau in enumerate(TAU):
                c = rev[i] + tau
                if 1 <= c <= cnt.shape[2] - 2:
                    P[r, j] = model.predict_proba(S[:, i, c - 1][None])[0, cls.index(y[i])]
                    Wd[r, j] = float(vdir[i, c] == y[i])
        curves.append(np.nanmean(P, 0))
        wheel_curves.append(np.nanmean(Wd, 0))
    shape = (0, len(TAU))
    return (np.array(curves) if curves else np.empty(shape), np.array(wheel_curves) if wheel_curves else np.empty(shape),
            pd.DataFrame(info))


def crossing(curve, taus_ms):
    """LAST upward crossing of 0.5 (after it the curve stays >= 0.5), linear interpolation.
    NaN if the curve is never below 0.5 (no switch) or never above it."""
    ok = np.isfinite(curve)
    if not ok.any() or np.nanmin(curve) >= 0.5 or np.nanmax(curve) < 0.5:
        return np.nan
    last = np.nan
    for j in range(1, len(curve)):
        if ok[j - 1] and ok[j] and curve[j - 1] < 0.5 <= curve[j]:
            last = taus_ms[j - 1] + (0.5 - curve[j - 1]) / (curve[j] - curve[j - 1]) * (taus_ms[j] - taus_ms[j - 1])
    return last


# ----------------------------------------------------------------------------
# B history
# ----------------------------------------------------------------------------
def history_glm(sess):
    out = {}
    stay = []
    for s in sess:
        t = add_history(s['t'])
        has_prev = (t['prev_choice'] != 0).to_numpy()
        stay.append(np.mean(t.loc[has_prev, 'choice'] == t.loc[has_prev, 'prev_choice']))
        for w in ('iti', 'premove'):
            Y = np.sqrt(K.window_counts(s['b'], w).T)
            X0, _, idx0 = K.design_matrix(t, w)
            X1, _, idx1 = history_design(t, w)
            for model, X, idx, terms in (('base', X0, idx0, ('choice',)), ('history', X1, idx1, ('choice', 'history'))):
                for term in terms:
                    F, *_r = glm.nested_F(X, Y, idx[term])
                    p = stats.f.sf(F, len(idx[term]), X.shape[0] - X.shape[1])
                    key = (w, model, term)
                    out.setdefault(key, [0, 0])
                    out[key][0] += int(np.nansum(p < C.ALPHA))
                    out[key][1] += Y.shape[1]
    return out, float(np.mean(stay))


def strata_hist(t):
    return fu.strata(t) * 3 + (t['prev_choice'].to_numpy() + 1)


def timecourse_history(sess, rng):
    pop = [dict(s, t=add_history(s['t'])) for s in sess if len(s['u']) >= C.MIN_UNITS_POPULATION]
    starts = np.round(np.arange(fu.TC_RANGE[0], fu.TC_RANGE[1] - fu.TC_WIDTH + 1e-9, fu.TC_STEP), 3)
    centers = starts + fu.TC_WIDTH / 2
    Xs = [[np.sqrt(K.window_counts(s['b'], ('firstMove', a, a + fu.TC_WIDTH)).T) for a in starts] for s in pop]

    def run(i, tb, n, n_sub):
        y = tb['choice'].to_numpy()[:n]
        st = strata_hist(tb.iloc[:n])
        return [fu.matched_acc(X[:n], y, st, rng, n_sub) for X in Xs[i]]

    real = np.array([run(i, s['t'], len(s['t']), N_SUB) for i, s in enumerate(pop)])
    imp = np.zeros((N_ROUNDS_IMP, len(pop), len(starts)))
    for r in range(N_ROUNDS_IMP):
        perm = fu.derangement(len(pop), rng)
        for i in range(len(pop)):
            j = perm[i]
            n = min(len(pop[i]['t']), len(pop[j]['t']))
            imp[r, i] = run(i, pop[j]['t'], n, N_SUB_IMP)
        print(f'  history-matched imposter round {r + 1}/{N_ROUNDS_IMP}')
    return centers, real, imp, [s['meta']['subject'] for s in pop]


def fdr_onset(centers, diff):
    p = np.array([stats.wilcoxon(diff[:, j], alternative='greater').pvalue for j in range(diff.shape[1])])
    q = K.fdr_bh(p)
    for j in range(len(q) - 1):
        if q[j] < C.ALPHA and q[j + 1] < C.ALPHA:
            return centers[j], q
    return np.nan, q


def halfmax_latency(centers, d):
    m = np.nanmax(d)
    if not np.isfinite(m) or m <= 0:
        return np.nan
    for j in range(len(d)):
        if d[j] >= m / 2:
            if j == 0:
                return centers[0]
            return centers[j - 1] + (m / 2 - d[j - 1]) / (d[j] - d[j - 1]) * (centers[j] - centers[j - 1])
    return np.nan


# ----------------------------------------------------------------------------
# C power-matched feedback control
# ----------------------------------------------------------------------------
def feedback_power(sess, rng):
    terms = ('choice', 'stimulus', 'speed')
    late_cnt = {tm: [0, 0] for tm in terms}
    rand_cnt = {tm: np.zeros((N_REP_FB, 2)) for tm in terms}

    def frac_sig(t, Y, tm):
        X, _, idx = K.design_matrix(t, 'move')
        F, *_r = glm.nested_F(X, Y, idx[tm])
        p = stats.f.sf(F, len(idx[tm]), X.shape[0] - X.shape[1])
        return int(np.nansum(p < C.ALPHA)), Y.shape[1]

    for s in sess:
        t = s['t']
        Y = np.sqrt(K.window_counts(s['b'], 'move').T)
        late = np.where((t['feedback_times'] - t['firstMovement_times']).to_numpy() > fu.FEEDBACK_MARGIN)[0]
        if len(late) < 50:
            continue
        for tm in terms:
            k, n = frac_sig(t.iloc[late].reset_index(drop=True), Y[late], tm)
            late_cnt[tm][0] += k
            late_cnt[tm][1] += n
        for r in range(N_REP_FB):
            sub = np.sort(rng.choice(len(t), len(late), replace=False))
            for tm in terms:
                k, n = frac_sig(t.iloc[sub].reset_index(drop=True), Y[sub], tm)
                rand_cnt[tm][r] += (k, n)
    return late_cnt, rand_cnt


# ----------------------------------------------------------------------------
def compare_regions():
    files = {}
    for d in [C.PROJECT_DIR / 'results'] + sorted(C.PROJECT_DIR.glob('results_*')):
        f = d / 'onset_sessions.npz'
        if f.exists():
            files['ZI' if d.name == 'results' else d.name.split('_', 1)[1]] = f
    if len(files) < 2:
        return ['--- D  Onset comparison: run this script for at least two regions (e.g. ZI and MOs)']
    rng = np.random.default_rng(C.RANDOM_SEED + 100)
    boots, lines = {}, ['--- D  Half-max latency of choice information (history+speed+block matched, minus imposter)']
    for reg, f in files.items():
        z = np.load(f)
        d = z['real'] - z['imp_mean']
        cen = z['centers'] * 1000
        lat = halfmax_latency(cen, d.mean(0))
        bs = np.array([halfmax_latency(cen, d[rng.integers(0, len(d), len(d))].mean(0)) for _ in range(N_BOOT)])
        boots[reg] = bs
        lines.append(f'{reg:5s}: {lat:.0f} ms (bootstrap 95% CI {np.nanpercentile(bs, 2.5):.0f} to {np.nanpercentile(bs, 97.5):.0f}; '
                     f'{len(d)} sessions)')
    if 'ZI' in boots:
        for reg in boots:
            if reg == 'ZI':
                continue
            diff = boots['ZI'] - boots[reg]
            diff = diff[np.isfinite(diff)]
            p = 2 * min((diff >= 0).mean(), (diff <= 0).mean())
            lines.append(f'ZI - {reg}: {np.median(diff):+.0f} ms (95% CI {np.percentile(diff, 2.5):+.0f} to '
                         f'{np.percentile(diff, 97.5):+.0f}; bootstrap p={max(p, 1 / len(diff)):.3g}); negative = ZI earlier')
    return lines


def main():
    rng = np.random.default_rng(C.RANDOM_SEED + 10)
    plt = K.setup_matplotlib()
    sess = fu.load_sessions()
    print(f'{len(sess)} insertions ({C.REGION})')
    L = [f'=== Reversal, history and feedback controls ({C.REGION}) ===']

    # A
    print('A reversal-aligned decoding ...')
    curves, wheel, info = reversal_analysis(sess)
    info.to_csv(C.RESULTS_DIR / 'reversal_sessions.csv', index=False)
    taus_ms = TAU * C.BIN * 1000
    pd.DataFrame(curves, columns=[f'tau_{int(x)}ms' for x in taus_ms]).to_csv(C.RESULTS_DIR / 'reversal_curves.csv', index=False)
    L.append(f"--- A  CoM trials with a detected reversal (<= {REV_MAX * 1000:.0f} ms): {info['n_com_with_reversal'].sum()} "
             f"in {len(curves)} sessions; median reversal {info['median_reversal_ms'].median():.0f} ms after movement onset")
    cross = boot_ci = None
    if len(curves) >= 5:
        m = np.nanmean(curves, 0)
        cross = crossing(m, taus_ms)
        bs = np.array([crossing(np.nanmean(curves[rng.integers(0, len(curves), len(curves))], 0), taus_ms)
                       for _ in range(N_BOOT)])
        boot_ci = np.nanpercentile(bs, [2.5, 97.5]) if np.isfinite(bs).any() else np.array([np.nan, np.nan])
        pre = (taus_ms >= -100) & (taus_ms <= -20)
        early = (taus_ms >= -300) & (taus_ms <= -200)
        v_pre, v_early = np.nanmean(curves[:, pre], 1), np.nanmean(curves[:, early], 1)
        L.append(f'P(final choice) -300..-200 ms before reversal: {np.nanmean(v_early):.3f} '
                 f'(Wilcoxon vs .5 p={stats.wilcoxon(v_early - 0.5).pvalue:.3g})')
        L.append(f'P(final choice) -100..-20 ms before reversal: {np.nanmean(v_pre):.3f} '
                 f'(Wilcoxon vs .5 p={stats.wilcoxon(v_pre - 0.5).pvalue:.3g})')
        L.append(f'decoder crosses 0.5 at {cross:.0f} ms relative to wheel reversal '
                 f'(bootstrap 95% CI {boot_ci[0]:.0f} to {boot_ci[1]:.0f}; '
                 f'{np.mean(bs[np.isfinite(bs)] < 0):.0%} of bootstraps before reversal; '
                 f'{np.mean(~np.isfinite(bs)):.0%} of bootstraps without a switch)')
    else:
        L.append('not enough sessions with reversals')

    # B
    print('B trial history ...')
    hg, stay = history_glm(sess)
    L.append(f'--- B  Trial history: P(repeat previous choice) = {stay:.3f}')
    for w in ('iti', 'premove'):
        a, h, hh = hg[(w, 'base', 'choice')], hg[(w, 'history', 'choice')], hg[(w, 'history', 'history')]
        L.append(f'{w:8s}: choice units {a[0] / a[1]:.1%} -> with history regressors {h[0] / h[1]:.1%}; '
                 f'units encoding history (3 df) {hh[0] / hh[1]:.1%}  [parametric p<0.05]')
    pd.DataFrame([dict(window=k[0], model=k[1], term=k[2], n_sig=v[0], n_units=v[1]) for k, v in hg.items()]) \
        .to_csv(C.RESULTS_DIR / 'history_glm.csv', index=False)
    print('B time-resolved decoding, history-matched vs imposter ...')
    centers, real, imp, subj = timecourse_history(sess, rng)
    imp_mean = imp.mean(0)
    np.savez(C.RESULTS_DIR / 'onset_sessions.npz', centers=centers, real=real, imp_mean=imp_mean)
    onset, q = fdr_onset(centers, real - imp_mean)
    old = C.RESULTS_DIR / 'followup_timecourse_vs_imposter.csv'
    old_on = np.nan
    if old.exists():
        o = pd.read_csv(old)
        for j in range(len(o) - 1):
            if o['q'].iloc[j] < C.ALPHA and o['q'].iloc[j + 1] < C.ALPHA:
                old_on = o['t_center'].iloc[j]
                break
    L.append(f'onset (speed x block x previous-choice matched, vs imposter, FDR): '
             f'{onset * 1000:.0f} ms' if np.isfinite(onset) else 'onset with history matching: none')
    L.append(f'   (without history matching, step 09: {old_on * 1000:.0f} ms)' if np.isfinite(old_on) else '')
    L.append(f'half-max latency: {halfmax_latency(centers * 1000, (real - imp_mean).mean(0)):.0f} ms')
    pd.DataFrame(dict(t_center=centers, real=real.mean(0), imposter=imp_mean.mean(0), q=q)) \
        .to_csv(C.RESULTS_DIR / 'history_timecourse.csv', index=False)

    # C
    print('C power-matched feedback control ...')
    late_cnt, rand_cnt = feedback_power(sess, rng)
    L.append(f'--- C  Feedback-free trials vs equally sized random subsets (move window, parametric p<0.05)')
    fb_plot = {}
    for tm in late_cnt:
        lf = late_cnt[tm][0] / max(late_cnt[tm][1], 1)
        rf = rand_cnt[tm][:, 0] / np.maximum(rand_cnt[tm][:, 1], 1)
        p = (1 + (rf <= lf).sum()) / (1 + len(rf))
        fb_plot[tm] = (lf, rf)
        L.append(f'{tm:9s}: feedback-free {lf:.1%} | random subsets {rf.mean():.1%} (range {rf.min():.1%}-{rf.max():.1%}); '
                 f'P(random <= feedback-free) = {p:.2f}')

    # D
    L += compare_regions()
    C.write_summary('summary_10_reversal_history', [x for x in L if x])

    # ---------------- figure ----------------
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.8))
    fig.subplots_adjust(hspace=0.7, wspace=0.55)
    ax = axs[0, 0]
    if len(curves):
        m, se = np.nanmean(curves, 0), np.nanstd(curves, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(curves), 0))
        ax.plot(taus_ms, np.nanmean(wheel, 0), color='0.6', lw=1, label='wheel in final direction')
        ax.plot(taus_ms, m, color=K.COL_CHOICE, lw=1.4, label='ZI decoder P(final)' if C.REGION == 'ZI' else 'decoder P(final)')
        ax.fill_between(taus_ms, m - se, m + se, color=K.COL_CHOICE, alpha=0.25, lw=0)
        if cross is not None and np.isfinite(cross) and np.all(np.isfinite(boot_ci)):
            ax.errorbar(cross, 0.5, xerr=[[cross - boot_ci[0]], [boot_ci[1] - cross]], fmt='o', ms=3,
                        color=K.COL_CHOICE, lw=1, capsize=2, label='crossing (95% CI)')
    ax.axhline(0.5, color='k', ls=':', lw=0.8)
    ax.axvline(0, color='k', lw=0.6)
    ax.set(xlabel='Time from wheel reversal (ms)', ylabel='Probability', title='A  Change of mind', ylim=(0, 1))
    ax.legend(frameon=False, fontsize=5.5, loc='upper left')

    ax = axs[0, 1]
    if len(curves):
        pre = (taus_ms >= -100) & (taus_ms <= -20)
        post = (taus_ms >= 40) & (taus_ms <= 120)
        early = (taus_ms >= -300) & (taus_ms <= -200)
        vals = [np.nanmean(curves[:, sel], 1) for sel in (early, pre, post)]
        for v_ in np.column_stack(vals):
            ax.plot(range(3), v_, color='0.8', lw=0.5)
        ax.plot(range(3), [np.nanmean(v_) for v_ in vals], color=K.COL_CHOICE, marker='o', lw=1.4)
    ax.axhline(0.5, color='k', ls=':', lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['-300..-200', '-100..-20', '+40..+120'], fontsize=6)
    ax.set(xlabel='ms from reversal', ylabel='P(final choice)', title='B  Per session')

    ax = axs[0, 2]
    xx = np.arange(2)
    base = [hg[(w, 'base', 'choice')][0] / hg[(w, 'base', 'choice')][1] for w in ('iti', 'premove')]
    hist = [hg[(w, 'history', 'choice')][0] / hg[(w, 'history', 'choice')][1] for w in ('iti', 'premove')]
    ax.bar(xx - 0.2, base, 0.38, color='0.6', label='base model')
    ax.bar(xx + 0.2, hist, 0.38, color=K.COL_CHOICE, label='+ trial history')
    ax.axhline(C.ALPHA, color='k', ls=':', lw=0.8)
    ax.set_xticks(xx)
    ax.set_xticklabels(['iti', 'premove'])
    ax.set(ylabel='Choice units (p<0.05)', title='C  History control')
    ax.legend(frameon=False, fontsize=6)

    ax = axs[1, 0]
    rm = real.mean(0)
    ax.fill_between(centers * 1000, np.percentile(imp.mean(1), 2.5, 0), np.percentile(imp.mean(1), 97.5, 0),
                    color='0.85', label='imposter 95%')
    ax.plot(centers * 1000, rm, color=K.COL_CHOICE, label='real')
    if np.isfinite(onset):
        ax.axvline(onset * 1000, color=K.COL_CHOICE, ls='--', lw=0.8)
    ax.axvline(0, color='k', lw=0.6)
    ax.set(xlabel='Time from movement onset (ms)', ylabel='Choice decoding', title='D  + previous-choice matched')
    ax.legend(frameon=False, fontsize=6, loc='upper left')

    ax = axs[1, 1]
    for xi, tm in enumerate(fb_plot):
        lf, rf = fb_plot[tm]
        ax.scatter(np.full(len(rf), xi) + rng.uniform(-0.12, 0.12, len(rf)), rf, s=5, color='0.6')
        ax.plot([xi - 0.25, xi + 0.25], [lf, lf], color=K.COL_CHOICE, lw=2)
    ax.set_xticks(range(len(fb_plot)))
    ax.set_xticklabels(list(fb_plot))
    ax.set(ylabel='Fraction of units (move)', title='E  Feedback-free (line)\nvs random subsets (dots)')

    ax = axs[1, 2]
    ax.axis('off')
    ax.text(0, 1, '\n'.join(compare_regions()).replace('--- D  ', 'F  '), va='top', fontsize=5.5, family='monospace',
            wrap=True)
    C.save_fig(fig, 'step10_reversal_history')


if __name__ == '__main__':
    main()
