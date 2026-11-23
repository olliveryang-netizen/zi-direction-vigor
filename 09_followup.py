"""
Step 09 - Change of mind, lateralisation, feedback control and time-resolved decoding
(paper Figures 4A and 6E). Uses existing processed data only.

A  CHANGE OF MIND - choice vs movement DIRECTION.
   In the IBL task the reported choice is the wheel direction, so speed controls cannot separate
   "decision" from "directional motor plan". On some trials the INITIAL wheel movement (first 60 ms
   after movement onset) goes opposite to the FINAL choice. A choice decoder is trained on consistent
   trials and tested on these change-of-mind (CoM) trials:
     agreement with final choice  > 0.5  -> ZI activity follows the eventual choice
     agreement with final choice  < 0.5  -> ZI activity follows the initial movement direction
   Windows: premove (-160..0 ms, before the initial movement) and early movement (0..100 ms).
   The sign mapping between wheel velocity and choice is estimated per session (robust to wheel polarity).

B  LATERALISATION - do choice-selective units prefer the choice that reports a contralateral stimulus?
   Choice code <-> stimulus side is estimated per session from behaviour; hemisphere from unit ML
   coordinate (IBL: x < 0 = left). Also a modulation index for ALL units.

C  FEEDBACK CONTAMINATION of the move window (0..200 ms). Feedback (reward / error tone) can arrive
   inside this window on fast trials. The move-window GLM is repeated using only trials whose feedback
   arrives > 250 ms after movement onset.

D  TIME-RESOLVED DECODING AGAINST THE IMPOSTER NULL, with trials matched for BOTH wheel speed AND block
   prior (removes choice-prior correlation). Onset = first window where real > imposter (session-level
   Wilcoxon, FDR across windows) for >= 2 consecutive windows. Plus premove block+speed-matched decoding
   vs imposter distribution.

Outputs: results/followup_*.csv, results/summary_09_followup.txt, figures/step09_followup.(png|pdf)
Run:     python 09_followup.py        (about 10-20 min)
"""
import importlib

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score

import common as K
import config as C

glm = importlib.import_module('03_single_unit_glm')
dec = importlib.import_module('04_decoding')

INIT_WIN = (0.0, 0.06)            # initial movement direction window (s after movement onset)
COM_WINDOWS = {'premove': ('firstMove', -0.16, 0.0), 'early move': ('firstMove', 0.0, 0.10)}
FEEDBACK_MARGIN = 0.25            # keep trials with feedback > 250 ms after movement onset
N_SUB = 5                         # matched resamples (real data)
N_SUB_IMP = 2                     # matched resamples per imposter pairing
N_ROUNDS_IMP_TC = 10              # imposter pairings for time-resolved decoding
N_ROUNDS_IMP_PM = 50              # imposter pairings for premove block+speed-matched decoding
TC_WIDTH, TC_STEP, TC_RANGE = 0.10, 0.04, (-0.40, 0.40)


# ----------------------------------------------------------------------------
def load_sessions():
    out = []
    for path in K.list_insertions():
        meta, t, u, b = K.load_insertion(path)
        ok = K.valid_trials(t)
        if ok.sum() < 100:
            continue
        t = t[ok].reset_index(drop=True)
        bb = {k: (v[:, ok] if v.ndim == 3 else (v[ok] if v.ndim == 2 else v)) for k, v in b.items()}
        out.append(dict(meta=meta, t=t, u=u, b=bb))
    return out


def strata(t):
    """Joint speed-quantile x block-prior strata for matching."""
    sb = pd.qcut(t['peak_speed'], C.N_SPEED_BINS, labels=False, duplicates='drop').to_numpy()
    blk = np.round(t['probabilityLeft'].to_numpy(), 1)
    return sb * 10 + (blk * 10).astype(int)


def matched_acc(X, y, st, rng, n_sub):
    return np.nanmean([dec.cv_balacc(X[i], y[i], rng) for i in (dec.matched_idx(y, st, rng) for _ in range(n_sub))])


# ----------------------------------------------------------------------------
# A change of mind
# ----------------------------------------------------------------------------
def change_of_mind(sess):
    _, cent = K.bin_centers()
    iw = (cent > INIT_WIN[0]) & (cent <= INIT_WIN[1])
    rows, trial_rows = [], []
    for s in sess:
        t, b = s['t'], s['b']
        init_v = b['wheel_vel_fm'][:, iw].mean(1)
        y = t['choice'].to_numpy()
        strong = np.abs(init_v) > np.percentile(np.abs(init_v), 10)
        sign_map = np.sign(np.sum(y[strong] * np.sign(init_v[strong]))) or 1.0
        init_dir = (sign_map * np.sign(init_v)).astype(int)            # initial direction in choice code
        com = strong & (init_dir != y)
        rows.append(dict(pid=s['meta']['pid'], subject=s['meta']['subject'], n_units=len(s['u']),
                         n_trials=len(t), n_com=int(com.sum()), frac_com=com.mean()))
        if len(s['u']) < C.MIN_UNITS_POPULATION or com.sum() < 3:
            continue
        train = strong & ~com
        for wname, win in COM_WINDOWS.items():
            X = np.sqrt(K.window_counts(b, win).T)
            if len(np.unique(y[train])) < 2:
                continue
            m = dec.clf().fit(X[train], y[train])
            pred = m.predict(X[com])
            for p_, yy, ii in zip(pred, y[com], init_dir[com]):
                trial_rows.append(dict(pid=s['meta']['pid'], subject=s['meta']['subject'], window=wname,
                                       agree_final=int(p_ == yy), agree_initial=int(p_ == ii)))
    return pd.DataFrame(rows), pd.DataFrame(trial_rows, columns=['pid', 'subject', 'window', 'agree_final', 'agree_initial'])


def cluster_boot_mean(df, col, rng, n=2000):
    pids = df['pid'].unique()
    g = {p: df.loc[df['pid'] == p, col].to_numpy() for p in pids}
    bs = [np.mean(np.concatenate([g[p] for p in rng.choice(pids, len(pids))])) for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])


# ----------------------------------------------------------------------------
# B lateralisation
# ----------------------------------------------------------------------------
def lateralisation(sess):
    gu = pd.read_csv(C.RESULTS_DIR / 'glm_units.csv')
    ch = gu[(gu['window'] == 'premove') & (gu['term'] == 'choice')].copy()
    right_choice = {}
    mi_rows = []
    for s in sess:
        t = s['t']
        rc = np.sign(t.loc[t['signed_contrast'] > 0, 'choice'].mean())       # choice reporting a RIGHT stimulus
        right_choice[s['meta']['pid']] = rc
        X = K.window_counts(s['b'], 'premove')                                 # units x trials
        for i in range(X.shape[0]):
            x = s['u']['x'].iloc[i]
            contra = rc if x < 0 else -rc                                       # left hemisphere -> right stimulus is contra
            rcon = X[i, t['choice'].to_numpy() == contra].mean()
            rips = X[i, t['choice'].to_numpy() == -contra].mean()
            mi = (rcon - rips) / (rcon + rips) if (rcon + rips) > 0 else np.nan
            mi_rows.append(dict(pid=s['meta']['pid'], subject=s['meta']['subject'],
                                unit_index=int(s['u']['unit_index'].iloc[i]),
                                hemisphere='left' if x < 0 else 'right', mi_contra=mi))
    ch['right_choice'] = ch['pid'].map(right_choice)
    ch['hemisphere'] = np.where(ch['x'] < 0, 'left', 'right')
    ch['contra_choice'] = np.where(ch['hemisphere'] == 'left', ch['right_choice'], -ch['right_choice'])
    ch['preferred'] = np.sign(ch['beta'])
    ch['prefers_contra'] = (ch['preferred'] == ch['contra_choice']).astype(int)
    return ch, pd.DataFrame(mi_rows)


# ----------------------------------------------------------------------------
# C feedback contamination
# ----------------------------------------------------------------------------
def feedback_control(sess):
    res = {('all', term): [0, 0] for term in ('choice', 'stimulus', 'speed')}
    res.update({('late_fb', term): [0, 0] for term in ('choice', 'stimulus', 'speed')})
    n_kept = []
    for s in sess:
        t = s['t']
        Yall = np.sqrt(K.window_counts(s['b'], 'move').T)
        late = (t['feedback_times'] - t['firstMovement_times']).to_numpy() > FEEDBACK_MARGIN
        n_kept.append(late.mean())
        for tag, sel in (('all', np.ones(len(t), bool)), ('late_fb', late)):
            if sel.sum() < 50:
                continue
            tt = t[sel].reset_index(drop=True)
            X, names, idx = K.design_matrix(tt, 'move')
            Y = Yall[sel]
            for term in ('choice', 'stimulus', 'speed'):
                F, *_r = glm.nested_F(X, Y, idx[term])
                p = stats.f.sf(F, len(idx[term]), X.shape[0] - X.shape[1])
                res[(tag, term)][0] += int(np.nansum(p < C.ALPHA))
                res[(tag, term)][1] += Y.shape[1]
    return res, float(np.mean(n_kept))


# ----------------------------------------------------------------------------
# D time-resolved decoding vs imposter (speed + block matched)
# ----------------------------------------------------------------------------
def tc_session(Xs_list, t_beh, n, rng, n_sub):
    y = t_beh['choice'].to_numpy()[:n]
    st = strata(t_beh.iloc[:n])
    return [matched_acc(X[:n], y, st, rng, n_sub) for X in Xs_list]


def timecourse_vs_imposter(sess, rng):
    pop = [s for s in sess if len(s['u']) >= C.MIN_UNITS_POPULATION]
    starts = np.round(np.arange(TC_RANGE[0], TC_RANGE[1] - TC_WIDTH + 1e-9, TC_STEP), 3)
    centers = starts + TC_WIDTH / 2
    Xs = [[np.sqrt(K.window_counts(s['b'], ('firstMove', a, a + TC_WIDTH)).T) for a in starts] for s in pop]
    Xpm = [np.sqrt(K.window_counts(s['b'], 'premove').T) for s in pop]
    real_tc = np.array([tc_session(Xs[i], s['t'], len(s['t']), rng, N_SUB) for i, s in enumerate(pop)])
    real_pm = np.array([matched_acc(Xpm[i], s['t']['choice'].to_numpy(), strata(s['t']), rng, N_SUB)
                        for i, s in enumerate(pop)])
    imp_tc = np.zeros((N_ROUNDS_IMP_TC, len(pop), len(starts)))
    imp_pm = np.zeros((N_ROUNDS_IMP_PM, len(pop)))
    for r in range(max(N_ROUNDS_IMP_TC, N_ROUNDS_IMP_PM)):
        perm = derangement(len(pop), rng)
        for i in range(len(pop)):
            j = perm[i]
            n = min(len(pop[i]['t']), len(pop[j]['t']))
            tb = pop[j]['t']
            if r < N_ROUNDS_IMP_TC:
                imp_tc[r, i] = tc_session(Xs[i], tb, n, rng, N_SUB_IMP)
            if r < N_ROUNDS_IMP_PM:
                imp_pm[r, i] = matched_acc(Xpm[i][:n], tb['choice'].to_numpy()[:n], strata(tb.iloc[:n]), rng, N_SUB_IMP)
        print(f'  imposter round {r + 1}')
    return centers, real_tc, imp_tc, real_pm, imp_pm, [s['meta']['subject'] for s in pop]


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


# ----------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(C.RANDOM_SEED + 9)
    plt = K.setup_matplotlib()
    sess = load_sessions()
    print(f'{len(sess)} insertions loaded')
    L = ['=== Follow-up analyses (%s) ===' % C.REGION]

    # A
    print('A change of mind ...')
    com_sess, com_tr = change_of_mind(sess)
    com_sess.to_csv(C.RESULTS_DIR / 'followup_com_sessions.csv', index=False)
    com_tr.to_csv(C.RESULTS_DIR / 'followup_com_trials.csv', index=False)
    L.append(f"--- A  Change of mind: CoM trials {com_sess['n_com'].sum()} / {com_sess['n_trials'].sum()} "
             f"({com_sess['n_com'].sum() / com_sess['n_trials'].sum():.1%}); median per session {com_sess['frac_com'].median():.1%}")
    com_summ = {}
    for w in COM_WINDOWS:
        d = com_tr[com_tr['window'] == w] if not com_tr.empty else com_tr
        if d.empty:
            L.append(f'{w}: not enough CoM trials')
            continue
        k, n = int(d['agree_final'].sum()), len(d)
        p = stats.binomtest(k, n, 0.5).pvalue
        ci = cluster_boot_mean(d, 'agree_final', rng)
        per_s = d.groupby('pid')['agree_final'].mean()
        pw = stats.wilcoxon(per_s - 0.5).pvalue if len(per_s) >= 5 else np.nan
        com_summ[w] = (k / n, ci)
        L.append(f'{w:10s}: decoder agrees with FINAL choice on {k}/{n} CoM trials = {k / n:.3f} '
                 f'(session-bootstrap 95% CI {ci[0]:.3f}-{ci[1]:.3f}; binomial p={p:.3g}; '
                 f'{len(per_s)} sessions, Wilcoxon p={pw:.3g}); agrees with INITIAL direction {d["agree_initial"].mean():.3f}')

    # B
    print('B lateralisation ...')
    ch, mi = lateralisation(sess)
    ch.to_csv(C.RESULTS_DIR / 'followup_lateralisation_units.csv', index=False)
    mi.to_csv(C.RESULTS_DIR / 'followup_modulation_index.csv', index=False)
    sig = ch[ch['p_perm'] < C.ALPHA]
    k, n = int(sig['prefers_contra'].sum()), len(sig)
    pb = stats.binomtest(k, n, 0.5).pvalue if n else np.nan
    pm = sig.groupby('subject')['prefers_contra'].mean()
    L.append('--- B  Lateralisation (premove choice units, p_perm<0.05)')
    L.append(f'units by hemisphere: ' + ', '.join(f'{h}: {c}' for h, c in ch['hemisphere'].value_counts().items()))
    L.append(f'prefer the choice reporting a CONTRALATERAL stimulus: {k}/{n} = {k / max(n, 1):.1%} (binomial p={pb:.3g}); '
             f'per-mouse median {pm.median():.2f} ({len(pm)} mice)')
    mv = mi['mi_contra'].dropna()
    L.append(f'all units, contra-ipsi modulation index: median {mv.median():+.3f}, Wilcoxon p={stats.wilcoxon(mv).pvalue:.3g} (n={len(mv)})')

    # C
    print('C feedback control ...')
    fb, frac_kept = feedback_control(sess)
    L.append(f'--- C  Move window with feedback > {FEEDBACK_MARGIN * 1000:.0f} ms after movement onset '
             f'(keeps {frac_kept:.0%} of trials); parametric p<0.05')
    for term in ('choice', 'stimulus', 'speed'):
        a, b_ = fb[('all', term)], fb[('late_fb', term)]
        L.append(f'{term:9s}: all trials {a[0] / max(a[1], 1):.1%}  ->  feedback-free {b_[0] / max(b_[1], 1):.1%}')

    # D
    print('D time-resolved decoding vs imposter (this part takes longest) ...')
    centers, real_tc, imp_tc, real_pm, imp_pm, subj = timecourse_vs_imposter(sess, rng)
    imp_mean = imp_tc.mean(0)                              # sessions x windows
    diff = real_tc - imp_mean
    p_w = np.array([stats.wilcoxon(diff[:, j], alternative='greater').pvalue for j in range(diff.shape[1])])
    q_w = K.fdr_bh(p_w)
    sigw = q_w < C.ALPHA
    onset = np.nan
    for j in range(len(sigw) - 1):
        if sigw[j] and sigw[j + 1]:
            onset = centers[j]
            break
    pd.DataFrame(dict(t_center=centers, real_mean=real_tc.mean(0), imposter_mean=imp_mean.mean(0),
                      p=p_w, q=q_w)).to_csv(C.RESULTS_DIR / 'followup_timecourse_vs_imposter.csv', index=False)
    imp_pm_mean = imp_pm.mean(1)                           # one value per pairing round
    p_pm = (1 + (imp_pm_mean >= real_pm.mean()).sum()) / (1 + len(imp_pm_mean))
    ms = pd.DataFrame(dict(subject=subj, d=real_pm - imp_pm.mean(0))).groupby('subject')['d'].mean()
    L.append('--- D  Decoding matched for speed AND block prior, vs imposter sessions')
    L.append(f'premove: real {real_pm.mean():.3f} | imposter mean {imp_pm_mean.mean():.3f} '
             f'(95th pct {np.percentile(imp_pm_mean, 95):.3f}) | p = {p_pm:.3g}; '
             f'mouse-level real-imposter > 0: Wilcoxon p={stats.wilcoxon(ms, alternative="greater").pvalue:.3g} ({len(ms)} mice)')
    L.append(f'time-resolved onset (real > imposter, FDR, 2 consecutive 100-ms windows): '
             f'{onset * 1000:.0f} ms from movement onset (window centre)' if np.isfinite(onset) else
             'time-resolved: no robust onset')
    C.write_summary('summary_09_followup', L)

    # ---------------- figure ----------------
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.8))
    fig.subplots_adjust(hspace=0.7, wspace=0.55)
    ax = axs[0, 0]
    for xi, w in enumerate(COM_WINDOWS):
        if w in com_summ:
            m, ci = com_summ[w]
            ax.bar(xi, m - 0.5, bottom=0.5, color=K.COL_CHOICE, width=0.6)
            ax.errorbar(xi, m, yerr=[[m - ci[0]], [ci[1] - m]], color='k', lw=0.8)
    ax.axhline(0.5, color='k', ls=':', lw=0.8)
    ax.set_xticks(range(len(COM_WINDOWS)))
    ax.set_xticklabels(list(COM_WINDOWS))
    ax.set(ylabel='Agreement with final choice\n(change-of-mind trials)', title='A  Choice vs initial direction',
           ylim=(0.2, 0.8))
    ax.text(0.02, 0.03, '<0.5: follows initial movement', transform=ax.transAxes, fontsize=5.5, color='0.4')

    ax = axs[0, 1]
    if n:
        ax.bar([0, 1], [k / n, 1 - k / n], color=[K.COL_CHOICE, '0.7'], width=0.6)
        ax.axhline(0.5, color='k', ls=':', lw=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['contra', 'ipsi'])
    ax.set(ylabel='Fraction of choice units', title=f'B  Preferred choice side\n(p={pb:.2g})', ylim=(0, 1))

    ax = axs[0, 2]
    ax.hist(mv.clip(-1, 1), bins=30, color='0.6')
    ax.axvline(0, color='k', lw=0.8)
    ax.axvline(mv.median(), color=K.COL_CHOICE, lw=1.5)
    ax.set(xlabel='Modulation index (contra - ipsi)', ylabel='Units', title='C  All units, premove')

    ax = axs[1, 0]
    terms = ('choice', 'stimulus', 'speed')
    a = [fb[('all', tm)][0] / max(fb[('all', tm)][1], 1) for tm in terms]
    b_ = [fb[('late_fb', tm)][0] / max(fb[('late_fb', tm)][1], 1) for tm in terms]
    xx = np.arange(3)
    ax.bar(xx - 0.2, a, 0.38, color='0.6', label='all trials')
    ax.bar(xx + 0.2, b_, 0.38, color=K.COL_CHOICE, label='feedback-free')
    ax.set_xticks(xx)
    ax.set_xticklabels(terms)
    ax.set(ylabel='Fraction of units (move window)', title='D  Feedback control')
    ax.legend(frameon=False, fontsize=6)

    ax = axs[1, 1]
    rm, rs = real_tc.mean(0), real_tc.std(0, ddof=1) / np.sqrt(len(real_tc))
    im = imp_tc.mean(1)                                     # rounds x windows
    ax.fill_between(centers * 1000, np.percentile(im, 2.5, 0), np.percentile(im, 97.5, 0), color='0.85', label='imposter 95%')
    ax.plot(centers * 1000, rm, color=K.COL_CHOICE, label='real')
    ax.fill_between(centers * 1000, rm - rs, rm + rs, color=K.COL_CHOICE, alpha=0.25, lw=0)
    if np.isfinite(onset):
        ax.axvline(onset * 1000, color=K.COL_CHOICE, ls='--', lw=0.8)
    ax.axvline(0, color='k', lw=0.6)
    ax.set(xlabel='Time from movement onset (ms)', ylabel='Choice decoding\n(speed + block matched)',
           title='E  Onset vs imposter null')
    ax.legend(frameon=False, fontsize=6, loc='upper left')

    ax = axs[1, 2]
    ax.hist(imp_pm_mean, bins=20, color='0.75')
    ax.axvline(real_pm.mean(), color=K.COL_CHOICE, lw=1.5)
    ax.set(xlabel='Mean accuracy (premove)', ylabel='Imposter pairings', title=f'F  Block+speed matched\np={p_pm:.2g}')
    C.save_fig(fig, 'step09_followup')


if __name__ == '__main__':
    main()
