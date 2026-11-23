"""
Step 07 - Real-data null: imposter sessions (paper Figures 2F, 3B and S2).

The neural activity of each insertion is paired with the REAL behaviour (choice, stimulus, block prior,
wheel speed, RT, video motion energy, trial order) of a DIFFERENT session, truncated to the same number
of trials - the "imposter session" approach used by the IBL brain-wide map decoding analyses.
Neural statistics (firing rates, slow drifts, autocorrelation) and behavioural statistics (the real
choice-speed confound, block structure) are both preserved, but the true neural-behaviour link is broken.
Anything the pipeline detects under this pairing is a false positive.

Repeated for N_ROUNDS random pairings (each session never paired with itself):
  * fraction of units with choice (premove), speed (move) and prior (iti) effects in the full GLM (parametric F)
  * mean movement-matched choice decoding accuracy (sessions with >= MIN_UNITS_POPULATION units)
  * mean UNMATCHED choice decoding accuracy (shows what matching buys on real data)
The real (correctly paired) values are computed with the identical code and compared with the
imposter distributions: p = (1 + #imposter >= real) / (1 + N_ROUNDS).

Outputs: results/imposter_null.csv, results/summary_07_imposter.txt, figures/step07_imposter_null.(png|pdf)
"""
import importlib

import numpy as np
import pandas as pd
from scipy import stats

import common as K
import config as C

glm = importlib.import_module('03_single_unit_glm')
dec = importlib.import_module('04_decoding')

N_ROUNDS_GLM = 200
N_ROUNDS_DEC = 100
N_SUB = 5
TESTS = (('premove', 'choice'), ('move', 'speed'), ('iti', 'prior'))


def load_all():
    sess = []
    for path in K.list_insertions():
        meta, t, u, b = K.load_insertion(path)
        ok = K.valid_trials(t)
        if ok.sum() < 100:
            continue
        counts = {w: np.sqrt(K.window_counts(b, w)[:, ok].T) for w in C.WINDOWS}     # trials x units
        sess.append(dict(pid=meta['pid'], trials=t[ok].reset_index(drop=True), counts=counts, n_units=len(u)))
    return sess


def glm_fractions(pairs):
    """pairs: list of (neural session, behaviour session). Returns {test: (n_sig, n_units)}."""
    out = {k: [0, 0] for k in TESTS}
    for s_neu, s_beh in pairs:
        n = min(len(s_neu['trials']), len(s_beh['trials']))
        t = s_beh['trials'].iloc[:n].reset_index(drop=True)
        for w, term in TESTS:
            X, names, idx = K.design_matrix(t, w)
            Y = s_neu['counts'][w][:n]
            F, *_r = glm.nested_F(X, Y, idx[term])
            p = stats.f.sf(F, len(idx[term]), X.shape[0] - X.shape[1])
            out[(w, term)][0] += int(np.nansum(p < C.ALPHA))
            out[(w, term)][1] += Y.shape[1]
    return out


def decoding_means(pairs, rng):
    acc_m, acc_u = [], []
    for s_neu, s_beh in pairs:
        if s_neu['n_units'] < C.MIN_UNITS_POPULATION:
            continue
        n = min(len(s_neu['trials']), len(s_beh['trials']))
        t = s_beh['trials'].iloc[:n]
        X = s_neu['counts']['premove'][:n]
        y = t['choice'].to_numpy()
        sbin = pd.qcut(t['peak_speed'], C.N_SPEED_BINS, labels=False, duplicates='drop').to_numpy()
        acc_u.append(dec.cv_balacc(X, y, rng))
        acc_m.append(np.nanmean([dec.cv_balacc(X[i], y[i], rng)
                                 for i in (dec.matched_idx(y, sbin, rng) for _ in range(N_SUB))]))
    return np.nanmean(acc_m), np.nanmean(acc_u)


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def main():
    rng = np.random.default_rng(C.RANDOM_SEED + 7)
    plt = K.setup_matplotlib()
    sess = load_all()
    print(f'{len(sess)} insertions')
    real_pairs = [(s, s) for s in sess]
    real_glm = glm_fractions(real_pairs)
    real_dec = decoding_means(real_pairs, rng)

    rows = []
    for r in range(N_ROUNDS_GLM):
        perm = derangement(len(sess), rng)
        pairs = [(sess[i], sess[perm[i]]) for i in range(len(sess))]
        g = glm_fractions(pairs)
        row = {'round': r, **{f'{w}_{term}': g[(w, term)][0] / g[(w, term)][1] for w, term in TESTS}}
        if r < N_ROUNDS_DEC:
            row['acc_matched'], row['acc_unmatched'] = decoding_means(pairs, rng)
        rows.append(row)
        if (r + 1) % 20 == 0:
            print(f'round {r + 1}/{N_ROUNDS_GLM}')
    null = pd.DataFrame(rows)
    null.to_csv(C.RESULTS_DIR / 'imposter_null.csv', index=False)

    real = {f'{w}_{term}': real_glm[(w, term)][0] / real_glm[(w, term)][1] for w, term in TESTS}
    real['acc_matched'], real['acc_unmatched'] = real_dec
    labels = {'premove_choice': 'Choice units\n(premove)', 'move_speed': 'Speed units\n(move)',
              'iti_prior': 'Prior units\n(iti)', 'acc_matched': 'Choice decoding\n(matched)',
              'acc_unmatched': 'Choice decoding\n(unmatched)'}

    fig, axs = plt.subplots(1, 5, figsize=(7.2, 1.9))
    fig.subplots_adjust(wspace=0.5)
    lines = [f'=== Imposter-session null ({C.REGION}; {len(sess)} insertions; '
             f'{N_ROUNDS_GLM} GLM / {N_ROUNDS_DEC} decoding pairings) ===']
    for ax, key in zip(axs, labels):
        v = null[key].dropna().to_numpy()
        p = (1 + (v >= real[key]).sum()) / (1 + len(v))
        ax.hist(v, bins=25, color='0.75')
        ax.axvline(real[key], color=K.COL_CHOICE if 'choice' in key or 'acc' in key else K.COL_SPEED, lw=1.5)
        ax.set(title=f'{labels[key]}\np={p:.3g}', xlabel='fraction' if 'acc' not in key else 'accuracy')
        ax.title.set_fontsize(6.5)
        name = labels[key].replace('\n', ' ')
        lines.append(f'{name:30s}: real {real[key]:.3f} | imposter mean {v.mean():.3f} '
                     f'(95th pct {np.percentile(v, 95):.3f}) | p = {p:.3g}')
    axs[0].set_ylabel('Imposter pairings')
    C.save_fig(fig, 'step07_imposter_null')
    lines.append('Interpretation: the imposter mean is the realistic chance level on real data '
                 '(it can exceed 5% / 0.5 because of slow drifts and the choice-speed confound).')
    C.write_summary('summary_07_imposter', lines)


if __name__ == '__main__':
    main()
