"""
Step 01 - Download IBL data for every ZI insertion and save compact processed files.

For each probe insertion with good-quality ZI units this saves, in processed/<pid>/:
  trials.parquet  one row per included trial: task variables, wheel kinematics,
                  video motion energy in each analysis window
  units.parquet   one row per ZI unit: id, CCF coordinates (x=ML, y=AP, z=DV, metres from bregma), firing rate
  binned.npz      spike counts (units x trials x 20-ms bins) aligned to stimulus onset and to
                  movement onset (-0.5..0.5 s), plus wheel velocity aligned to movement onset
  meta.json       session identifiers

Raw IBL files are cached in ibl_cache/ inside the project folder.
The script can be stopped and restarted: finished insertions are skipped.

Run:  python 01_download_preprocess.py            (all insertions)
      python 01_download_preprocess.py --max 3    (quick test)
"""
import argparse
import json
import logging
import traceback

import numpy as np
import pandas as pd

import config as C

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger('01')


def zi_inventory(one):
    from brainwidemap import bwm_query, download_aggregate_tables
    from iblatlas.atlas import BrainRegions

    bwm_df = bwm_query(freeze=C.BWM_FREEZE)
    clus = pd.read_parquet(download_aggregate_tables(one, type='clusters'))
    clus = clus[clus['eid'].isin(bwm_df['eid'])].copy()
    clus['Beryl'] = BrainRegions().id2acronym(clus['atlas_id'], mapping='Beryl')
    zi = clus[(clus['Beryl'] == C.REGION) & (clus['label'] >= C.MIN_QC)]
    keys = ['eid', 'pid'] if 'pid' in zi.columns else ['eid']
    inv = zi.groupby(keys).size().rename('n_good_zi_units').reset_index()
    inv = inv.merge(bwm_df[['eid', 'pid', 'subject', 'lab', 'date', 'probe_name']], on=keys, how='left')
    inv = inv.sort_values('n_good_zi_units', ascending=False).reset_index(drop=True)
    inv.to_csv(C.RESULTS_DIR / f'{C.REGION}_insertions.csv', index=False)
    log.info(C.REGION + ' inventory: %d units, %d insertions, %d mice',
             inv['n_good_zi_units'].sum(), len(inv), inv['subject'].nunique())
    return inv


def window_mean(times, values, event_times, t0, t1):
    """Mean of a continuous signal in [event+t0, event+t1) for every trial."""
    cs = np.concatenate([[0], np.cumsum(np.nan_to_num(values))])
    a = np.searchsorted(times, event_times + t0)
    b = np.searchsorted(times, event_times + t1)
    n = b - a
    out = np.full(len(event_times), np.nan)
    good = n > 0
    out[good] = (cs[b[good]] - cs[a[good]]) / n[good]
    return out


def binned_counts(spike_times, spike_clusters, unit_ids, event_times, edges):
    """Counts (units x trials x bins) using searchsorted on sorted spike times."""
    out = np.zeros((len(unit_ids), len(event_times), len(edges) - 1), dtype=np.uint16)
    abs_edges = event_times[:, None] + edges[None, :]
    for i, u in enumerate(unit_ids):
        st = np.sort(spike_times[spike_clusters == u])
        idx = np.searchsorted(st, abs_edges)
        out[i] = np.diff(idx, axis=1)
    return out


def trial_features(trials, mask, wheel):
    t = trials.loc[mask.values].copy()
    t['trial_number'] = t.index.to_numpy()
    t = t[t['choice'] != 0]
    t['signed_contrast'] = t['contrastRight'].fillna(0) - t['contrastLeft'].fillna(0)
    t['correct'] = (t['feedbackType'] == 1).astype(int)
    t['rt'] = t['firstMovement_times'] - t['stimOn_times']
    end = t['response_times'].where(t['response_times'].notna(), t['feedback_times'])
    t['move_end'] = end

    wt = wheel['times'].to_numpy()
    wv = wheel['velocity'].to_numpy()
    wp = wheel['position'].to_numpy()
    i0 = np.searchsorted(wt, t['firstMovement_times'].to_numpy())
    i1 = np.searchsorted(wt, end.to_numpy())
    peak, mean, disp = (np.full(len(t), np.nan) for _ in range(3))
    for k, (a, b) in enumerate(zip(i0, i1)):
        if b > a:
            seg = np.abs(wv[a:b])
            peak[k], mean[k] = np.nanmax(seg), np.nanmean(seg)
            disp[k] = wp[min(b, len(wp) - 1)] - wp[a]
    t['peak_speed'], t['mean_speed'], t['wheel_disp'] = peak, mean, disp
    t['move_dur'] = end - t['firstMovement_times']
    return t.reset_index(drop=True)


def process_insertion(one, row):
    from brainwidemap import load_good_units, load_trials_and_mask
    from brainbox.io.one import SessionLoader
    from iblatlas.atlas import BrainRegions

    eid, pid, pname = row['eid'], row['pid'], row['probe_name']
    out = C.PROCESSED_DIR / pid
    if (out / 'meta.json').exists():
        log.info('skip (done) %s', pid)
        return
    out.mkdir(exist_ok=True)

    trials, mask = load_trials_and_mask(one, eid)
    sl = SessionLoader(one=one, eid=eid)
    sl.load_wheel()
    t = trial_features(trials, mask, sl.wheel)

    # video motion energy (optional: some sessions lack good video)
    # each camera is loaded separately, so a missing body camera does not discard the whisker (left) camera
    has_me = {}
    for cam, view, key, col in (('whisker', 'left', 'leftCamera', 'whiskerMotionEnergy'),
                                ('body', 'body', 'bodyCamera', 'bodyMotionEnergy')):
        try:
            sl.load_motion_energy(views=[view])
            me = sl.motion_energy.get(key)
            if me is None or len(me) == 0:
                continue
            times, vals = me['times'].to_numpy(), me[col].to_numpy()
            for w, (ev, t0, t1) in C.WINDOWS.items():
                ev_t = t['stimOn_times' if ev == 'stimOn' else 'firstMovement_times'].to_numpy()
                t[f'me_{cam}_{w}'] = window_mean(times, vals, ev_t, t0, t1)
            has_me[cam] = True
        except Exception as e:
            log.info('no %s motion energy for %s: %r', cam, eid, e)

    spikes, clusters = load_good_units(one, pid, eid=eid, pname=pname)
    clusters['Beryl'] = BrainRegions().id2acronym(clusters['atlas_id'], mapping='Beryl')
    zi = clusters[clusters['Beryl'] == C.REGION]
    if len(zi) == 0:
        log.warning('no %s units in %s', C.REGION, pid)
        return
    unit_ids = zi.index.to_numpy()

    keep = [c for c in ('x', 'y', 'z', 'depths', 'firing_rate', 'amp_median', 'uuids', 'cluster_id', 'acronym')
            if c in zi.columns]
    units = zi[keep].copy()
    units.insert(0, 'unit_index', unit_ids)
    units['eid'], units['pid'], units['subject'], units['lab'] = eid, pid, row['subject'], row['lab']

    edges = np.round(np.arange(C.EDGES[0], C.EDGES[1] + C.BIN / 2, C.BIN), 6)
    fm = t['firstMovement_times'].to_numpy()
    so = t['stimOn_times'].to_numpy()
    fm_counts = binned_counts(spikes['times'], spikes['clusters'], unit_ids, fm, edges)
    so_counts = binned_counts(spikes['times'], spikes['clusters'], unit_ids, so, edges)
    centers = (edges[:-1] + edges[1:]) / 2
    wv_fm = np.interp(fm[:, None] + centers[None, :], sl.wheel['times'].to_numpy(),
                      sl.wheel['velocity'].to_numpy()).astype(np.float32)

    t.to_parquet(out / 'trials.parquet')
    units.reset_index(drop=True).to_parquet(out / 'units.parquet')
    np.savez_compressed(out / 'binned.npz', fm_counts=fm_counts, so_counts=so_counts,
                        edges=edges, wheel_vel_fm=wv_fm)
    meta = {k: str(row[k]) for k in ('eid', 'pid', 'subject', 'lab', 'date', 'probe_name')}
    meta.update(n_units=int(len(unit_ids)), n_trials=int(len(t)), motion_energy=has_me)
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    log.info('saved %s: %d units, %d trials', pid, len(unit_ids), len(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max', type=int, default=None)
    args = ap.parse_args()

    one = C.get_one()
    log.info('IBL cache folder: %s', C.CACHE_DIR)
    inv = zi_inventory(one)
    todo = inv if args.max is None else inv.head(args.max)
    failed = []
    for _, row in todo.iterrows():
        try:
            log.info('processing %s %s (%d units)', row['subject'], row['probe_name'], row['n_good_zi_units'])
            process_insertion(one, row)
        except Exception as e:
            failed.append({'eid': row['eid'], 'pid': row['pid'], 'error': repr(e)})
            log.warning('FAILED %s: %r', row['pid'], e)
            log.debug(traceback.format_exc())
    pd.DataFrame(failed, columns=['eid', 'pid', 'error']).to_csv(C.RESULTS_DIR / 'download_failures.csv', index=False)
    log.info('done. processed insertions: %d, failed: %d',
             sum((p / 'meta.json').exists() for p in C.PROCESSED_DIR.iterdir()), len(failed))


if __name__ == '__main__':
    main()
