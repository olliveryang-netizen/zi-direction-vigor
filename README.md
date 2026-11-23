# zi-direction-vigor

Analysis code for:

> Yang Y. Choice-related activity in the zona incerta reflects the direction and vigor of the ongoing movement. *Submitted to Journal of Neurophysiology.*

The analyses use the public International Brain Laboratory (IBL) Brain-Wide Map release
(`2023_12_bwm_release`). The pipeline downloads the data, preprocesses every zona incerta (ZI)
insertion, and asks whether choice-related activity in ZI reflects the decision or the movement
that reports it. The same pipeline is run on secondary motor cortex (MOs) and the ventral
posteromedial thalamic nucleus (VPM) for comparison.

## Requirements

- Python 3 (tested with Python 3.13 on macOS)
- The packages in `requirements.txt`
- The IBL `brainwidemap` package from
  [int-brain-lab/paper-brain-wide-map](https://github.com/int-brain-lab/paper-brain-wide-map)
- Enough free disk space for the IBL cache (several tens of GB for the three regions)

## Installation

```bash
git clone https://github.com/USERNAME/zi-direction-vigor.git
cd zi-direction-vigor
python3 -m venv ibl_env
source ibl_env/bin/activate
pip install -r requirements.txt

# brainwidemap: clone it into the project folder; config.py adds it to the Python path
git clone https://github.com/int-brain-lab/paper-brain-wide-map.git
```

Data are read from the public IBL server (`openalyx.internationalbrainlab.org`) with the public
credentials set in `config.py`. No account is needed.

## Folder layout

Everything is written inside the project folder:

| Folder | Contents |
|---|---|
| `ibl_cache/` | raw IBL files downloaded by ONE (large) |
| `processed/`, `processed_MOs/`, `processed_VPM/` | compact per-insertion files (step 01) |
| `results/`, `results_MOs/`, `results_VPM/` | tables (`.csv`) and `summary_*.txt` files |
| `results_comparison/` | across-region results (steps 08, 11, 12, 13) |
| `figures/`, `figures_comparison/` | quick-look figures made by the individual steps |
| `figures_paper/` | the paper figures (step 14) |

The region is chosen with the `ANALYSIS_REGION` environment variable (default `ZI`) or with
`run_all.py --region`.

## Running the analysis

```bash
# 1. Steps 01-07, 09 and 10 for each region
python run_all.py
python run_all.py --region MOs
python run_all.py --region VPM

# 2. Across-region comparison and unit-count-matched decoding
python 08_compare_regions.py
python 11_unit_matched.py

# 3. Pose control, once per region (downloads pose files the first time)
python 12_pose_control.py
ANALYSIS_REGION=MOs python 12_pose_control.py
ANALYSIS_REGION=VPM python 12_pose_control.py

# 4. Neural vs pose timing
python 13_pose_timecourse.py

# 5. Paper figures and legend numbers
python 14_make_figures.py
```

Step 01 can be stopped and restarted; finished insertions are skipped. `run_all.py --skip-download`
runs the analyses only, and `run_all.py --only 03` runs a single step.

## Scripts

| Script | Analysis | Paper figure |
|---|---|---|
| `01_download_preprocess.py` | download, trial selection, spike binning, wheel and video features | – |
| `02_dataset_behavior.py` | dataset, behaviour, choice–speed confound | Fig. 1 |
| `03_single_unit_glm.py` | single-unit encoding model with permutation tests | Fig. 2, S1 |
| `04_decoding.py` | speed-matched population decoding and generalisation | Fig. 3 |
| `05_subspace.py` | direction and speed population axes | Fig. 5A–C |
| `06_location.py` | anatomical location of unit types within ZI | Fig. S3 |
| `07_imposter_null.py` | imposter-session null distributions | Fig. 2F, 3B, S2 |
| `08_compare_regions.py` | ZI vs MOs vs VPM | Fig. 6, S5 |
| `09_followup.py` | change of mind, lateralisation, feedback control, time-resolved decoding | Fig. 4A, 6E |
| `10_reversal_history.py` | reversal-aligned decoding, trial history, onset latencies | Fig. 3E–F, 4B, S5 |
| `11_unit_matched.py` | decoding with equal numbers of units per region | Fig. 6C |
| `12_pose_control.py` | control for paw and nose movements (pose estimation) | Fig. 6F |
| `13_pose_timecourse.py` | timing of neural vs pose direction information | Fig. 4C–F, S4 |
| `14_make_figures.py` | all paper figures from the saved results | all |
| `config.py`, `common.py` | shared settings and helper functions | – |

Steps 02–13 write `summary_NN_*.txt` files with the numbers reported in the paper;
step 14 writes `figures_paper/figure_legends.md`.

## Figures

`14_make_figures.py` uses Helvetica Bold. On macOS it is extracted from the system font
collection into `fonts/` the first time the script runs (the font is not distributed here).
On other systems it falls back to Arial Bold, TeX Gyre Heros Bold or Liberation Sans Bold;
any bold `.ttf` or `.otf` file placed in `fonts/` is also used.

## Validation on simulated data

```bash
python validation/run_validation.py
```

This builds two simulated datasets in the same format as the processed data and runs the
analysis steps (`run_all.py --skip-download`) on each: one in which a subset of units carries a choice signal independent of movement,
and one in which no unit carries choice information but choice and speed are confounded.
Unmatched decoding is above chance in the second dataset, whereas the GLM and speed-matched
decoding stay at chance.

## License

MIT License (see `LICENSE`).

## Contact

Yuheng Yang, Department of Physiology, Yamaguchi University Graduate School of Medicine,
Yamaguchi 755-8505, Japan.
