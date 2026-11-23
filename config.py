"""
Shared settings for the analysis pipeline (ZI by default; any Beryl region can be analysed).

All downloads, processed data, results and figures are stored inside the folder
that holds these scripts (the project folder):

  <project>/ibl_cache/   raw IBL files downloaded by ONE (can be large)
  <project>/processed/   compact per-insertion files made by 01_download_preprocess.py
  <project>/results/     tables (.csv) and summary numbers (.txt)
  <project>/figures/     quick-look figures made by the individual steps (.png, .pdf)
"""
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Region to analyse. ZI is the main region; control regions (e.g. MOs, RSPd, VPM, SCm) are run with
#   python run_all.py --region MOs
# and get their own folders (processed_MOs/, results_MOs/, figures_MOs/). The IBL cache is shared.
REGION = os.environ.get('ANALYSIS_REGION', 'ZI')
_SUFFIX = '' if REGION == 'ZI' else f'_{REGION}'

CACHE_DIR = PROJECT_DIR / 'ibl_cache'
PROCESSED_DIR = PROJECT_DIR / f'processed{_SUFFIX}'
RESULTS_DIR = PROJECT_DIR / f'results{_SUFFIX}'
FIGURES_DIR = PROJECT_DIR / f'figures{_SUFFIX}'
for _d in (CACHE_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(exist_ok=True)

# IBL brain-wide-map code (downloaded ZIP unpacked next to these scripts)
for _cand in (PROJECT_DIR / 'paper-brain-wide-map', PROJECT_DIR / 'paper-brain-wide-map-main'):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break

# macOS python.org builds sometimes lack root certificates -> use certifi's bundle
try:
    import certifi
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
except ImportError:
    pass

# ----------------------------------------------------------------------------
# Data selection
# ----------------------------------------------------------------------------
MIN_QC = 1.0                   # IBL "good unit" label
BWM_FREEZE = '2023_12_bwm_release'
MIN_UNITS_POPULATION = 5       # min ZI units per insertion for population analyses (decoding)

# ----------------------------------------------------------------------------
# Time binning (seconds)
# ----------------------------------------------------------------------------
BIN = 0.02
EDGES = (-0.5, 0.5)            # binned spike counts around each alignment event

# analysis windows: name -> (alignment event, start, end)
WINDOWS = {
    'iti':     ('stimOn',    -0.40, -0.10),   # quiescent period: block prior
    'premove': ('firstMove', -0.16,  0.00),   # decision -> action (main choice window)
    'move':    ('firstMove',  0.00,  0.20),   # early movement
}

# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
N_PERM = 2000                  # Freedman-Lane permutations per session (GLM); min p = 1/2001
N_PERM_DECODING = 200          # label permutations for decoding null
N_SUBSAMPLE = 50               # movement-matched resamples for decoding
N_SPEED_BINS = 5               # speed quantile bins for movement matching
ALPHA = 0.05
RANDOM_SEED = 2026


def get_one():
    """Connect to the public IBL database, caching files in the project folder."""
    from one.api import ONE
    return ONE(base_url='https://openalyx.internationalbrainlab.org',
               username='intbrainlab', password='international',
               cache_dir=CACHE_DIR, silent=True)


def save_fig(fig, name):
    """Save a figure as PNG (viewing) and PDF (journal submission)."""
    fig.savefig(FIGURES_DIR / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / f'{name}.pdf', bbox_inches='tight')


def write_summary(name, lines):
    """Write key numbers for the manuscript to results/<name>.txt and print them."""
    text = '\n'.join(lines)
    (RESULTS_DIR / f'{name}.txt').write_text(text, encoding='utf-8')
    print(text)
