import os
from joblib import Memory
from pathlib import Path
import getpass
import socket

hostname = socket.gethostname()
username = getpass.getuser()

PROJECT_ROOT = Path(__file__).parent
PROJECT_DIR = PROJECT_ROOT
DATA_DIR = PROJECT_DIR / 'data'
LOCAL_DATA_DIR = Path('data')
TEST_DATA_DIR = LOCAL_DATA_DIR

EXP_DIR = LOCAL_DATA_DIR / 'models'
RESULTS_DIR = LOCAL_DATA_DIR / 'results'
DEBUG_DATA_DIR = LOCAL_DATA_DIR / 'debug_data'
DEPS_DIR = LOCAL_DATA_DIR / 'deps'
CACHE_DIR = LOCAL_DATA_DIR / 'joblib_cache'
assert LOCAL_DATA_DIR.exists()
CACHE_DIR.mkdir(exist_ok=True)
TEST_DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
DEBUG_DATA_DIR.mkdir(exist_ok=True)

ASSET_DIR = DATA_DIR / 'assets'
MEMORY = Memory(CACHE_DIR, verbose=2)

# ROBOTS URDF
DREAM_DS_DIR = LOCAL_DATA_DIR / 'dream'

PANDA_DESCRIPTION_PATH = os.path.abspath(DEPS_DIR / "panda-description/panda.urdf")
PANDA_DESCRIPTION_PATH_VISUAL = os.path.abspath(DEPS_DIR / "panda-description/patched_urdf/panda.urdf")
KUKA_DESCRIPTION_PATH = os.path.abspath(DEPS_DIR / "kuka-description/iiwa_description/urdf/iiwa7.urdf")
BAXTER_DESCRIPTION_PATH = os.path.abspath("/DATA/disk1/cvda_share/robopose_data/deps/baxter-description/baxter_description/urdf/baxter.urdf")

OWI_DESCRIPTION = os.path.abspath(DEPS_DIR / 'owi-description' / 'owi535_description' / 'owi535.urdf')
OWI_KEYPOINTS_PATH = os.path.abspath(DEPS_DIR / 'owi-description' / 'keypoints.json')
