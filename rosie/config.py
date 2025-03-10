import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
HOME_DIR = Path(os.path.expanduser("~"))
DOWNLOADS_DIR = BASE_DIR.joinpath('Downloads').resolve()

DATA_DIR  = BASE_DIR.joinpath("data").resolve()
DOC_DIR   = BASE_DIR.joinpath("docs").resolve()
TMP_DIR   = BASE_DIR.joinpath("tmp").resolve()
#for shiny
ASSET_DIR = BASE_DIR.joinpath("assets").resolve()
BACKEND_DIR = BASE_DIR.joinpath("backend").resolve()