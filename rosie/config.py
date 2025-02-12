import os
from pathlib import Path

HOME_DIR = Path(os.path.expanduser("~"))
DOWNLOADS_DIR = HOME_DIR.joinpath('Downloads').resolve()

BASE_DIR = Path(__file__).resolve().parent.parent
# /Users/onesixx/my/git/shiny_refer
ASSET_DIR = BASE_DIR.joinpath("assets").resolve()
BACKEND_DIR = BASE_DIR.joinpath("backend").resolve()
DATA_DIR  = BASE_DIR.joinpath("data").resolve()
DOC_DIR   = BASE_DIR.joinpath("docs").resolve()
TMP_DIR   = BASE_DIR.joinpath("tmp").resolve()
