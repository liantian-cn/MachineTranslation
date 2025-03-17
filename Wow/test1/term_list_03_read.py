import pickle
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

print(pickle.load(open(BASE_DIR.joinpath("term_list.pkl"), "rb")))