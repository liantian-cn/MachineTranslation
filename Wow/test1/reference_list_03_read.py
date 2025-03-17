import pickle
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

print(pickle.load(open(BASE_DIR.joinpath("reference_list.pkl"), "rb")))