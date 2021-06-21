import pandas as pd
from src.utils import visualize_embeddings

import yaml
import pickle

with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)

DATA_PATH = "data/preprocessed_dataframe.pkl"

if __name__ == "__main__":
    with open(f'data/embeddings.pkl', 'rb') as handle:
        embeddings = pickle.load(handle)

    preprocessed_df = pd.read_pickle(DATA_PATH)

    visualize_embeddings(embeddings, preprocessed_df)