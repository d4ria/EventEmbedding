from src.utils import preprocess_data
import yaml

with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)

TIMEDELTA = params["timedelta"]
DATA_PATH = params["datapath"]

if __name__ == "__main__":
    network = preprocess_data(timedelta='D')
    network.to_pickle("data/preprocessed_dataframe.pkl")