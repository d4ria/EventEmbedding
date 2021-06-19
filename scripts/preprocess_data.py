from src.utils import preprocess_data
import yaml

with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)

TIMEDELTA = params["timedelta"]
DATAPATH = params["datapath"]

if __name__ == "__main__":
    network = preprocess_data(data_path=DATAPATH, timedelta=TIMEDELTA)
    network.to_pickle("data/preprocessed_dataframe.pkl")