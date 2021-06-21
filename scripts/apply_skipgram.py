from src.skipgram import get_embeddings

import yaml
import pickle

with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)

WINDOW_SIZE = params['skipgram']['window']
VECTOR_SIZE = params['skipgram']['vector_size']
MIN_WORD_COUNT = params['skipgram']['min_count']
EPOCHS = params['skipgram']['epochs']

if __name__ == "__main__":
    with open(f'data/sampled_neighbourhoods.pkl', 'rb') as handle:
        nbs = pickle.load(handle)
    
    embeddings = get_embeddings(nbs,
                                VECTOR_SIZE,
                                WINDOW_SIZE,
                                MIN_WORD_COUNT,
                                EPOCHS)

    with open(f'data/embeddings.pkl', 'wb') as handle:
        pickle.dump(embeddings, handle)
