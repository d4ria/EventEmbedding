from itertools import product
from gensim.models import Word2Vec

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

    model = Word2Vec(sentences=nbs, 
                     vector_size=VECTOR_SIZE, 
                     window=WINDOW_SIZE, 
                     min_count=MIN_WORD_COUNT, 
                     epochs=EPOCHS,
                     sg=1)
    
    X = model.wv.get_normed_vectors()
    keys = model.wv.index_to_key

    embeddings = {keys[idx]: X[idx] for idx in range(len(keys))}

    with open(f'data/embeddings.pkl', 'wb') as handle:
        pickle.dump(embeddings, handle)
