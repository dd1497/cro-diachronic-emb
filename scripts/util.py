import json
import numpy as np

from gensim.models import Word2Vec
from dataclasses import dataclass

def load_vectors(vec_path):
    w2v = Word2Vec.load(vec_path)
    vectors = w2v.wv
    del w2v
    return vectors

@dataclass
class PauzaInstance:
    text: str # List[tuple(str, str)]
    rating: str
    docid: str

    def aggregate_embeddings(self, model):
        aggregate = []
        for wpos_tuple in self.text:
            wp = '_'.join(wpos_tuple)
            try:
                vector = model[wp]
                aggregate.append(vector)
            except Exception as e:
                # print(e)
                # Skip since we're averaging anyway. Maybe add random uniform?
                pass

        if len(aggregate) == 0:
            print("Empty vectors")
            print(self.text)
            return None

        stacked_vectors = np.array(aggregate) 
        mu = stacked_vectors.mean(0)

        return mu # Average across axis 0
  
def load_jsonl(path):
  data = []
  with open(path, 'r') as infile:
    for line in infile:
        instance = json.loads(line)
        data.append(PauzaInstance(**instance))
  return data