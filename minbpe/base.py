def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class Tokenizer:
    def __init__(self):
        self.merges = None  # (int, int) -> int
        self.vocab = None  # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError
    
    def encode(self, text):
        raise NotImplementedError
    
    def decode(self, ids):
        raise NotImplementedError