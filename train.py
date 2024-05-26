import os
import time
from minbpe import BasicTokenizer

if __name__ == '__main__':
    text = open("tests/taylorswift.txt", 'r', encoding="utf-8").read()

    t0 = time.time()
    tokenizer = BasicTokenizer()
    tokenizer.train(text, 512, verbose=True)
    t1 = time.time()
    print(f"Training took {t1 - t0:.2f} seconds")

    ids = tokenizer.encode(text)
    assert text == tokenizer.decode(ids)

    print("raw len:", len(list(text.encode('utf-8'))))
    print("ids len:", len(ids))
    print(f"compression ratio: {len(list(text.encode('utf-8'))) / len(ids):.2f}X")