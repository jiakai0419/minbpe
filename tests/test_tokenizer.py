import pytest
import os

from minbpe import BasicTokenizer, RegexTokenizer

test_strings = [
    "",
    "?",
    "hello world!!!? (안녕하세요!) lol123 😉",
    "FILE:taylorswift.txt",
]

def unpack(text):
    if text.startswith("FILE:"):
        dirname = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(dirname, text[5:])
        contents = open(file, 'r', encoding="utf-8").read()
        return contents
    else:
        return text

@pytest.mark.parametrize("tokenizer_factory", [BasicTokenizer, RegexTokenizer])
@pytest.mark.parametrize("text", test_strings)
def test_encode_decode_identity_0(tokenizer_factory, text):
    text = unpack(text)
    tokenizer = tokenizer_factory()
    tokenizer.train("", 256, verbose=False)
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert text == decoded

basic_tokenizer_trained = BasicTokenizer()
basic_tokenizer_trained.train(unpack("FILE:taylorswift.txt"), 512, verbose=False)
@pytest.mark.skip(reason="It's running too slow")
@pytest.mark.parametrize("text", test_strings)
def test_encode_decode_identity_1(text):
    text = unpack(text)
    ids = basic_tokenizer_trained.encode(text)
    decoded = basic_tokenizer_trained.decode(ids)
    assert text == decoded

regex_tokenizer_trained = RegexTokenizer()
regex_tokenizer_trained.train(unpack("FILE:taylorswift.txt"), 512, verbose=False)
@pytest.mark.skip(reason="It's running too slow")
@pytest.mark.parametrize("text", test_strings)
def test_encode_decode_identity_2(text):
    text = unpack(text)
    ids = regex_tokenizer_trained.encode(text)
    decoded = regex_tokenizer_trained.decode(ids)
    assert text == decoded

@pytest.mark.parametrize("tokenizer_factory", [BasicTokenizer, RegexTokenizer])
def test_wikipedia_example(tokenizer_factory):
    """
    Quick unit test, following along the Wikipedia example:
    https://en.wikipedia.org/wiki/Byte_pair_encoding

    According to Wikipedia, running bpe on the input string:
    "aaabdaaabac"

    for 3 merges will result in string:
    "XdXac"

    where:
    X=ZY
    Y=ab
    Z=aa

    Keep in mind that for us a=97, b=98, c=99, d=100 (ASCII values)
    so Z will be 256, Y will be 257, X will be 258.

    So we expect the output list of ids to be [258, 100, 258, 97, 99]
    """
    tokenizer = tokenizer_factory()
    text = "aaabdaaabac"
    tokenizer.train(text, 256 + 3, verbose=False)
    ids = tokenizer.encode(text)
    assert ids == [258, 100, 258, 97, 99]
    assert tokenizer.decode(ids) == text


if __name__ == "__main__":
    pytest.main()