from vectorstore.sparse_index import bm25_encode


def test_empty_text():
    indices, values = bm25_encode("")
    assert indices == []
    assert values == []


def test_returns_parallel_lists():
    indices, values = bm25_encode("hello world hello")
    assert len(indices) == len(values)


def test_repeated_token_higher_score():
    _, values_many = bm25_encode("apple apple apple apple")
    _, values_one  = bm25_encode("apple banana cherry date")
    # "apple" appearing 4× in a 4-token doc should score higher than once in 4 tokens
    assert max(values_many) > max(values_one)
