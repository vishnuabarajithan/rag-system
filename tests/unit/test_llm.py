from agents.llm import filter_think, parse_json, strip_think


def _stream(*parts):
    return "".join(filter_think(iter(parts)))


def test_strip_think_removes_block():
    assert strip_think("<think>plan it</think>\n\nThe answer [1].") == "The answer [1]."


def test_strip_think_passthrough_and_unterminated():
    assert strip_think("Plain answer.") == "Plain answer."
    assert strip_think("<think>never closed") == ""


def test_filter_think_drops_split_think_block():
    assert _stream("<thi", "nk>reason", "ing</th", "ink>\n", "Hello", " world") == "Hello world"


def test_filter_think_passthrough_without_think():
    assert _stream("Hel", "lo ", "world") == "Hello world"


def test_filter_think_short_prefix_that_is_not_think():
    assert _stream("<", "b>bold</b>") == "<b>bold</b>"


def test_parse_json_plain_fenced_and_wrapped():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here it is: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    assert parse_json('<think>x</think>{"a": 1}') == {"a": 1}


def test_parse_json_invalid_returns_empty():
    assert parse_json("not json") == {}
    assert parse_json("{broken") == {}
    assert parse_json("[1, 2]") == {}
    assert parse_json(None) == {}


def test_query_analyzer_defaults_on_garbage(monkeypatch):
    from agents.nodes import query_analyzer

    monkeypatch.setattr(query_analyzer.llm, "chat", lambda *a, **k: ("no json here", 0, 0))
    out = query_analyzer.run({"user_query": "What is X?", "iteration_count": 0})
    assert out["expanded_queries"] == ["What is X?"]
    assert out["query_intent"] == "factual"
    assert out["query_filters"] == {}
    assert out["iteration_count"] == 1


def test_query_analyzer_rejects_bad_types(monkeypatch):
    from agents.nodes import query_analyzer

    monkeypatch.setattr(
        query_analyzer.llm, "chat",
        lambda *a, **k: ('{"expanded_queries": "oops", "query_filters": []}', 0, 0),
    )
    out = query_analyzer.run({"user_query": "Q", "iteration_count": 0})
    assert out["expanded_queries"] == ["Q"]
    assert out["query_filters"] == {}
