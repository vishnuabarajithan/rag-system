import os

os.environ["QDRANT_LOCAL_PATH"] = ":memory:"

import webapp  # noqa: E402

CITES = [
    {"ref": "[1]", "source_file": "a.pdf", "page_num": 2, "content": "alpha text", "content_type": "text"},
    {"ref": "[2]", "source_file": "b.xlsx", "sheet_name": "S", "content": "| x |\n|---|\n| 1 |",
     "content_type": "table"},
    {"ref": "[3]", "source_file": "c.txt", "content": "unused", "content_type": "text"},
]


def _html(text, cites=CITES):
    return webapp.to_xml(webapp.assistant_bubble(text, cites, {}))


def test_inline_markers_become_links_and_only_cited_chips_show():
    html = _html("Revenue rose [1]. Also see [1, 2].")
    assert html.count('class="cite-link"') == 3
    assert 'data-ref="1"' in html and 'data-ref="2"' in html
    assert "c.txt" not in html                   # [3] never cited → no chip


def test_no_citations_in_answer_means_no_chips():
    html = _html("I can't answer that from your documents.")
    assert "cite-link" not in html and 'class="cites"' not in html


def test_unknown_ref_and_code_are_left_alone():
    html = _html("See [9]. Code: `arr[1]`")
    assert "cite-link" not in html


def test_table_excerpt_rendered_as_markdown():
    html = _html("Totals [2].")
    assert "<table>" in html


def test_page_load_discards_previous_session(monkeypatch):
    dropped = []
    monkeypatch.setattr(webapp._store, "drop_collection", dropped.append)
    session = {"sid": "old123"}
    webapp._sessions.get("old123")["docs"].append("x.pdf")
    webapp._discard_session(session)
    assert session["sid"] != "old123"
    assert dropped == ["session_old123"]
    assert webapp._sessions.get(session["sid"])["docs"] == []
