import json

import pytest

from scripts.llm_reporting.sql_generator import generate_queries


def test_generate_queries_valid(monkeypatch):
    payload = {"queries": [{"question": "q1", "sql": "SELECT 1"}]}
    plan = {"questions": ["q1"], "sql_intent": ["i1"], "assumptions": []}

    def fake_call_chat(*_args, **_kwargs):
        return json.dumps(payload)

    monkeypatch.setattr("scripts.llm_reporting.sql_generator.call_chat", fake_call_chat)

    result = generate_queries(json.dumps(plan), table_name="t")
    assert result["queries"][0]["sql"].lower().startswith("select")
    assert "limit" in result["queries"][0]["sql"].lower()


def test_generate_queries_rejects_non_select(monkeypatch):
    payload = {"queries": [{"question": "q1", "sql": "DELETE FROM t"}]}
    plan = {"questions": ["q1"], "sql_intent": ["i1"], "assumptions": []}

    def fake_call_chat(*_args, **_kwargs):
        return json.dumps(payload)

    monkeypatch.setattr("scripts.llm_reporting.sql_generator.call_chat", fake_call_chat)

    with pytest.raises(ValueError):
        generate_queries(json.dumps(plan), table_name="t")
