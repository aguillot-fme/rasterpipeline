import json

import pytest

from scripts.llm_reporting.planner import generate_questions


def test_generate_questions_valid(monkeypatch):
    payload = {"questions": ["q1"], "sql_intent": ["i1"], "assumptions": []}
    profile = {"schema_sql": "CREATE TABLE t (x INT)"}

    def fake_call_chat(*_args, **_kwargs):
        return json.dumps(payload)

    monkeypatch.setattr("scripts.llm_reporting.planner.call_chat", fake_call_chat)

    result = generate_questions(json.dumps(profile), "goal")
    assert result["questions"] == ["q1"]
    assert result["sql_intent"] == ["i1"]


def test_generate_questions_invalid(monkeypatch):
    def fake_call_chat(*_args, **_kwargs):
        return json.dumps({"questions": "bad"})

    monkeypatch.setattr("scripts.llm_reporting.planner.call_chat", fake_call_chat)

    with pytest.raises(ValueError):
        generate_questions(json.dumps({"schema_sql": "x"}), "goal")
