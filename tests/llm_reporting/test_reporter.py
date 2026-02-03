import json

import pytest

from scripts.llm_reporting.reporter import compile_report


def test_compile_report_writes_files(tmp_path, monkeypatch):
    results = {"results": [{"query_id": 1, "status": "success", "row_count": 2}]}

    def fake_call_chat(*_args, **_kwargs):
        return "# Summary\n\nAll good."

    monkeypatch.setattr("scripts.llm_reporting.reporter.call_chat", fake_call_chat)

    results_path = tmp_path / "results_index.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")
    meta = compile_report(
        results_index_json=str(results_path),
        user_goal="test goal",
        output_dir=str(tmp_path),
    )

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "report_meta.json").exists()
    assert meta["model"]
    assert "report.html" in meta["report_html"]


def test_compile_report_requires_goal(tmp_path):
    with pytest.raises(ValueError):
        compile_report(
            results_index_json=json.dumps({"results": []}),
            user_goal="",
            output_dir=str(tmp_path),
        )
