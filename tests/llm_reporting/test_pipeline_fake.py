import json

import pandas as pd

from scripts.llm_reporting.executor import run_queries
from scripts.llm_reporting.planner import generate_questions
from scripts.llm_reporting.profile import analyze_dataset
from scripts.llm_reporting.reporter import compile_report
from scripts.llm_reporting.sql_generator import generate_queries


def test_fake_pipeline_end_to_end(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "city": ["A", "B", "C"],
            "value": [10, 20, 30],
        }
    )
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    profile_path = tmp_path / "profile.json"
    plan_path = tmp_path / "plan.json"
    queries_path = tmp_path / "queries.json"
    results_dir = tmp_path / "results"
    report_dir = tmp_path / "report"

    profile = analyze_dataset(str(data_path), output_path=str(profile_path))

    def fake_planner_call(*_args, **_kwargs):
        return json.dumps(
            {
                "questions": ["What are the top values?"],
                "sql_intent": ["Rank by value"],
                "assumptions": [],
            }
        )

    def fake_sql_call(*_args, **_kwargs):
        return json.dumps(
            {
                "queries": [
                    {
                        "question": "What are the top values?",
                        "sql": "SELECT city, value FROM reporting_table ORDER BY value DESC",
                    }
                ]
            }
        )

    def fake_report_call(*_args, **_kwargs):
        return "# Summary\n\nTop values identified."

    monkeypatch.setattr("scripts.llm_reporting.planner.call_chat", fake_planner_call)
    monkeypatch.setattr("scripts.llm_reporting.sql_generator.call_chat", fake_sql_call)
    monkeypatch.setattr("scripts.llm_reporting.reporter.call_chat", fake_report_call)

    plan = generate_questions(json.dumps(profile), "Find top values")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    queries = generate_queries(json.dumps(plan), table_name="reporting_table")
    queries_path.write_text(json.dumps(queries), encoding="utf-8")

    run_queries(
        queries_json=str(queries_path),
        input_path=str(data_path),
        output_dir=str(results_dir),
        output_format="json",
    )

    results_index_path = results_dir / "results_index.json"
    assert results_index_path.exists()

    compile_report(
        results_index_json=str(results_index_path),
        user_goal="Find top values",
        output_dir=str(report_dir),
    )

    assert (report_dir / "report.md").exists()
    assert (report_dir / "report.html").exists()
