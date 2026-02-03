import json
import os
from typing import Dict, Optional

try:
    import sqlglot  # type: ignore
except Exception:  # pragma: no cover
    sqlglot = None

from scripts.llm_reporting.contracts import loads_json, validate_queries
from scripts.llm_reporting.llm_client import call_chat, ensure_json_response


SQL_SYSTEM_PROMPT = """You are a SQL generator for DuckDB.
Given analytical questions and SQL intent, return JSON:
{ "queries": [ { "question": "...", "sql": "SELECT ..."} ] }
Rules:
- Only SELECT queries.
- Use read_parquet or the provided table name.
- Keep results small (LIMIT 30).
"""


def _default_model() -> str:
    return os.getenv("LLM_SQL_MODEL", "gpt-4o-mini")


def _sanitize_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";")
    if "limit" not in sql.lower():
        sql = f"{sql} LIMIT 30"
    return sql


def _validate_sql(sql: str) -> None:
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed")
    if sqlglot is None:
        return
    parsed = sqlglot.parse_one(sql, read="duckdb")
    if parsed is None:
        raise ValueError("SQL parser could not parse query")
    if parsed.key != "select":
        raise ValueError("Only SELECT queries are allowed")


def generate_queries(
    plan_json: str,
    table_name: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
) -> Dict[str, object]:
    if not table_name:
        raise ValueError("table_name is required")
    if plan_json.strip().startswith("{"):
        plan = loads_json(plan_json)
    else:
        with open(plan_json, "r", encoding="utf-8") as f:
            plan = loads_json(f.read())

    user_prompt = json.dumps(
        {
            "plan": plan,
            "table_name": table_name,
            "format": {"queries": [{"question": "...", "sql": "SELECT ..."}]},
        },
        indent=2,
    )

    response_text = call_chat(
        system_prompt=SQL_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model or _default_model(),
        temperature=temperature,
    )
    payload = ensure_json_response(response_text)
    payload = validate_queries(payload)

    for item in payload["queries"]:
        raw_sql = item["sql"]
        cleaned = _sanitize_sql(raw_sql)
        _validate_sql(cleaned)
        item["sql"] = cleaned

    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_json", required=True)
    parser.add_argument("--table_name", required=True)
    parser.add_argument("--output_path", default="")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    result = generate_queries(
        plan_json=args.plan_json,
        table_name=args.table_name,
        model=args.model or None,
    )

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))
