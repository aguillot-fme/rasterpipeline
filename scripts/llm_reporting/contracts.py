import json
from typing import Any, Dict, List


def _ensure_list(value: Any, name: str) -> List[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def validate_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("plan payload must be an object")
    _ensure_list(payload.get("questions"), "questions")
    _ensure_list(payload.get("sql_intent"), "sql_intent")
    assumptions = payload.get("assumptions", [])
    if assumptions is not None:
        _ensure_list(assumptions, "assumptions")
    return payload


def validate_queries(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("queries payload must be an object")
    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise ValueError("queries must be a list")
    for idx, item in enumerate(queries):
        if not isinstance(item, dict):
            raise ValueError(f"queries[{idx}] must be an object")
        sql = item.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError(f"queries[{idx}].sql must be a non-empty string")
    return payload


def loads_json(payload: str) -> Dict[str, Any]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON payload must be an object")
    return data
