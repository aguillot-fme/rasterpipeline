import json
import os
from typing import Dict, Optional

from scripts.llm_reporting.contracts import loads_json, validate_plan
from scripts.llm_reporting.llm_client import call_chat, ensure_json_response


PLANNER_SYSTEM_PROMPT = """You are a data analysis planner.
Given dataset metadata and a user goal, produce:
- questions: a list of analytic questions
- sql_intent: a list of concise SQL intents (not full SQL)
- assumptions: a list of assumptions
Return JSON with keys: questions, sql_intent, assumptions.
"""


def _default_model() -> str:
    return os.getenv("LLM_PLANNER_MODEL", "gpt-4o-mini")


def generate_questions(
    profile_json: str,
    user_goal: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, object]:
    if not user_goal:
        raise ValueError("user_goal is required")
    if profile_json.strip().startswith("{"):
        profile = loads_json(profile_json)
    else:
        with open(profile_json, "r", encoding="utf-8") as f:
            profile = loads_json(f.read())

    user_prompt = json.dumps(
        {
            "task": user_goal,
            "profile": profile,
            "format": {"questions": ["..."], "sql_intent": ["..."], "assumptions": ["..."]},
        },
        indent=2,
    )

    response_text = call_chat(
        system_prompt=PLANNER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model or _default_model(),
        temperature=temperature,
    )
    payload = ensure_json_response(response_text)
    return validate_plan(payload)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_json", required=True)
    parser.add_argument("--user_goal", required=True)
    parser.add_argument("--output_path", default="")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    result = generate_questions(
        profile_json=args.profile_json,
        user_goal=args.user_goal,
        model=args.model or None,
    )

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))
