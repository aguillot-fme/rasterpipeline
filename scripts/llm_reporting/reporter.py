import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from scripts.llm_reporting.llm_client import call_chat

try:
    from markdown_it import MarkdownIt  # type: ignore
except Exception:  # pragma: no cover
    MarkdownIt = None


ANALYST_SYSTEM_PROMPT = """You are an analyst producing a concise report.
Given the user goal and query results, write a markdown report with:
- Summary
- Key findings (bullet list)
- Limitations
- Next steps
"""


def _default_model() -> str:
    return os.getenv("LLM_ANALYST_MODEL", "gpt-4o")


def _markdown_to_html(markdown_text: str) -> str:
    if MarkdownIt is None:
        escaped = markdown_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<pre>{escaped}</pre>"
    return MarkdownIt().render(markdown_text)


def compile_report(
    results_index_json: str,
    user_goal: str,
    output_dir: str,
    output_path: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    if not user_goal:
        raise ValueError("user_goal is required")
    os.makedirs(output_dir, exist_ok=True)
    if results_index_json.strip().startswith("{"):
        results = json.loads(results_index_json)
    else:
        with open(results_index_json, "r", encoding="utf-8") as f:
            results = json.loads(f.read())

    user_prompt = json.dumps(
        {"task": user_goal, "results": results},
        indent=2,
    )

    markdown_report = call_chat(
        system_prompt=ANALYST_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model or _default_model(),
        temperature=temperature,
    )

    report_md_path = os.path.join(output_dir, "report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    html = _markdown_to_html(markdown_report)
    report_html_path = output_path or os.path.join(output_dir, "report.html")
    with open(report_html_path, "w", encoding="utf-8") as f:
        f.write(html)

    meta = {
        "model": model or _default_model(),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "report_md": report_md_path,
        "report_html": report_html_path,
    }
    meta_path = os.path.join(output_dir, "report_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_index_json", required=True)
    parser.add_argument("--user_goal", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_path", default="")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    compile_report(
        results_index_json=args.results_index_json,
        user_goal=args.user_goal,
        output_dir=args.output_dir,
        output_path=args.output_path or None,
        model=args.model or None,
    )
