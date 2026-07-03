"""Phase 4 -- the manager. A LangChain workflow that splits a document into
clauses, classifies each one via app/model.py, and checks them against a
simple set of compliance rules.
"""

import re

from langchain_core.runnables import RunnableLambda

from app.model import classify_clause

REQUIRED_CLAUSES = [
    "Governing Laws",
    "Notices",
    "Amendments",
    "Severability",
    "Assignments",
]

MIN_CLAUSE_LENGTH = 20

_BLANK_LINE_RE = re.compile(r"\n\s*\n+")
_NUMBERED_ITEM_RE = re.compile(r"(?m)^\s*\(?\d+[\.\)]\s+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z0-9])")


def split_into_clauses(text: str) -> list:
    """Break a document into clause-sized chunks: paragraphs first, then
    numbered list items within a paragraph, then a sentence-boundary fallback
    for long paragraphs that aren't numbered.
    """
    paragraphs = [p.strip() for p in _BLANK_LINE_RE.split(text) if p.strip()]

    clauses = []
    for para in paragraphs:
        pieces = [p.strip() for p in _NUMBERED_ITEM_RE.split(para) if p.strip()]
        if len(pieces) > 1:
            clauses.extend(pieces)
            continue
        if len(para) > 400:
            clauses.extend(s.strip() for s in _SENTENCE_BOUNDARY_RE.split(para) if s.strip())
        else:
            clauses.append(para)

    return [c for c in clauses if len(c) >= MIN_CLAUSE_LENGTH]


def _classify_step(clauses: list) -> list:
    return [{"text": c, "label": classify_clause(c)["label"]} for c in clauses]


def _apply_rules_step(classified: list) -> dict:
    present_types = sorted({c["label"] for c in classified})
    missing_required = [r for r in REQUIRED_CLAUSES if r not in present_types]
    return {
        "clauses": classified,
        "present_types": present_types,
        "missing_required": missing_required,
    }


def _build_report_step(data: dict) -> dict:
    num_clauses = len(data["clauses"])
    if data["missing_required"]:
        summary = (
            f"Reviewed {num_clauses} clauses. "
            f"Missing required clause type(s): {', '.join(data['missing_required'])}."
        )
    else:
        summary = f"Reviewed {num_clauses} clauses. All required clause types are present."
    return {
        "num_clauses": num_clauses,
        "clauses": data["clauses"],
        "present_types": data["present_types"],
        "missing_required": data["missing_required"],
        "summary": summary,
    }


review_chain = (
    RunnableLambda(split_into_clauses)
    | RunnableLambda(_classify_step)
    | RunnableLambda(_apply_rules_step)
    | RunnableLambda(_build_report_step)
)


def review_document(text: str) -> dict:
    return review_chain.invoke(text)
