"""
safety.py — D3.
done by Yousef

Lightweight input/output validation at the API boundary.
Called by GraphRAGExecutor before processing and before returning answers.
"""

import re

_BLOCKLIST = [
    r"ignore\s+(previous|all)\s+instructions",
    r"you\s+are\s+now\s+",
    r"pretend\s+(you|to\s+be)",
    r"jailbreak",
    r"\bDAN\b",
    r"disregard\s+(your|all)\s+(previous\s+)?instructions",
    r"act\s+as\s+(if\s+you\s+are|a\s+)",
]

_MAX_INPUT  = 1000
_MAX_OUTPUT = 2000


def check_input(query: str) -> str:
    """
    Validate and clean query text before processing.

    Raises ValueError for: empty queries, oversized queries, and queries
    matching prompt-injection or jailbreak patterns.

    Returns the stripped query on success.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    query = query.strip()

    if len(query) > _MAX_INPUT:
        raise ValueError(f"Query too long — max {_MAX_INPUT} chars, got {len(query)}")

    for pattern in _BLOCKLIST:
        if re.search(pattern, query, re.IGNORECASE):
            raise ValueError("Query blocked by content filter")

    return query


def check_output(answer: str) -> str:
    """
    Clean and truncate generated answer text before returning to the caller.

    Strips common LLM preamble phrases and hard-truncates at _MAX_OUTPUT chars.
    """
    if not answer:
        return ""

    answer = re.sub(
        r"^(As an AI( language model)?[,.]?\s*|I('m| am) an AI[,.]?\s*)",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    return answer.strip()[:_MAX_OUTPUT]
