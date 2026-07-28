"""One definition of which failure-taxonomy fields count as filled in.

Two callers ask closely-related questions about the same three fields, and each used to
answer with its own copy of the logic:

* :mod:`episode_validator` asks *is this acceptable to upload?* — the trio must be
  all-filled or all-empty. A failure with no taxonomy at all is allowed through.
* The annotation server asks *is this fully annotated?*, to drive the tick in the episode
  list. There, a failure with no taxonomy is "partial" and still wants attention.

Those thresholds are deliberately different, and unifying them would be a policy change:
tightening the validator would reject already-uploaded data, and loosening the UI would
stop prompting annotators to finish. What is shared — and now lives here once — is what
"filled" means for each field, which is the part that must not drift.
"""

from __future__ import annotations

from typing import Any


def _is_filled(value: Any) -> bool:
    """A scalar field counts as filled when it has non-whitespace content."""
    if value is None:
        return False
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return bool(str(value).strip())


def _category_is_filled(value: Any) -> bool:
    """``failure_category`` may arrive as a list (the UI) or a scalar (older records)."""
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    return _is_filled(value)


def failure_trio_flags(
    failure_category: Any, failure_description: Any, severity: Any
) -> tuple[bool, bool, bool]:
    """Whether category, description and severity are each filled in.

    Returned in that order, so ``sum(...)`` gives how many of the three are present.
    """
    return (
        _category_is_filled(failure_category),
        _is_filled(failure_description),
        _is_filled(severity),
    )
