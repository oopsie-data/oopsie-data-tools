"""The error every validation failure raises."""

from __future__ import annotations


class EpisodeValidationError(AssertionError):
    """An episode does not conform to the oopsiedata schema.

    Subclasses :class:`AssertionError` deliberately. Validation used to be built entirely
    out of bare ``assert`` statements, and ``except AssertionError`` is the documented way
    to catch a validation failure — lab scripts will have copied it. Inheriting keeps every
    one of those callers working while giving new code something specific to catch.

    The reason for moving off ``assert`` at all: ``python -O`` strips assert statements, so
    the entire schema check silently evaporated under optimised interpreters. This is the
    gate in front of a public dataset; it should not depend on an interpreter flag.
    """
