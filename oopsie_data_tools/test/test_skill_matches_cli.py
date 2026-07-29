"""Pin the skill's CLI claims to the parser that has to honour them.

``SKILL.md`` and its reference pages name commands and flags in running prose. Prose does
not fail when a flag is renamed, so an agent following the skill invents an invocation
that argparse rejects — the failure lands on the contributor, far from the rename.

This checks only what is mechanically checkable: that every ``oopsie-data <command>`` and
every ``--flag`` written next to one actually exists. Whether the prose describes them
correctly is a human's job, and deliberately not attempted here.
"""

from __future__ import annotations

import argparse
import re

import pytest

from oopsie_data_tools import cli
from oopsie_data_tools.utils.claude_skill import bundled_skill_dir

#: Flags belonging to something other than ``oopsie-data``, which appear in the pages as
#: part of an install or inference command line. Keyed by the flag so a genuine typo in an
#: oopsie-data flag is not silently excused by a substring match.
FOREIGN_FLAGS = {
    "--extra",  # uv sync --extra droid
    "--robot-profile",  # examples/inference_examples/
}


def _skill_pages():
    root = bundled_skill_dir()
    return [root / "SKILL.md"] + sorted((root / "reference").glob("*.md"))


def _parser_flags() -> dict[str, set[str]]:
    """Every option string the parser accepts, per subcommand, plus top-level under ""."""
    parser = cli.build_parser()
    flags: dict[str, set[str]] = {"": set()}
    for action in parser._actions:
        flags[""].update(action.option_strings)
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                flags[name] = set()
                for sub_action in sub._actions:
                    flags[name].update(sub_action.option_strings)
    return flags


#: ``oopsie-data <command>`` followed by the flags on the rest of that line. The command
#: may be wrapped in backticks or sit inside a fenced block; either way it starts the
#: match, and the line ends the invocation. Horizontal whitespace only — ``\s`` would run
#: past the end of a line and read the next one's first word as a command, which is how
#: the frontmatter's ``name: oopsie-data`` came out looking like ``oopsie-data description``.
_INVOCATION = re.compile(r"oopsie-data[ \t]+(?:--config-dir[ \t]+\S+[ \t]+)?([a-z][a-z-]*)([^\n`]*)")
_FLAG = re.compile(r"--[a-z][a-z-]*")


@pytest.mark.parametrize("page", _skill_pages(), ids=lambda p: p.name)
def test_every_command_the_skill_names_exists(page):
    flags = _parser_flags()
    text = page.read_text(encoding="utf-8")

    unknown = {
        command
        for command, _ in _INVOCATION.findall(text)
        # '--help' and '--version' follow the program name directly, so the regex reads
        # the next word as a command; both are top-level flags, matched below.
        if not command.startswith("-") and command not in flags
    }
    assert not unknown, f"{page.name} names commands the CLI does not have: {sorted(unknown)}"


@pytest.mark.parametrize("page", _skill_pages(), ids=lambda p: p.name)
def test_every_flag_the_skill_names_exists(page):
    flags = _parser_flags()
    text = page.read_text(encoding="utf-8")

    unknown = set()
    for command, rest in _INVOCATION.findall(text):
        if command not in flags:
            continue
        accepted = flags[command] | flags[""] | FOREIGN_FLAGS
        unknown.update(f"{command} {flag}" for flag in _FLAG.findall(rest) if flag not in accepted)

    assert not unknown, f"{page.name} names flags the CLI does not accept: {sorted(unknown)}"


def test_the_regex_would_actually_catch_a_rename():
    """A test that can never fail is worse than no test, so prove this one can."""
    flags = _parser_flags()
    command, rest = _INVOCATION.findall("  oopsie-data validate --path ./samples --nonesuch\n")[0]

    assert command == "validate"
    assert "--nonesuch" in _FLAG.findall(rest)
    assert "--nonesuch" not in flags["validate"]
    assert "--path" in flags["validate"]
