"""Version-sync guard — the "revision block" of this repository.

Every user-visible statement of the app version must agree:
gui/version.py (the single source of truth), pyproject.toml, the README
title and badge, and the newest CHANGELOG entry. The v1.4.2 release
shipped with the in-app version still reading 1.4.1 (issue #22 follow-up)
because nothing enforced this; now any drift fails CI on every PR.

The tag-to-version check (the other half of that failure) lives in
.github/workflows/release.yml, which refuses to publish a release whose
tag does not match APP_VERSION.
"""

import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def app_version() -> str:
    text = (REPO / "gui" / "version.py").read_text(encoding="utf-8")
    m = re.search(r"""APP_VERSION\s*=\s*['"]([^'"]+)['"]""", text)
    assert m, "APP_VERSION not found in gui/version.py"
    return m.group(1)


def test_app_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", app_version())


def test_pyproject_matches_app_version():
    with open(REPO / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    assert pyproject["project"]["version"] == app_version()


def test_readme_title_and_badge_match_app_version():
    version = app_version()
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    title = re.search(r"<h1[^>]*>([^<]*)</h1>", readme)
    assert title, "README <h1> title not found"
    assert f"V{version}" in title.group(1), (
        f"README title says {title.group(1)!r} but APP_VERSION is {version}"
    )

    badge = re.search(r"img\.shields\.io/badge/version-([\d.]+)-", readme)
    assert badge, "README version badge not found"
    assert badge.group(1) == version, (
        f"README badge says {badge.group(1)} but APP_VERSION is {version}"
    )


def test_newest_changelog_entry_matches_app_version():
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = re.search(r"^## \[([\d.]+)\]", changelog, re.MULTILINE)
    assert entry, "no version heading found in CHANGELOG.md"
    assert entry.group(1) == app_version(), (
        f"newest CHANGELOG entry is {entry.group(1)} but APP_VERSION is "
        f"{app_version()} — add a CHANGELOG section for the new version"
    )


def test_release_gate_accepts_the_same_quoting_as_this_guard():
    """The PR guard and the release gate must parse APP_VERSION identically.

    This guard accepted `['"]`; .github/workflows/release.yml accepted only
    `"`. A switch to single quotes therefore passed CI and then crashed the
    release job with an AttributeError on None — after the tag was pushed.
    """
    workflow = (REPO / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    gate = re.search(r"APP_VERSION\\s\*=\\s\*(\S+?),", workflow)
    assert gate, "APP_VERSION pattern not found in release.yml"
    assert "['\"]" in gate.group(1), (
        "release.yml must accept both quote styles, like app_version() above; "
        f"found {gate.group(1)!r}"
    )

    # And prove it end to end against both spellings of the assignment.
    pattern = r"""APP_VERSION\s*=\s*['"]([^'"]+)['"]"""
    for source in ('APP_VERSION = "1.2.3"', "APP_VERSION = '1.2.3'"):
        assert re.search(pattern, source).group(1) == "1.2.3"


def test_readme_test_count_matches_the_suite():
    """The README's advertised test count must match what actually collects.

    It read "139 tests" while the suite had grown well past that; nothing
    checked it, so it drifted with every added test.
    """
    import subprocess

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    claimed = re.search(r"Test suite grew to \*\*(\d+) tests\*\*", readme)
    assert claimed, "README test-count sentence not found"

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--collect-only"],
        cwd=REPO, capture_output=True, text=True, errors="replace",
    )
    # Check the exit code BEFORE parsing: a module that fails to import still
    # prints "N tests collected, 1 error", and a broken conftest reports only
    # on stderr — either way the README would be blamed for a collection bug.
    assert result.returncode == 0, (
        f"pytest --collect-only exited {result.returncode}; fix collection "
        f"before checking the README count.\n--- stdout ---\n"
        f"{result.stdout[-2000:]}\n--- stderr ---\n{result.stderr[-2000:]}"
    )
    collected = re.search(r"(\d+) tests collected", result.stdout)
    assert collected, (
        f"could not parse a collection count from:\n--- stdout ---\n"
        f"{result.stdout[-2000:]}\n--- stderr ---\n{result.stderr[-2000:]}"
    )

    assert int(claimed.group(1)) == int(collected.group(1)), (
        f"README says {claimed.group(1)} tests but the suite collects "
        f"{collected.group(1)}"
    )


def _fake_collect_only(monkeypatch, returncode, stdout, stderr=""):
    """Replace subprocess.run so the README-count test sees a canned run."""
    import subprocess

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, returncode, stdout, stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)


def _readme_count() -> str:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    return re.search(r"Test suite grew to \*\*(\d+) tests\*\*", readme).group(1)


def test_readme_count_guard_fails_on_a_collection_error(monkeypatch):
    """A broken test module must fail the guard even though pytest still
    prints a parseable (and here matching) count — it used to pass silently."""
    _fake_collect_only(
        monkeypatch, 2,
        "ERROR tests/test_broken.py\n"
        f"{_readme_count()} tests collected, 1 error in 0.10s\n",
    )
    with pytest.raises(AssertionError, match="exited 2"):
        test_readme_test_count_matches_the_suite()


def test_readme_count_guard_surfaces_stderr(monkeypatch):
    """A conftest crash writes only to stderr; the failure must show it."""
    _fake_collect_only(
        monkeypatch, 4, "",
        "ImportError while loading conftest 'tests/conftest.py'.\n",
    )
    with pytest.raises(AssertionError, match="ImportError while loading conftest"):
        test_readme_test_count_matches_the_suite()


# ── Contributor docs run tools with the .venv interpreter ────────────────

_BARE_TOOL = re.compile(r"^\s*(python3?|py|pytest|ruff)(\s|$)")


def _fenced_code_lines(markdown: str):
    """Yield (fence language, line) for every line inside a fenced block."""
    lang = None
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            lang = None if lang is not None else line.strip()[3:].strip()
            continue
        if lang is not None:
            yield lang, line


def test_dev_docs_run_tools_with_the_venv_interpreter():
    """CONTRIBUTING.md said `python -m pytest tests/ -q` right under two lines
    that used .venv/bin/python. A bare `python` is whatever is on PATH — on
    the owner's Windows PC a system Python without pytest — not the .venv
    that setup created. requirements-dev.txt repeated the same command, and
    the Windows lines sat in a bash block, where Git Bash strips backslashes.
    """
    contributing = (REPO / "CONTRIBUTING.md").read_text(encoding="utf-8")
    offenders = []
    for lang, line in _fenced_code_lines(contributing):
        if line.lstrip().startswith("#"):
            continue  # a comment line, not a command
        command = line.split(" #", 1)[0]
        if _BARE_TOOL.match(command):
            offenders.append(line.strip())
        elif lang in ("bash", "sh") and ".venv\\" in command:
            offenders.append(f"{line.strip()}  (backslash path in a {lang} block)")
    assert not offenders, (
        "CONTRIBUTING.md commands must use the .venv interpreter:\n"
        + "\n".join(offenders)
    )

    requirements = (REPO / "requirements-dev.txt").read_text(encoding="utf-8")
    for line in requirements.splitlines():
        if "-m pytest" in line:
            assert ".venv" in line, (
                f"requirements-dev.txt runs tests without the .venv interpreter: {line!r}"
            )
