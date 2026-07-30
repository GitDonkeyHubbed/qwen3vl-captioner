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
import tomllib
from pathlib import Path

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
