"""The install doctor must notice a Pillow below the security floor.

Raising the floor in requirements.txt only reaches an install when setup runs
again. Unzipping a new version over an old folder keeps the old .venv, the app
starts normally on the vulnerable Pillow, and diagnose.bat still reported
"All checks passed".
"""

import re
import tomllib
from pathlib import Path

import pytest

import doctor

REPO = Path(__file__).resolve().parent.parent


def _pillow(monkeypatch, installed):
    import importlib.metadata as metadata

    real = metadata.version

    def fake(name):
        if name.lower() == "pillow":
            return installed
        return real(name)

    monkeypatch.setattr(metadata, "version", fake)


@pytest.mark.parametrize(
    ("installed", "flagged"),
    [("12.2.0", True), ("12.3.0", False), ("12.10.1", False)],
)
def test_doctor_flags_a_pillow_below_the_security_floor(
    monkeypatch, capsys, installed, flagged
):
    _pillow(monkeypatch, installed)
    problems = []

    doctor._check_pillow(problems, "setup.bat")

    assert bool(problems) is flagged
    if flagged:
        assert "setup.bat" in problems[0] and installed in problems[0]
        assert "[FAIL]" in capsys.readouterr().out


def _declared_floor(spec_text):
    m = re.search(r"^\s*\"?Pillow>=([\d.]+)", spec_text, re.MULTILINE | re.IGNORECASE)
    assert m, "Pillow floor not found"
    return tuple(int(x) for x in m.group(1).split("."))


def test_doctor_pillow_floor_matches_the_declared_floor():
    requirements = (REPO / "requirements.txt").read_text(encoding="utf-8")
    with open(REPO / "pyproject.toml", "rb") as f:
        dependencies = "\n".join(tomllib.load(f)["project"]["dependencies"])

    assert _declared_floor(requirements) == doctor.PILLOW_MIN
    assert _declared_floor(dependencies) == doctor.PILLOW_MIN
