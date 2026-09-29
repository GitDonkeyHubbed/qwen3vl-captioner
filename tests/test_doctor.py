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


def test_doctor_flags_a_missing_pillow(monkeypatch, capsys):
    """Unzipping over a broken venv can drop Pillow entirely. The app then
    fails on the first image; diagnose used to print nothing about it."""
    import importlib.metadata as metadata

    real = metadata.version

    def fake(name):
        if name.lower() == "pillow":
            raise metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(metadata, "version", fake)
    problems = []

    doctor._check_pillow(problems, "setup.bat")

    assert problems and "setup.bat" in problems[0]
    assert "NOT INSTALLED" in capsys.readouterr().out


def _healthy_linux_report():
    return {
        "platform": "linux",
        "python": "3.12.0",
        "llama_cpp_installed": True,
        "llama_cpp_version": "0.3.40",
        "llama_cpp_importable": True,
        "import_error": None,
    }


def test_main_reports_an_old_pillow_on_linux(monkeypatch, capsys):
    """The helper is not enough: unzip-over-old-venv is diagnosed through
    main(), and Linux is the path that has no other Windows/Mac checks."""
    monkeypatch.setattr(doctor.sys, "platform", "linux")
    monkeypatch.setattr(doctor, "diagnose", _healthy_linux_report)
    _pillow(monkeypatch, "12.2.0")

    assert doctor.main() == 1
    out = capsys.readouterr().out
    assert "PROBLEMS FOUND" in out
    assert "12.2.0" in out
    assert "requirements.txt" in out


def test_huggingface_hub_pin_matches_across_manifests():
    """1.4.4 capped huggingface-hub below 3 in both install files. A drift
    would let one path pull an unchecked major while the other stays capped."""
    requirements = (REPO / "requirements.txt").read_text(encoding="utf-8")
    with open(REPO / "pyproject.toml", "rb") as f:
        dependencies = "\n".join(tomllib.load(f)["project"]["dependencies"])

    req = re.search(r"huggingface-hub\s*([^\s#]+)", requirements, re.IGNORECASE)
    pyproj = re.search(r"huggingface-hub\s*([^\s\"#]+)", dependencies, re.IGNORECASE)
    assert req and pyproj
    assert req.group(1) == pyproj.group(1)
    assert "<3" in req.group(1)
