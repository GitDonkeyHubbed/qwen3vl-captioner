"""Tests for the model registry, backend routing, and dropdown grouping
(gui.model_download_manager).

Covers registry integrity (every GGUF model paired with an mmproj vision
encoder), backend routing (``gguf`` vs ``mlx``), the recommended default, and
the parallel group/label structure that drives the dropdown headers.
"""

from gui.model_download_manager import (
    MLX_MODEL_REGISTRY,
    MODEL_REGISTRY,
    get_all_model_display_names,
    get_model_group_labels,
    get_model_groups,
    get_model_info,
    mlx_backend_supported,
    mlx_model_exists,
    model_file_exists,
)


def test_gguf_registry_non_empty():
    assert len(MODEL_REGISTRY) >= 1


def test_every_gguf_entry_has_required_fields():
    for entry in MODEL_REGISTRY.values():
        assert entry["repo_id"]
        assert entry["filename"].endswith(".gguf")
        # Every GGUF model must be paired with an mmproj vision encoder,
        # otherwise the model loads but produces blank/garbage captions.
        assert entry["mmproj_filename"].endswith(".gguf")
        assert isinstance(entry["size_gb"], (int, float))
        assert isinstance(entry["recommended"], bool)


def test_only_v2_family_is_recommended():
    recommended = [n for n, e in MODEL_REGISTRY.items() if e["recommended"]]
    assert recommended, "expected at least one recommended default model"
    for name in recommended:
        assert "v2" in name


def test_v2_filename_and_mmproj_templating():
    entry = MODEL_REGISTRY["Qwen3-VL 8B ABL v2 — Q4_K_M (4.68 GB)"]
    assert entry["filename"] == "Qwen3-VL-8B-Instruct-abliterated-v2.Q4_K_M.gguf"
    assert entry["mmproj_filename"] == (
        "Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-f16.gguf"
    )
    assert entry["recommended"] is True


def test_huihui_uses_dash_quant_template():
    # The huihui family uses "{stem}-{quant}.gguf", not "{stem}.{quant}.gguf".
    entry = MODEL_REGISTRY["Huihui Qwen3-VL 8B ABL — Q4_K_M (4.68 GB)"]
    assert entry["filename"] == "Huihui-Qwen3-VL-8B-Instruct-abliterated-Q4_K_M.gguf"
    assert entry["mmproj_filename"] == "mmproj-F16.gguf"


def test_mlx_registry_entries_are_folder_based():
    assert len(MLX_MODEL_REGISTRY) >= 1
    for entry in MLX_MODEL_REGISTRY.values():
        assert entry["backend"] == "mlx"
        assert entry["repo_id"]
        assert entry["folder"]
        # MLX models embed the vision tower — no separate mmproj file.
        assert "mmproj_filename" not in entry


def test_get_model_info_routes_gguf():
    info = get_model_info("Qwen3-VL 8B ABL v2 — Q4_K_M (4.68 GB)")
    assert info is not None
    assert info["backend"] == "gguf"
    assert info["mmproj_filename"].endswith(".gguf")


def test_get_model_info_routes_mlx():
    info = get_model_info("Qwen3-VL 8B ABL v2 MLX — 4bit (5.4 GB)")
    assert info is not None
    assert info["backend"] == "mlx"
    assert info["folder"]


def test_get_model_info_unknown_returns_none():
    assert get_model_info("Totally Not A Model") is None


def test_no_duplicate_display_names():
    combined = list(MODEL_REGISTRY) + list(MLX_MODEL_REGISTRY)
    assert len(combined) == len(set(combined))


def test_groups_and_labels_are_parallel():
    groups = get_model_groups()
    labels = get_model_group_labels()
    assert len(groups) == len(labels)
    assert all(isinstance(label, str) and label for label in labels)
    assert all(len(group) >= 1 for group in groups)


def test_groups_flatten_to_display_names():
    flat = [name for group in get_model_groups() for name in group]
    assert flat == get_all_model_display_names()


def test_display_names_are_known_registry_entries():
    known = set(MODEL_REGISTRY) | set(MLX_MODEL_REGISTRY)
    for name in get_all_model_display_names():
        assert name in known


def test_mlx_backend_supported_matches_platform():
    import platform
    import sys

    assert mlx_backend_supported() == (
        sys.platform == "darwin" and platform.machine() == "arm64"
    )


def test_model_file_exists(tmp_path):
    assert model_file_exists(tmp_path, "x.gguf") is False
    (tmp_path / "x.gguf").write_bytes(b"\x00")
    assert model_file_exists(tmp_path, "x.gguf") is True


def test_mlx_model_exists(tmp_path):
    assert mlx_model_exists(tmp_path, "m") is False
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    # config.json alone is not enough — needs safetensors too.
    assert mlx_model_exists(tmp_path, "m") is False
    (folder / "model.safetensors").write_bytes(b"\x00")
    assert mlx_model_exists(tmp_path, "m") is True
