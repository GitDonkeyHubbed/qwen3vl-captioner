"""Target-preset tests: the Krea 2 and MOSS-Image presets (issue #52) and the
contract every TARGET_PRESETS entry must keep.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui.settings_panel import (  # noqa: E402
    CAPTION_LENGTHS,
    EXTRA_OPTIONS,
    TARGET_PRESETS,
    SettingsPanel,
    _ALL_OPTION_KEYS,
    _build_prompt_krea2,
    _build_prompt_moss,
)


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def _all_on(*except_keys):
    return {key: key not in except_keys for key, _, _ in EXTRA_OPTIONS}


def _preset(preset_id):
    return next(p for p in TARGET_PRESETS if p["id"] == preset_id)


def _build(preset_id, length_key, extra_opts, name_value=""):
    """Build a preset's prompt the way the panel does: unsupported options off."""
    preset = _preset(preset_id)
    opts = {k: v for k, v in extra_opts.items() if k in preset["supported_options"]}
    return preset["prompt_builder"](
        length_key, CAPTION_LENGTHS.get(length_key, ""), opts, name_value
    )


# ── Contract every preset keeps ─────────────────────────────────────────

def test_preset_ids_are_unique():
    ids = [p["id"] for p in TARGET_PRESETS]
    assert len(ids) == len(set(ids))


def test_sd_stays_first():
    # The prompt box is seeded from TARGET_PRESETS[0] at startup; new presets
    # are appended so that stays the Stable Diffusion template.
    assert TARGET_PRESETS[0]["id"] == "sd"


@pytest.mark.parametrize("preset", TARGET_PRESETS, ids=lambda p: p["id"])
def test_every_preset_builds_a_prompt_for_every_allowed_length(preset):
    assert set(preset["supported_options"]) <= _ALL_OPTION_KEYS
    assert set(preset["allowed_lengths"]) <= set(CAPTION_LENGTHS)
    for length_key in preset["allowed_lengths"]:
        prompt = preset["prompt_builder"](
            length_key, CAPTION_LENGTHS[length_key], _all_on(), "Alice"
        )
        assert isinstance(prompt, str) and prompt.strip()


# ── Krea 2 ──────────────────────────────────────────────────────────────

def test_krea2_asks_for_one_prose_paragraph_with_quoted_text():
    prompt = _build("krea2", "Medium", {})
    assert "Krea 2" in prompt
    assert "NOT tags" in prompt
    assert "inside double quotes" in prompt
    assert "if there is no text, do not mention text at all" in prompt
    assert "single paragraph with no line breaks" in prompt


def test_krea2_drops_the_quoting_rule_when_text_is_excluded():
    prompt = _build("krea2", "Medium", {"excludeText": True})
    assert "Ignore text in the image." in prompt
    assert "double quotes" not in prompt


def test_krea2_caption_length_changes_the_prompt():
    prompts = {key: _build("krea2", key, {}) for key in CAPTION_LENGTHS}
    assert len(set(prompts.values())) == len(CAPTION_LENGTHS)
    assert "one or two sentences" in prompts["Short"]
    assert "350 words" in prompts["Descriptive (Longest)"]
    assert "350 words" not in prompts["Long"]


def test_krea2_unknown_length_key_falls_back_to_medium():
    assert _build_prompt_krea2("Nonsense", "", {}, "") == _build_prompt_krea2(
        "Medium", "", {}, ""
    )


def test_krea2_never_asks_for_mood_or_quality():
    # The shared longest-length sentence asks for mood and technical details,
    # and the Aesthetic quality option asks for quality words; Krea 2's own
    # rules forbid both, so neither may reach its prompt.
    for length_key in CAPTION_LENGTHS:
        prompt = _build("krea2", length_key, _all_on())
        assert prompt.count("mood") == 1  # only "Avoid storytelling or mood."
        assert "image quality" not in prompt
        assert CAPTION_LENGTHS[length_key] not in prompt
    assert "includeAestheticQuality" not in _preset("krea2")["supported_options"]


def test_krea2_honours_options_and_names_the_subject_with_its_class():
    prompt = _build("krea2", "Long", {"keepPG": True, "includeLighting": True}, "Alice")
    assert "Keep the description SFW." in prompt
    assert "precise lighting details" in prompt
    assert "Refer to the subject as Alice, followed by what it is" in prompt


def test_krea2_blank_name_does_not_become_a_trigger_word():
    # get_prompt falls back to "the subject" when Refer-as is on but the name
    # field is blank; that must not be turned into "<name>, <class>".
    prompt = _build("krea2", "Long", {}, "the subject")
    assert "followed by what it is" not in prompt


# ── MOSS-Image ──────────────────────────────────────────────────────────

def test_moss_follows_the_openmoss_caption_rules():
    prompt = _build("moss", "Long", {})
    assert "MOSS-Image" in prompt
    assert "English" in prompt
    assert "no tags, JSON, or Markdown" in prompt
    assert "other than watermarks exactly as written" in prompt
    assert "do not translate" in prompt
    assert "position and layout" in prompt
    assert "do not mention text at all" in prompt
    assert "no line breaks" in prompt


def test_moss_short_and_medium_still_keep_text_in_its_original_language():
    for length_key in ("Short", "Medium"):
        prompt = _build("moss", length_key, {})
        assert "do not translate or romanize it" in prompt
        assert "do not mention text at all" in prompt


def test_moss_caption_length_changes_the_prompt():
    prompts = {key: _build("moss", key, {}) for key in CAPTION_LENGTHS}
    assert len(set(prompts.values())) == len(CAPTION_LENGTHS)
    assert "one sentence" in prompts["Short"]
    assert "3-4 sentences" in prompts["Medium"]
    # The guide's text-layout rule is for detailed captions only.
    assert "position and layout" not in prompts["Short"]
    assert "position and layout" not in prompts["Medium"]


def test_moss_unknown_length_key_falls_back_to_medium():
    assert _build_prompt_moss("Nonsense", "", {}, "") == _build_prompt_moss(
        "Medium", "", {}, ""
    )


@pytest.mark.parametrize("length_key", list(CAPTION_LENGTHS))
def test_moss_only_ever_forbids_watermarks(length_key):
    # Even with every option on (as if unsupported ones leaked through), the
    # only watermark wording is the ban and the text-copy exclusion.
    prompt = _build_prompt_moss(length_key, "", _all_on("excludeText"), "")
    assert "Never mention watermarks." in prompt
    assert prompt.count("watermark") == 2


def test_moss_drops_options_the_guide_does_not_allow():
    supported = _preset("moss")["supported_options"]
    for key in ("includeWatermark", "includeTechnicalDetails",
                "includeAestheticQuality", "includeSafety"):
        assert key not in supported
    prompt = _build_prompt_moss("Long", "", _all_on(), "")
    assert "technical camera parameters" not in prompt
    assert "image quality" not in prompt
    assert "NSFW/SFW rating" not in prompt


def test_moss_drops_the_transcription_rule_when_text_is_excluded():
    prompt = _build("moss", "Medium", {"excludeText": True})
    assert "Ignore text in the image." in prompt
    assert "double quotes" not in prompt


# ── Panel behaviour ─────────────────────────────────────────────────────

def test_moss_preset_disables_the_watermark_option_and_restores_it(qapp):
    panel = SettingsPanel()
    watermark = panel._extra_checkboxes["includeWatermark"]
    watermark.setChecked(True)

    panel._apply_preset(_preset("moss"))
    assert watermark.isEnabled() is False
    assert watermark.isChecked() is False
    assert panel.get_prompt().count("watermark") == 2  # ban + text exclusion

    panel._apply_preset(_preset("moss"))  # toggles the preset off
    assert watermark.isEnabled() is True
    assert watermark.isChecked() is True
    panel.deleteLater()


def test_switching_presets_re_checks_an_option_the_last_one_dropped(qapp):
    panel = SettingsPanel()
    watermark = panel._extra_checkboxes["includeWatermark"]
    quality = panel._extra_checkboxes["includeAestheticQuality"]
    watermark.setChecked(True)
    quality.setChecked(True)

    panel._apply_preset(_preset("moss"))      # drops both
    assert (watermark.isChecked(), quality.isChecked()) == (False, False)

    panel._apply_preset(_preset("krea2"))     # supports watermark, not quality
    assert watermark.isChecked() is True
    assert quality.isChecked() is False
    assert quality.isEnabled() is False

    panel._apply_preset(_preset("flux1"))     # supports both
    assert (watermark.isChecked(), quality.isChecked()) == (True, True)
    panel.deleteLater()


def test_an_option_left_unchecked_stays_unchecked(qapp):
    panel = SettingsPanel()
    watermark = panel._extra_checkboxes["includeWatermark"]
    watermark.setChecked(False)

    panel._apply_preset(_preset("moss"))
    panel._apply_preset(_preset("moss"))  # toggles the preset off
    assert watermark.isChecked() is False
    panel.deleteLater()


def test_an_option_unchecked_after_being_restored_stays_unchecked(qapp):
    panel = SettingsPanel()
    watermark = panel._extra_checkboxes["includeWatermark"]

    # Restored by deselecting, then unchecked by hand: a later preset must
    # not re-check it from a stale record.
    watermark.setChecked(True)
    panel._apply_preset(_preset("moss"))
    panel._apply_preset(_preset("moss"))  # toggles the preset off
    watermark.setChecked(False)
    panel._apply_preset(_preset("krea2"))
    assert watermark.isChecked() is False

    # Restored by switching presets, then unchecked by hand: deselecting must
    # not re-check it either.
    panel._apply_preset(_preset("krea2"))  # toggles the preset off
    watermark.setChecked(True)
    panel._apply_preset(_preset("moss"))
    panel._apply_preset(_preset("krea2"))
    watermark.setChecked(False)
    panel._apply_preset(_preset("krea2"))  # toggles the preset off
    assert watermark.isChecked() is False
    panel.deleteLater()


@pytest.mark.parametrize("preset_id,option,banned", [
    ("moss", "includeWatermark", "Never mention watermarks"),
    ("krea2", "includeAestheticQuality", "no quality or score words"),
])
def test_deselecting_restores_the_users_own_prompt(qapp, preset_id, option, banned):
    panel = SettingsPanel()
    panel.prompt_text.setPlainText("Describe this image.")
    panel._extra_checkboxes[option].setChecked(True)

    panel._apply_preset(_preset(preset_id))
    panel._apply_preset(_preset(preset_id))  # toggles the preset off

    # The re-checked option must not be appended to the departed preset's
    # prompt, which forbids it.
    assert panel.prompt_text.toPlainText() == "Describe this image."
    assert banned not in panel.get_prompt()
    panel.deleteLater()


def test_custom_edit_without_changes_still_restores_the_users_prompt(qapp):
    panel = SettingsPanel()
    panel.prompt_text.setPlainText("Describe this image.")

    panel._apply_preset(_preset("moss"))
    panel._toggle_custom_edit()           # Edit, but change nothing
    panel._apply_preset(_preset("moss"))  # toggles the preset off
    assert panel.prompt_text.toPlainText() == "Describe this image."
    panel.deleteLater()


def test_deselecting_restores_custom_edit_mode_with_the_prompt(qapp):
    panel = SettingsPanel()
    panel._toggle_custom_edit()  # no preset: the user's text is sent verbatim
    panel.prompt_text.setPlainText("Return only the dominant color.")
    assert panel.get_prompt() == "Return only the dominant color."

    panel._apply_preset(_preset("moss"))
    panel._apply_preset(_preset("moss"))  # toggles the preset off
    assert panel._custom_edit_mode is True
    assert panel.custom_edit_btn.text() == "↻ Reset"
    assert panel.get_prompt() == "Return only the dominant color."
    panel.deleteLater()


def test_custom_edit_with_changes_keeps_the_users_new_prompt(qapp):
    panel = SettingsPanel()
    panel.prompt_text.setPlainText("Describe this image.")

    panel._apply_preset(_preset("moss"))
    panel._toggle_custom_edit()
    panel.prompt_text.setPlainText("My own MOSS prompt.")
    panel._apply_preset(_preset("moss"))  # toggles the preset off
    assert panel.prompt_text.toPlainText() == "My own MOSS prompt."
    panel.deleteLater()


@pytest.mark.parametrize("preset_id", ["krea2", "moss"])
def test_new_presets_lock_the_prompt_and_leave_user_affixes_alone(qapp, preset_id):
    panel = SettingsPanel()
    panel.prefix_input.setText("photo of")
    panel.suffix_input.setText("by me")

    panel._apply_preset(_preset(preset_id))
    assert panel.get_active_preset() == preset_id
    assert panel.prompt_text.isReadOnly() is True
    assert panel.get_prefix() == ""
    assert panel.get_suffix() == ""
    assert panel.prompt_text.toPlainText() == panel.get_prompt()

    panel._apply_preset(_preset(preset_id))  # toggles the preset off
    assert panel.get_active_preset() is None
    assert panel.get_prefix() == "photo of"
    assert panel.get_suffix() == "by me"
    panel.deleteLater()


def test_switching_presets_does_not_overwrite_the_stashed_prompt(qapp):
    """Pony → Krea must not stash Pony's tokens or drop a pre-preset custom edit.

    The stash is taken only on the first entry into preset mode. Overwriting it
    on the second click restores the outgoing preset's prompt and prefix, and
    leaves custom edit off, so get_prompt() appends length text to a prompt
    that was supposed to be sent verbatim.
    """
    panel = SettingsPanel()
    panel._toggle_custom_edit()
    panel.prompt_text.setPlainText("Return only the dominant color.")
    panel.prefix_input.setText("photo of")
    panel.suffix_input.setText("by me")

    panel._apply_preset(_preset("pony"))
    assert panel._custom_edit_mode is False
    assert panel.get_prefix() == "score_9, score_8_up, score_7_up,"
    assert panel.get_suffix() == ", rating_safe"
    assert "dominant color" not in panel.get_prompt()

    panel._apply_preset(_preset("krea2"))
    assert panel._custom_edit_mode is False
    assert panel.custom_edit_btn.text() == "✏ Edit"
    assert panel.prompt_text.isReadOnly() is True
    assert panel.prompt_text.toPlainText() == panel.get_prompt()
    assert "Krea 2" in panel.get_prompt()
    assert "dominant color" not in panel.get_prompt()
    assert "score_9" not in panel.get_prompt()
    assert panel.get_prefix() == ""

    panel._apply_preset(_preset("krea2"))  # toggles the preset off
    assert panel._custom_edit_mode is True
    assert panel.custom_edit_btn.text() == "↻ Reset"
    assert panel.get_prompt() == "Return only the dominant color."
    assert panel.get_prefix() == "photo of"
    assert panel.get_suffix() == "by me"
    panel.deleteLater()


def test_a_custom_rewrite_does_not_become_the_next_presets_prompt(qapp):
    panel = SettingsPanel()
    panel.prompt_text.setPlainText("Describe this image.")
    panel._apply_preset(_preset("moss"))
    panel._toggle_custom_edit()
    panel.prompt_text.setPlainText("My MOSS rewrite.")

    panel._apply_preset(_preset("krea2"))
    assert panel._custom_edit_mode is False
    assert panel.custom_edit_btn.text() == "✏ Edit"
    assert panel.prompt_text.toPlainText() == panel.get_prompt()
    assert "Krea 2" in panel.get_prompt()
    assert "My MOSS rewrite." not in panel.get_prompt()

    panel._apply_preset(_preset("krea2"))  # toggles the preset off
    assert panel.prompt_text.toPlainText() == "Describe this image."
    assert panel._custom_edit_mode is False
    prompt = panel.get_prompt()
    assert prompt.startswith("Describe this image.")
    assert "Provide a moderately detailed description in 2-4 sentences." in prompt
    assert "My MOSS rewrite." not in prompt
    assert "Never mention watermarks" not in prompt
    panel.deleteLater()


def test_a_rewrite_is_not_sent_verbatim_just_because_custom_edit_was_on(qapp):
    """Editing the preset's own preview wins over the pre-preset custom-edit flag.

    Restoring that flag anyway would send the rewrite with no length sentence.
    Restoring the old verbatim prompt would throw the rewrite away.
    """
    panel = SettingsPanel()
    panel._toggle_custom_edit()
    panel.prompt_text.setPlainText("Return only the dominant color.")
    panel._apply_preset(_preset("moss"))
    panel._toggle_custom_edit()
    panel.prompt_text.setPlainText("Only the dominant color, nothing else.")

    panel._apply_preset(_preset("moss"))  # toggles the preset off
    assert panel.prompt_text.toPlainText() == "Only the dominant color, nothing else."
    assert panel._custom_edit_mode is False
    assert panel.custom_edit_btn.text() == "✏ Edit"
    prompt = panel.get_prompt()
    assert prompt.startswith("Only the dominant color, nothing else.")
    assert prompt != "Only the dominant color, nothing else."
    assert "Provide a moderately detailed description in 2-4 sentences." in prompt
    assert "Return only the dominant color." not in prompt
    panel.deleteLater()


def test_an_empty_custom_prompt_is_restored_on_deselect(qapp):
    panel = SettingsPanel()
    panel._toggle_custom_edit()
    panel.prompt_text.setPlainText("")
    assert panel.get_prompt() == ""

    panel._apply_preset(_preset("moss"))
    assert "MOSS-Image" in panel.get_prompt()
    panel._apply_preset(_preset("moss"))  # toggles the preset off
    assert panel.prompt_text.toPlainText() == ""
    assert panel._custom_edit_mode is True
    assert panel.get_prompt() == ""
    panel.deleteLater()


def test_rechecked_options_change_the_caption_not_just_the_checkbox(qapp):
    panel = SettingsPanel()
    for key in (
        "includeWatermark",
        "includeSafety",
        "includeTechnicalDetails",
        "includeAestheticQuality",
    ):
        panel._extra_checkboxes[key].setChecked(True)

    panel._apply_preset(_preset("krea2"))
    prompt = panel.get_prompt()
    assert "Note if a watermark is present." in prompt
    assert "State the NSFW/SFW rating." in prompt
    assert "technical camera parameters" in prompt
    assert "image quality" not in prompt
    assert "subjective aesthetic" not in prompt

    panel._apply_preset(_preset("moss"))
    prompt = panel.get_prompt()
    assert panel._extra_checkboxes["includeWatermark"].isChecked() is False
    assert panel._extra_checkboxes["includeAestheticQuality"].isChecked() is False
    assert "Note if a watermark is present." not in prompt
    assert "technical camera parameters" not in prompt
    assert "NSFW/SFW rating" not in prompt
    assert "image quality" not in prompt
    assert prompt.count("watermark") == 2

    panel._apply_preset(_preset("krea2"))
    prompt = panel.get_prompt()
    assert panel._extra_checkboxes["includeWatermark"].isChecked() is True
    assert panel._extra_checkboxes["includeSafety"].isChecked() is True
    assert panel._extra_checkboxes["includeTechnicalDetails"].isChecked() is True
    assert panel._extra_checkboxes["includeAestheticQuality"].isChecked() is False
    assert "Note if a watermark is present." in prompt
    assert "State the NSFW/SFW rating." in prompt
    assert "technical camera parameters" in prompt
    assert "image quality" not in prompt
    panel.deleteLater()


@pytest.mark.parametrize(
    "preset_id,key,phrase",
    [
        ("krea2", "includeCameraAngle", "exact camera angle"),
        ("krea2", "includeComposition", "composition style"),
        ("krea2", "includeDOF", "focal depth"),
        ("krea2", "includeLightSource", "light sources"),
        ("krea2", "includeUncensored", "Do not use euphemisms."),
        ("krea2", "excludeStaticAttributes", "Do not describe unchangeable physical attributes."),
        ("krea2", "excludeResolution", "Ignore resolution."),
        ("krea2", "includeArtifacts", "Note any image artifacts."),
        ("krea2", "noAmbiguity", "Use precise, clinical language."),
        ("moss", "includeLighting", "precise lighting details"),
        ("moss", "includeCameraAngle", "exact camera angle"),
        ("moss", "includeComposition", "composition style"),
        ("moss", "includeDOF", "focal depth"),
        ("moss", "includeLightSource", "light sources"),
        ("moss", "keepPG", "Keep the description SFW."),
        ("moss", "includeUncensored", "Do not use euphemisms."),
        ("moss", "excludeStaticAttributes", "Do not describe unchangeable physical attributes."),
        ("moss", "excludeResolution", "Ignore resolution."),
        ("moss", "includeArtifacts", "Note any image artifacts."),
        ("moss", "noAmbiguity", "Use precise, clinical language."),
    ],
)
def test_option_sentence_is_present_only_when_that_option_is_on(preset_id, key, phrase):
    assert phrase in _build(preset_id, "Medium", {key: True})
    assert phrase not in _build(preset_id, "Medium", {})


def test_refer_as_blank_is_the_subject_and_does_not_leak_when_off(qapp):
    panel = SettingsPanel()
    refer = panel._extra_checkboxes["referAsName"]
    refer.setChecked(True)
    panel.name_input.setText("   ")

    panel._apply_preset(_preset("krea2"))
    prompt = panel.get_prompt()
    assert "Refer to the subject as the subject." in prompt
    assert "followed by what it is" not in prompt

    panel.name_input.setText(" Alice ")
    prompt = panel.get_prompt()
    assert "Refer to the subject as Alice, followed by what it is" in prompt

    refer.setChecked(False)
    assert "Alice" not in panel.get_prompt()

    refer.setChecked(True)
    panel._apply_preset(_preset("moss"))
    prompt = panel.get_prompt()
    assert "Refer to the subject as Alice." in prompt
    assert "followed by what it is" not in prompt

    panel.name_input.setText("\t")
    prompt = panel.get_prompt()
    assert "Refer to the subject as the subject." in prompt
    assert "Alice" not in prompt
    panel.deleteLater()


def test_descriptive_length_returns_after_a_tag_preset_and_survives_the_switch(qapp):
    panel = SettingsPanel()
    panel.caption_length_combo.setCurrentText("Descriptive (Longest)")

    panel._apply_preset(_preset("sd"))
    labels = [
        panel.caption_length_combo.itemText(i)
        for i in range(panel.caption_length_combo.count())
    ]
    assert labels == ["Short", "Medium", "Long"]
    assert panel.caption_length_combo.currentText() == "Short"
    assert "5-15 tags" in panel.get_prompt()
    assert "50+ tags" not in panel.get_prompt()

    panel._apply_preset(_preset("krea2"))
    labels = [
        panel.caption_length_combo.itemText(i)
        for i in range(panel.caption_length_combo.count())
    ]
    assert labels == list(CAPTION_LENGTHS)
    assert panel.caption_length_combo.currentText() == "Short"
    assert "one or two sentences" in panel.get_prompt()
    panel.caption_length_combo.setCurrentText("Descriptive (Longest)")
    assert "350 words" in panel.get_prompt()
    assert "one or two sentences" not in panel.get_prompt()

    panel._apply_preset(_preset("moss"))
    assert panel.caption_length_combo.currentText() == "Descriptive (Longest)"
    prompt = panel.get_prompt()
    assert "exhaustive paragraph" in prompt
    assert "position and layout" in prompt
    assert "350 words" not in prompt
    panel.deleteLater()
