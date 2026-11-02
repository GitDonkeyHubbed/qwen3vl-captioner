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
    # Dropping the fallback branch leaves this green if we only forbid the
    # class sentence, so the plain "the subject" wording has to be present.
    prompt = _build("krea2", "Long", {}, "the subject")
    assert "followed by what it is" not in prompt
    assert "Refer to the subject as the subject." in prompt
    assert "Refer to the subject" not in _build_prompt_krea2("Long", "", {}, "")


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


# Phrases that appear only in an option branch. The base prompts already say
# "lighting" and "composition", so those words alone would not catch a deleted
# branch or a sentence that leaked into every caption.
_PRESET_OPTION_SENTENCES = [
    ("krea2", _build_prompt_krea2, "includeLighting", "precise lighting details"),
    ("krea2", _build_prompt_krea2, "includeCameraAngle", "exact camera angle"),
    ("krea2", _build_prompt_krea2, "includeTechnicalDetails", "technical camera parameters"),
    ("krea2", _build_prompt_krea2, "includeComposition", "composition style"),
    ("krea2", _build_prompt_krea2, "includeDOF", "focal depth"),
    ("krea2", _build_prompt_krea2, "includeLightSource", "light sources"),
    ("krea2", _build_prompt_krea2, "keepPG", "Keep the description SFW."),
    ("krea2", _build_prompt_krea2, "includeUncensored", "Do not use euphemisms."),
    ("krea2", _build_prompt_krea2, "excludeStaticAttributes", "Do not describe unchangeable physical attributes."),
    ("krea2", _build_prompt_krea2, "excludeText", "Ignore text in the image."),
    ("krea2", _build_prompt_krea2, "excludeResolution", "Ignore resolution."),
    ("krea2", _build_prompt_krea2, "includeWatermark", "Note if a watermark is present."),
    ("krea2", _build_prompt_krea2, "includeArtifacts", "Note any image artifacts."),
    ("krea2", _build_prompt_krea2, "includeSafety", "State the NSFW/SFW rating."),
    ("krea2", _build_prompt_krea2, "noAmbiguity", "Use precise, clinical language."),
    ("moss", _build_prompt_moss, "includeLighting", "precise lighting details"),
    ("moss", _build_prompt_moss, "includeCameraAngle", "exact camera angle"),
    ("moss", _build_prompt_moss, "includeComposition", "composition style"),
    ("moss", _build_prompt_moss, "includeDOF", "focal depth"),
    ("moss", _build_prompt_moss, "includeLightSource", "light sources"),
    ("moss", _build_prompt_moss, "keepPG", "Keep the description SFW."),
    ("moss", _build_prompt_moss, "includeUncensored", "Do not use euphemisms."),
    ("moss", _build_prompt_moss, "excludeStaticAttributes", "Do not describe unchangeable physical attributes."),
    ("moss", _build_prompt_moss, "excludeText", "Ignore text in the image."),
    ("moss", _build_prompt_moss, "excludeResolution", "Ignore resolution."),
    ("moss", _build_prompt_moss, "includeArtifacts", "Note any image artifacts."),
    ("moss", _build_prompt_moss, "noAmbiguity", "Use precise, clinical language."),
]


@pytest.mark.parametrize(
    "builder,key,phrase",
    [(builder, key, phrase) for _, builder, key, phrase in _PRESET_OPTION_SENTENCES],
    ids=[f"{preset}-{key}" for preset, _, key, _ in _PRESET_OPTION_SENTENCES],
)
def test_new_preset_option_is_opt_in(builder, key, phrase):
    off = builder("Medium", "", {}, "")
    on = builder("Medium", "", {key: True}, "")
    assert phrase not in off
    assert phrase in on


def test_krea2_ignores_a_leaked_aesthetic_quality_flag():
    # The other Krea tests go through _build(), which strips options the preset
    # does not support. A new includeAestheticQuality branch would stay green.
    prompt = _build_prompt_krea2(
        "Long", "", {"includeAestheticQuality": True, "includeLighting": True}, ""
    )
    assert "image quality" not in prompt
    assert "precise lighting details" in prompt


def test_moss_refer_as_does_not_use_krea_trigger_word_wording():
    named = _build_prompt_moss("Medium", "", {}, "Alice")
    assert "Refer to the subject as Alice." in named
    assert "followed by what it is" not in named
    # Krea 2 special-cases the blank-name fallback. MOSS must not: "the subject"
    # is still the name, and an empty name adds no refer-as sentence at all.
    fallback = _build_prompt_moss("Medium", "", {}, "the subject")
    assert "Refer to the subject as the subject." in fallback
    assert "Refer to the subject" not in _build_prompt_moss("Medium", "", {}, "")


def test_krea2_refer_as_follows_the_name_field(qapp):
    panel = SettingsPanel()
    panel._apply_preset(_preset("krea2"))
    panel._extra_checkboxes["referAsName"].setChecked(True)

    assert "Refer to the subject as the subject." in panel.prompt_text.toPlainText()
    assert panel.prompt_text.toPlainText() == panel.get_prompt()

    panel.name_input.setText("   ")
    assert "followed by what it is" not in panel.get_prompt()
    assert panel.prompt_text.toPlainText() == panel.get_prompt()

    panel.name_input.setText("Alice")
    assert "followed by what it is" in panel.prompt_text.toPlainText()
    assert panel.prompt_text.toPlainText() == panel.get_prompt()

    panel._extra_checkboxes["referAsName"].setChecked(False)
    assert "Alice" not in panel.get_prompt()
    assert "followed by what it is" not in panel.get_prompt()
    assert panel.prompt_text.toPlainText() == panel.get_prompt()
    panel.deleteLater()


def test_custom_edit_is_sent_verbatim_and_then_as_the_generic_base(qapp):
    panel = SettingsPanel()
    panel._extra_checkboxes["includeWatermark"].setChecked(True)
    panel._apply_preset(_preset("moss"))
    panel._toggle_custom_edit()
    panel.prompt_text.setPlainText("  My own MOSS prompt.  ")

    # While Edit is on, the caption is the box (stripped), not the builder —
    # even though watermark was on before the preset cleared it.
    assert panel.get_prompt() == "My own MOSS prompt."
    assert "watermark" not in panel.get_prompt().lower()

    panel._apply_preset(_preset("moss"))  # toggles the preset off
    assert panel.get_active_preset() is None
    sent = panel.get_prompt()
    assert sent.startswith("My own MOSS prompt.")
    # Generic mode appends the restored watermark option. The MOSS builder,
    # which forbids watermarks, must not be the caption anymore.
    assert "Never mention watermarks." not in sent
    assert "whether there is a watermark or not" in sent
    panel.deleteLater()


def test_reverted_custom_edit_still_restores_the_users_prompt(qapp):
    # Equality with the last preview decides "untouched", not the fact that
    # Edit was opened or that the box changed along the way.
    panel = SettingsPanel()
    panel.prompt_text.setPlainText("Describe this image.")
    panel._apply_preset(_preset("moss"))
    preview = panel.prompt_text.toPlainText()
    panel._toggle_custom_edit()
    panel.prompt_text.setPlainText("A temporary rewrite.")
    panel.prompt_text.setPlainText(preview)
    panel._apply_preset(_preset("moss"))
    assert panel.prompt_text.toPlainText() == "Describe this image."
    panel.deleteLater()


def test_length_change_during_custom_edit_keeps_the_rewrite(qapp):
    # Refreshing the preview while Edit is on would store the rewrite as the
    # "last preset preview", and deselect would then treat it as untouched
    # and throw it away.
    panel = SettingsPanel()
    panel.prompt_text.setPlainText("Describe this image.")
    panel._apply_preset(_preset("moss"))
    preview = panel.prompt_text.toPlainText()
    panel._toggle_custom_edit()
    panel.caption_length_combo.setCurrentText("Long")
    assert panel.prompt_text.toPlainText() == preview
    assert panel.get_prompt() == preview

    panel.prompt_text.setPlainText("Keep this rewrite.")
    panel.caption_length_combo.setCurrentText("Descriptive (Longest)")
    assert panel.prompt_text.toPlainText() == "Keep this rewrite."
    assert "exhaustive" not in panel.get_prompt()
    panel._apply_preset(_preset("moss"))
    assert panel.prompt_text.toPlainText() == "Keep this rewrite."
    panel.deleteLater()


def test_reset_rebuilds_the_preset_at_the_length_chosen_while_editing(qapp):
    panel = SettingsPanel()
    panel.prompt_text.setPlainText("Describe this image.")
    panel._apply_preset(_preset("krea2"))
    panel._toggle_custom_edit()
    panel.prompt_text.setPlainText("My rewrite.")
    panel.caption_length_combo.setCurrentText("Descriptive (Longest)")
    assert panel.get_prompt() == "My rewrite."

    panel._toggle_custom_edit()  # Reset
    rebuilt = panel.get_prompt()
    assert "350 words" in rebuilt
    assert "My rewrite." not in rebuilt
    assert panel.prompt_text.toPlainText() == rebuilt

    panel._apply_preset(_preset("krea2"))
    assert panel.prompt_text.toPlainText() == "Describe this image."
    panel.deleteLater()


def test_switching_presets_discards_an_unfinished_custom_edit(qapp):
    # The pre-preset prompt and affixes are stashed only on the way in.
    # Switching must not replace that stash with the outgoing preset, and
    # must not keep a half-written custom edit as the new preset's caption.
    panel = SettingsPanel()
    panel.prompt_text.setPlainText("Describe this image.")
    panel.prefix_input.setText("pre")
    panel.suffix_input.setText("suf")
    panel._apply_preset(_preset("moss"))
    panel._toggle_custom_edit()
    panel.prompt_text.setPlainText("Half-written MOSS prompt.")

    panel._apply_preset(_preset("krea2"))
    assert panel.get_active_preset() == "krea2"
    assert panel.prompt_text.isReadOnly() is True
    assert "Krea 2" in panel.get_prompt()
    assert "Half-written" not in panel.get_prompt()
    assert panel.prompt_text.toPlainText() == panel.get_prompt()

    panel._apply_preset(_preset("krea2"))
    assert panel.get_active_preset() is None
    assert panel.prompt_text.toPlainText() == "Describe this image."
    assert panel.get_prefix() == "pre"
    assert panel.get_suffix() == "suf"
    assert panel.get_prompt().startswith("Describe this image.")
    panel.deleteLater()


def test_unchecking_a_supported_option_stays_off_and_updates_the_preview(qapp):
    panel = SettingsPanel()
    lighting = panel._extra_checkboxes["includeLighting"]
    lighting.setChecked(True)
    panel._apply_preset(_preset("krea2"))
    assert "precise lighting details" in panel.get_prompt()

    lighting.setChecked(False)
    assert "precise lighting details" not in panel.prompt_text.toPlainText()
    assert panel.prompt_text.toPlainText() == panel.get_prompt()
    panel._toggle_custom_edit()
    assert "precise lighting details" not in panel.get_prompt()
    panel._toggle_custom_edit()

    # Both presets support Lighting, so a re-check here is the stash treating
    # a hand-uncheck as something the preset cleared.
    panel._apply_preset(_preset("moss"))
    assert lighting.isChecked() is False
    assert "precise lighting details" not in panel.get_prompt()
    panel._apply_preset(_preset("moss"))
    assert lighting.isChecked() is False
    panel.deleteLater()


def test_custom_edit_mode_survives_a_preset_switch(qapp):
    # The mode is stashed with the prompt, and only on the way into preset
    # mode. A later preset must not overwrite that with the off switch that
    # applying a preset does.
    panel = SettingsPanel()
    panel._toggle_custom_edit()
    panel.prompt_text.setPlainText("Return only the dominant color.")
    panel._extra_checkboxes["includeWatermark"].setChecked(True)

    panel._apply_preset(_preset("moss"))
    panel._apply_preset(_preset("krea2"))
    panel._apply_preset(_preset("krea2"))

    assert panel._custom_edit_mode is True
    assert panel.custom_edit_btn.text() == "↻ Reset"
    assert panel.get_prompt() == "Return only the dominant color."
    assert "watermark" not in panel.get_prompt().lower()
    panel.deleteLater()


def test_custom_edit_left_off_is_not_restored(qapp):
    panel = SettingsPanel()
    panel._toggle_custom_edit()
    panel._toggle_custom_edit()
    panel.prompt_text.setPlainText("Describe this image.")

    panel._apply_preset(_preset("moss"))
    panel._apply_preset(_preset("moss"))
    assert panel._custom_edit_mode is False
    assert panel.custom_edit_btn.text() == "✏ Edit"
    sent = panel.get_prompt()
    assert sent.startswith("Describe this image.")
    assert sent != "Describe this image."
    panel.deleteLater()


def test_rewriting_the_preset_prompt_does_not_restore_custom_edit(qapp):
    # Mode comes back only together with the pre-preset prompt. A rewrite of
    # the preset text is kept as the generic base, length and options included.
    panel = SettingsPanel()
    panel._toggle_custom_edit()
    panel.prompt_text.setPlainText("Return only the dominant color.")
    panel._apply_preset(_preset("moss"))
    panel._toggle_custom_edit()
    panel.prompt_text.setPlainText("A different prompt.")
    panel._apply_preset(_preset("moss"))

    assert panel._custom_edit_mode is False
    assert panel.prompt_text.toPlainText() == "A different prompt."
    sent = panel.get_prompt()
    assert sent.startswith("A different prompt.")
    assert sent != "A different prompt."
    panel.deleteLater()


def test_caption_length_survives_a_switch_between_the_new_presets(qapp):
    panel = SettingsPanel()
    panel._apply_preset(_preset("moss"))
    panel.caption_length_combo.setCurrentText("Descriptive (Longest)")
    assert "exhaustive" in panel.get_prompt()
    assert "position and layout" in panel.get_prompt()

    panel._apply_preset(_preset("krea2"))
    assert panel.get_caption_length() == "Descriptive (Longest)"
    prompt = panel.get_prompt()
    assert "350 words" in prompt
    assert "exhaustive" not in prompt
    assert "position and layout" not in prompt
    assert panel.prompt_text.toPlainText() == prompt
    panel.deleteLater()
