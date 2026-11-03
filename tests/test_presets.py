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
    assert "Write a single paragraph" in prompt
    assert "with no line breaks, lists, or Markdown" in prompt


def test_krea2_drops_the_quoting_rule_when_text_is_excluded():
    for length_key in CAPTION_LENGTHS:
        prompt = _build("krea2", length_key, {"excludeText": True})
        assert "Ignore text in the image." in prompt
        assert "double quotes" not in prompt
        # Short normally asks for visible text; it must not contradict
        # "Ignore text".
        assert "visible text" not in prompt
        assert "quoted text" not in prompt
    assert "one or two short sentences, under 40 words:" in _build(
        "krea2", "Short", {"excludeText": True}
    )


def test_krea2_caption_length_changes_the_prompt():
    prompts = {key: _build("krea2", key, {}) for key in CAPTION_LENGTHS}
    assert len(set(prompts.values())) == len(CAPTION_LENGTHS)
    assert "under 40 words not counting quoted text" in prompts["Short"]
    assert "and any visible text" in prompts["Short"]
    assert "never exceed 300 words" in prompts["Descriptive (Longest)"]
    assert "groups of similar people or objects" in prompts["Descriptive (Longest)"]
    assert "300 words" not in prompts["Long"]
    # Medium alone caps sentence length: the model wrote one run-on sentence
    # without it, and looped on short sentences when Long had the same cap.
    assert "each under about 30 words" in prompts["Medium"]
    for key in ("Short", "Long", "Descriptive (Longest)"):
        assert "each under about 30 words" not in prompts[key]
        assert "Write a single paragraph with no line breaks" in prompts[key]


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


def test_krea2_length_rules_stay_on_their_own_length():
    # The 40-word line, the 30-word sentence cap, and the 300-word cap were
    # tuned separately. Copying any one of them onto the other lengths, or
    # using the short no-text line for every length once text is excluded,
    # makes Long and Descriptive the same caption.
    plain = {key: _build("krea2", key, {}) for key in CAPTION_LENGTHS}
    excluded = {
        key: _build("krea2", key, {"excludeText": True}) for key in CAPTION_LENGTHS
    }
    assert len(set(excluded.values())) == len(CAPTION_LENGTHS)

    assert "under 40 words" not in plain["Medium"]
    assert "under 40 words" not in plain["Long"]
    assert "under 40 words" not in plain["Descriptive (Longest)"]
    assert "3-5 sentences" not in plain["Short"]
    assert "300 words" not in plain["Short"]
    assert "300 words" not in plain["Medium"]
    assert "groups of similar" not in plain["Short"]
    assert "groups of similar" not in plain["Medium"]
    assert "groups of similar" not in plain["Long"]
    assert "3-5 sentences" not in plain["Descriptive (Longest)"]

    short_off = excluded["Short"]
    assert "one or two short sentences, under 40 words:" in short_off
    assert "3-5 sentences" not in short_off
    assert "long, detailed paragraph" not in short_off
    assert "300 words" not in short_off
    assert CAPTION_LENGTHS["Short"] not in short_off

    medium_off = excluded["Medium"]
    assert "Write about 3-5 sentences." in medium_off
    assert "each under about 30 words" in medium_off
    assert "under 40 words" not in medium_off
    assert CAPTION_LENGTHS["Medium"] not in medium_off

    long_off = excluded["Long"]
    assert "Write a long, detailed paragraph." in long_off
    assert "Write a single paragraph with no line breaks" in long_off
    assert "each under about 30 words" not in long_off
    assert "under 40 words" not in long_off
    assert "300 words" not in long_off
    assert CAPTION_LENGTHS["Long"] not in long_off

    descriptive_off = excluded["Descriptive (Longest)"]
    assert "never exceed 300 words" in descriptive_off
    assert "groups of similar people or objects" in descriptive_off
    assert "under 40 words" not in descriptive_off
    assert "each under about 30 words" not in descriptive_off
    assert CAPTION_LENGTHS["Descriptive (Longest)"] not in descriptive_off

    # A caller-supplied length sentence must not replace the no-text Short line.
    leaked = "LEAKED SHARED LENGTH SENTENCE"
    leaked_short = _build_prompt_krea2("Short", leaked, {"excludeText": True}, "")
    assert leaked not in leaked_short
    assert "one or two short sentences, under 40 words:" in leaked_short
    leaked_long = _build_prompt_krea2("Long", leaked, {"excludeText": True}, "")
    assert leaked not in leaked_long
    assert "under 40 words" not in leaked_long


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
    # Without this the model added guesses and mood ("cozy", "suggesting
    # daytime play") that the guide's "only what is visible" rules out.
    for length_key in CAPTION_LENGTHS:
        assert "Do not speculate or interpret" in _build("moss", length_key, {})


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


def test_moss_speculation_ban_blocks_the_shared_length_sentence():
    # The ban is the only place "mood" may appear. Appending CAPTION_LENGTHS
    # would add the shared longest sentence's mood and technical details on
    # top of it, which the guide does not allow. Dropping the ban leaves no
    # "mood" at all. Excluding text must not drop the ban or the length line
    # with the transcription rule.
    leaked = "LEAKED SHARED LENGTH SENTENCE"
    for length_key, shared in CAPTION_LENGTHS.items():
        prompt = _build("moss", length_key, {})
        assert prompt.count("mood") == 1
        assert "no mood or evaluative words" in prompt
        assert shared not in prompt
        assert "technical details" not in prompt
        assert leaked not in _build_prompt_moss(length_key, leaked, {}, "")

    detailed = _build("moss", "Descriptive (Longest)", {})
    assert "position and layout" in detailed
    assert "exhaustive paragraph" in detailed

    excluded = _build("moss", "Descriptive (Longest)", {"excludeText": True})
    assert "Do not speculate or interpret" in excluded
    assert "no mood or evaluative words" in excluded
    assert "position and layout" not in excluded
    assert "double quotes" not in excluded
    assert "Ignore text in the image." in excluded
    assert "exhaustive paragraph" in excluded
    assert CAPTION_LENGTHS["Descriptive (Longest)"] not in excluded


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
    # Still in custom edit, so the edited prompt goes out exactly as typed.
    assert panel._custom_edit_mode is True
    assert panel.get_prompt() == "My own MOSS prompt."
    panel.deleteLater()


def test_krea2_panel_length_and_exclude_text_reach_the_prompt(qapp):
    # The builder tests pass the length and the checkbox in directly. This
    # is the path the caption button uses: the combo and the checkbox.
    panel = SettingsPanel()
    panel._apply_preset(_preset("krea2"))

    panel.caption_length_combo.setCurrentText("Descriptive (Longest)")
    shown = panel.prompt_text.toPlainText()
    assert shown == panel.get_prompt()
    assert "never exceed 300 words" in shown
    assert "groups of similar people or objects" in shown
    assert "under 40 words" not in shown
    assert "each under about 30 words" not in shown
    assert CAPTION_LENGTHS["Descriptive (Longest)"] not in shown

    panel.caption_length_combo.setCurrentText("Short")
    panel._extra_checkboxes["excludeText"].setChecked(True)
    shown = panel.prompt_text.toPlainText()
    assert shown == panel.get_prompt()
    assert "Ignore text in the image." in shown
    assert "one or two short sentences, under 40 words:" in shown
    assert "not counting quoted text" not in shown
    assert "visible text" not in shown
    assert "3-5 sentences" not in shown

    panel._extra_checkboxes["excludeText"].setChecked(False)
    shown = panel.prompt_text.toPlainText()
    assert shown == panel.get_prompt()
    assert "under 40 words not counting quoted text" in shown
    assert "and any visible text" in shown
    assert "Ignore text in the image." not in shown
    panel.deleteLater()


def test_reverting_a_custom_edit_restores_the_generic_prompt(qapp):
    # Typing and then putting the preset prompt back is not an edit. Custom
    # edit must turn off, or the length and option sentences never append
    # and a later length change does nothing.
    panel = SettingsPanel()
    panel.prompt_text.setPlainText("Describe this image.")
    panel.caption_length_combo.setCurrentText("Short")
    panel._extra_checkboxes["includeLighting"].setChecked(True)

    panel._apply_preset(_preset("moss"))
    panel._toggle_custom_edit()
    preview = panel.prompt_text.toPlainText()
    panel.prompt_text.setPlainText("nope")
    panel.prompt_text.setPlainText(preview)
    panel._apply_preset(_preset("moss"))  # toggles the preset off

    assert panel.prompt_text.toPlainText() == "Describe this image."
    assert panel._custom_edit_mode is False
    assert panel.custom_edit_btn.text() == "✏ Edit"
    sent = panel.get_prompt()
    assert sent.startswith("Describe this image.")
    assert "Keep the description brief, around 1-2 sentences." in sent
    assert "Include information about lighting." in sent
    assert "Never mention watermarks" not in sent
    assert "precise lighting details" not in sent
    panel.deleteLater()


def test_trailing_space_keeps_custom_edit_and_strips_on_send(qapp):
    # The edit check is exact. Stripping before the compare would throw away
    # this prompt and append the generic length sentence to the old one.
    # get_prompt still strips, so the caption has no trailing space.
    panel = SettingsPanel()
    panel.prompt_text.setPlainText("Describe this image.")
    panel._extra_checkboxes["includeLighting"].setChecked(True)

    panel._apply_preset(_preset("moss"))
    panel._toggle_custom_edit()
    preview = panel.prompt_text.toPlainText()
    panel.prompt_text.setPlainText(preview + " ")
    assert panel.prompt_text.toPlainText() == preview + " "
    panel._apply_preset(_preset("moss"))  # toggles the preset off

    assert panel.prompt_text.toPlainText() == preview + " "
    assert panel._custom_edit_mode is True
    assert panel.custom_edit_btn.text() == "↻ Reset"
    assert panel.prompt_text.isReadOnly() is False
    assert panel.get_prompt() == preview
    assert "Describe this image." not in panel.get_prompt()
    assert "Include information about lighting." not in panel.get_prompt()
    panel.deleteLater()


def test_in_preset_edit_does_not_become_the_stashed_custom_prompt(qapp):
    # Switching presets drops the in-progress edit. It must not be saved as
    # the prompt from before any preset, and it must not flip the stashed
    # custom-edit flag on, or deselect would send the edit verbatim.
    panel = SettingsPanel()
    panel.prompt_text.setPlainText("Describe this image.")

    panel._apply_preset(_preset("moss"))
    panel._toggle_custom_edit()
    panel.prompt_text.setPlainText("temporary edit")
    panel._apply_preset(_preset("krea2"))

    assert panel._custom_edit_mode is False
    assert panel.prompt_text.isReadOnly() is True
    assert "temporary edit" not in panel.get_prompt()
    assert "each under about 30 words" in panel.get_prompt()

    panel._apply_preset(_preset("krea2"))  # toggles the preset off
    assert panel.prompt_text.toPlainText() == "Describe this image."
    assert panel._custom_edit_mode is False
    assert panel.custom_edit_btn.text() == "✏ Edit"
    sent = panel.get_prompt()
    assert sent.startswith("Describe this image.")
    assert "temporary edit" not in sent
    assert "Never mention watermarks" not in sent
    assert "Provide a moderately detailed description in 2-4 sentences." in sent
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
