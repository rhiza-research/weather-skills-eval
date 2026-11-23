from types import SimpleNamespace

import pytest
from fakes import display_image, display_page, tool_call

from weather_skills_eval import runner
from weather_skills_eval.app_client import highest_skill_versions, version_key, visible_text
from weather_skills_eval.checks import ReplyFiles, reply_files
from weather_skills_eval.config import (
    DEFAULT_APP_PORT,
    ConfigError,
    pack_name,
    parse_config,
    secret_references,
)
from weather_skills_eval.judge import JudgeRuling, PanelRuling
from weather_skills_eval.runner import comparability_hash
from weather_skills_eval.scoring import split_usage, trace_metrics
from weather_skills_eval.settings import SettingsError, _number, _whole, load_settings
from weather_skills_eval.tasks import TaskError, parse_task, parse_tasks

BASE_ENV = {
    "WEBUI_SECRET_KEY": "${S}",
    "BOOTSTRAP_ADMIN_EMAIL": "eval@example.invalid",
    "BOOTSTRAP_ADMIN_PASSWORD": "${P}",
    "BOOTSTRAP_ADMIN_API_KEY": "${K}",
}
ENVIRON = {"S": "secret", "P": "password", "K": "sk-key"}


def config(**overrides):
    raw = {
        "image": "img:tag",
        "runner_image": "runner:tag",
        "model": "m",
        "dataset": "d",
        "env": dict(BASE_ENV),
    }
    raw.update(overrides)
    return raw


def test_config_resolves_references_and_defaults_name():
    result = parse_config(config(), default_name="prod", environ=ENVIRON)
    assert result.name == "prod"
    assert result.env["BOOTSTRAP_ADMIN_API_KEY"] == "sk-key"
    assert result.api_key == "sk-key"
    assert result.port == DEFAULT_APP_PORT


def test_config_pull_defaults_to_true_and_accepts_booleans():
    assert parse_config(config(), default_name="x", environ=ENVIRON).pull is True
    assert parse_config(config(pull=True), default_name="x", environ=ENVIRON).pull is True
    assert parse_config(config(pull=False), default_name="x", environ=ENVIRON).pull is False


@pytest.mark.parametrize("value", ["no", "false", 0, 1, None])
def test_config_pull_must_be_a_boolean(value):
    with pytest.raises(ConfigError) as error:
        parse_config(config(pull=value), default_name="x", environ=ENVIRON)
    assert error.value.field == "pull"


@pytest.mark.parametrize(
    "value, variant",
    [("reasoning_high", "reasoning_high"), ("  low  ", "low"), (None, ""), ("", ""), ("   ", "")],
    ids=repr,
)
def test_config_variant_is_stripped_and_empty_when_null_or_blank(value, variant):
    assert parse_config(config(variant=value), default_name="x", environ=ENVIRON).variant == variant


def test_config_variant_defaults_to_empty():
    assert parse_config(config(), default_name="x", environ=ENVIRON).variant == ""


def test_config_variant_must_be_a_string():
    with pytest.raises(ConfigError) as error:
        parse_config(config(variant=3), default_name="x", environ=ENVIRON)
    assert error.value.field == "variant"


APP_IMAGE = "weather-skills-chat:40be9ee"


def test_a_reference_resolves_from_the_environment_and_an_unset_or_empty_one_names_the_field():
    # image, runner_image and every env value resolve references the same way.
    environ = {**ENVIRON, "APP_IMAGE": APP_IMAGE, "APP_TAG": "40be9ee", "BLANK": ""}
    for image in ("${APP_IMAGE}", "weather-skills-chat:${APP_TAG}", APP_IMAGE):
        result = parse_config(config(image=image), default_name="x", environ=environ)
        assert result.image == APP_IMAGE
    for image, name in (
        ("${MISSING}", "MISSING"),
        ("weather-skills-chat:${MISSING}", "MISSING"),
        ("${BLANK}", "BLANK"),
        ("weather-skills-chat:${BLANK}", "BLANK"),
    ):
        with pytest.raises(ConfigError) as error:
            parse_config(config(image=image), default_name="x", environ=environ)
        assert error.value.field == "image"
        assert f"${{{name}}}" in str(error.value)


def test_secret_references_are_the_variables_secret_named_keys_reference():
    raw = config(
        env={
            **BASE_ENV,
            "OPENAI_API_KEYS": ["${OR}", "${OTHER}"],
            "LANGFUSE_BASE_URL": "${URL}",
            "TOOL_SERVERS": {"0": {"key": "${TOOL}", "url": "${TOOL_URL}"}},
            # A literal is left for parse_config to reject.
            "MISTRAL_API_KEY": "literal",
        },
        image="${APP_IMAGE}",
    )
    assert secret_references(raw) == {"S", "P", "K", "OR", "OTHER", "TOOL"}
    assert secret_references(None) == set()


def test_config_requires_the_admin_api_key():
    env = {k: v for k, v in BASE_ENV.items() if k != "BOOTSTRAP_ADMIN_API_KEY"}
    with pytest.raises(ConfigError) as error:
        parse_config(config(env=env), default_name="x", environ=ENVIRON)
    assert error.value.field == "env.BOOTSTRAP_ADMIN_API_KEY"


def test_config_rejects_secret_embedded_in_text():
    raw = config(env={**BASE_ENV, "WEBUI_SECRET_KEY": "prefix-${S}"})
    with pytest.raises(ConfigError) as error:
        parse_config(raw, default_name="x", environ=ENVIRON)
    assert error.value.field == "env.WEBUI_SECRET_KEY"


def test_config_serializes_structured_values_as_json():
    raw = config(
        env={**BASE_ENV, "OPENAI_API_CONFIGS": {"0": {"model_call_tracing": "openrouter"}}}
    )
    result = parse_config(raw, default_name="x", environ=ENVIRON)
    assert result.env["OPENAI_API_CONFIGS"] == '{"0": {"model_call_tracing": "openrouter"}}'


def test_config_rejects_non_string_scalars():
    raw = config(env={**BASE_ENV, "ENABLE_SIGNUP": False})
    with pytest.raises(ConfigError) as error:
        parse_config(raw, default_name="x", environ=ENVIRON)
    assert error.value.field == "env.ENABLE_SIGNUP"


def with_skill_packs(packs):
    raw = config(env={**BASE_ENV, "SKILL_PACKS": packs})
    return parse_config(raw, default_name="x", environ=ENVIRON)


def test_skill_packs_on_are_named_by_repository():
    packs = [
        {"git_url": "https://github.com/rhiza-research/chc-skills.git"},
        {"git_url": "https://example.invalid/forecasting-skills", "enabled_by_default": True},
        {"git_url": "https://example.invalid/off-skills.git", "enabled_by_default": False},
    ]
    assert with_skill_packs(packs).skill_packs_on == ["chc-skills", "forecasting-skills"]


@pytest.mark.parametrize(
    "git_url",
    [
        "https://github.com/rhiza-research/chc-skills.git",
        "https://github.com/rhiza-research/chc-skills",
        "https://github.com/rhiza-research/chc-skills/",
        "https://github.com/rhiza-research/chc-skills//",
        "  https://github.com/rhiza-research/chc-skills.git  ",
        "git@github.com:rhiza-research/chc-skills.git",
    ],
)
def test_pack_name_is_the_repository_name(git_url):
    assert pack_name(git_url) == "chc-skills"
    assert with_skill_packs([{"git_url": git_url}]).skill_packs_on == ["chc-skills"]
    assert parse_config(config(), default_name="x", environ=ENVIRON).skill_packs_on == []


@pytest.mark.parametrize(
    "packs",
    [
        "not json",
        {"git_url": "https://example.invalid/pack.git"},
        ["https://example.invalid/pack.git"],
        [{"ref": "main"}],
        [{"git_url": ""}],
        [{"git_url": "https://example.invalid/pack.git", "enabled_by_default": "no"}],
    ],
)
def test_config_rejects_invalid_skill_packs(packs):
    with pytest.raises(ConfigError) as error:
        with_skill_packs(packs)
    assert error.value.field == "env.SKILL_PACKS"


def settings_env(**overrides):
    env = {
        "LANGFUSE_PUBLIC_KEY": "pk",
        "LANGFUSE_SECRET_KEY": "sk",
        "OPENROUTER_API_KEY": "or",
        "LANGFUSE_TRACING_ENVIRONMENT": "eval",
        "EVAL_SIMULATED_USER_MODEL": "u",
        "EVAL_JUDGE_MODELS": "j",
        "EVAL_DOCKER_NETWORK": "n",
    }
    env.update(overrides)
    return env


def test_settings_refuse_the_production_environment():
    with pytest.raises(SettingsError):
        load_settings(settings_env(LANGFUSE_TRACING_ENVIRONMENT=" PRODUCTION "))


def test_settings_refuse_a_missing_variable():
    env = settings_env()
    del env["EVAL_JUDGE_MODELS"]
    with pytest.raises(SettingsError):
        load_settings(env)


def test_judge_models_are_a_comma_separated_panel():
    assert load_settings(settings_env()).judge_models == ("j",)
    panel = " openrouter/b/two , openrouter/a/one"
    settings = load_settings(settings_env(EVAL_JUDGE_MODELS=panel))
    assert settings.judge_models == ("openrouter/b/two", "openrouter/a/one")


@pytest.mark.parametrize("value", ["", "a,,b", "a, a"], ids=repr)
def test_invalid_judge_models_are_refused(value):
    with pytest.raises(SettingsError):
        load_settings(settings_env(EVAL_JUDGE_MODELS=value))


def test_the_old_judge_model_variable_is_not_read():
    env = settings_env(EVAL_JUDGE_MODEL="j")
    del env["EVAL_JUDGE_MODELS"]
    with pytest.raises(SettingsError):
        load_settings(env)


HASH_ITEMS = [
    {
        "id": "t1",
        "input": "rain next week",
        "expected_output": {"judge_criteria": ["gives totals"]},
        "metadata": {"facts": "Kenya"},
    },
    {
        "id": "t2",
        "input": "weekly rain map",
        "expected_output": {"deliverables": ["image"]},
        "metadata": {"facts": "Kenya", "hidden_steps": ["use weekly sums"]},
    },
]


def graded(**overrides) -> str:
    """The comparability hash of a run graded by two judges, changed by ``overrides``."""
    inputs = {
        "judge_models": ["openrouter/a/one", "openrouter/b/two"],
        "simulated_user_model": "openrouter/u/user",
        "items": HASH_ITEMS,
        "max_turns": 20,
        "model_max_tokens": 8192,
        "turn_timeout": 1800.0,
    }
    inputs.update(overrides)
    return comparability_hash(**inputs)


def test_the_comparability_hash_is_hex_of_its_length_and_stable():
    first = graded()
    assert len(first) == runner.COMPARABILITY_HASH_LENGTH
    assert int(first, 16) >= 0
    assert graded() == first
    # Neither the judges' nor the items' order changes it.
    assert graded(judge_models=["openrouter/b/two", "openrouter/a/one"]) == first
    assert graded(items=list(reversed(HASH_ITEMS))) == first
    # Items read from Langfuse are objects, not dicts.
    objects = [SimpleNamespace(**item) for item in HASH_ITEMS]
    assert graded(items=objects) == first
    # A whole-number timeout read as a float is the same setting.
    assert graded(turn_timeout=1800) == first


def first_item_with(**changes) -> list[dict]:
    """HASH_ITEMS with ``changes`` made to the first item."""
    return [{**HASH_ITEMS[0], **changes}, *HASH_ITEMS[1:]]


@pytest.mark.parametrize(
    "overrides",
    [
        {"judge_models": ["openrouter/a/one", "openrouter/c/three"]},
        {"judge_models": ["openrouter/a/one"]},
        {"simulated_user_model": "openrouter/u/other"},
        {"items": first_item_with(input="rain next month")},
        {"items": first_item_with(metadata={"facts": "Uganda"})},
        {"items": first_item_with(expected_output={"judge_criteria": ["gives daily totals"]})},
        {"items": HASH_ITEMS[:1]},
        {"max_turns": 21},
        {"model_max_tokens": 4096},
        {"turn_timeout": 900.0},
    ],
    ids=[
        "one-judge-changed",
        "one-judge-dropped",
        "user-model",
        "item-input",
        "item-metadata",
        "item-expected-output",
        "item-removed",
        "max-turns",
        "model-max-tokens",
        "turn-timeout",
    ],
)
def test_the_comparability_hash_changes_with_what_grades_a_run(overrides):
    assert graded(**overrides) != graded()


def test_the_comparability_hash_changes_with_the_grading_version_and_prompts(monkeypatch):
    first = graded()
    monkeypatch.setattr(runner, "GRADING_VERSION", runner.GRADING_VERSION + 1)
    assert graded() != first
    monkeypatch.undo()
    for name in (
        "SCENARIO_DESCRIPTION",
        "STARTED_LINE",
        "PROVENANCE_CRITERION",
        "EVIDENCE_HEADER",
        "QUALITY_PROMPT",
        "SIMULATED_USER_PROMPT",
        "CONVERSATION_HEADER",
        "FILES_HEADER",
        "REPLY_REQUEST",
        "RETRY_REQUEST",
    ):
        monkeypatch.setattr(runner, name, getattr(runner, name) + " ")
        assert graded() != first, name
        monkeypatch.undo()
    # Scenario's judge system prompt is in the installed langwatch-scenario.
    installed = runner.importlib.metadata.version
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: installed(name) + ".x")
    assert graded() != first
    monkeypatch.undo()
    assert graded() == first


def ruling(model, passed, reasoning, model_called=True) -> JudgeRuling:
    return JudgeRuling(f"openrouter/{model}", passed, reasoning, model_called)


def test_the_judges_that_ruled_decide_and_an_errored_judge_is_left_out():
    panel = PanelRuling(
        [ruling("a", True, "ok"), ruling("b", None, "Timeout: x"), ruling("c", True, "ok")]
    )
    assert panel.error is None
    assert panel.passed is True
    assert panel.agree is True
    assert [j["state"] for j in panel.output()] == ["passed", "error", "passed"]
    assert panel.output()[1]["passed"] is None
    # One pass and one fail among the judges that ruled is a tie, which fails.
    tied = PanelRuling([ruling("a", True, "ok"), ruling("b", None, "x"), ruling("c", False, "no")])
    assert tied.passed is False
    assert tied.agree is False


def test_a_panel_where_no_judge_ruled_is_an_error():
    panel = PanelRuling([ruling("a", None, "Timeout: x"), ruling("b", None, "Timeout: y")])
    assert panel.error is not None
    assert panel.passed is False
    assert panel.agree is None


def test_judges_agree_needs_two_judges_that_ruled_and_a_model_call():
    assert PanelRuling([ruling("a", True, "ok")]).agree is None
    assert PanelRuling([ruling("a", True, "ok"), ruling("b", None, "x")]).agree is None
    no_call = [ruling(m, False, "No result presented", model_called=False) for m in "ab"]
    assert PanelRuling(no_call).agree is None


def test_the_verdict_is_shared_only_for_the_same_verdict_and_reasoning():
    same = PanelRuling([ruling("a", False, "too wet"), ruling("b", False, "too wet")])
    assert same.verdict == "too wet"
    split = PanelRuling([ruling("a", True, "too wet"), ruling("b", False, "too wet")])
    assert split.verdict != "too wet"


def item(**overrides):
    data = {
        "id": "i1",
        "input": "rain next week",
        "expected_output": {"judge_criteria": ["gives totals"]},
        "metadata": {"facts": "Kenya", "turn_limit": 3},
    }
    data.update(overrides)
    return data


def test_task_parses_a_valid_item():
    task = parse_task(item())
    assert task.opening_message == "rain next week"
    assert task.pause_before(2) == 0.0


def test_task_rejects_unknown_metadata_key():
    with pytest.raises(TaskError, match="difficulty"):
        parse_task(item(metadata={"facts": "Kenya", "turn_limit": 3, "difficulty": "x"}))


def test_task_rejects_pause_past_turn_limit():
    with pytest.raises(TaskError):
        parse_task(
            item(
                metadata={
                    "facts": "K",
                    "turn_limit": 2,
                    "pauses": [{"before_turn": 3, "seconds": 1}],
                }
            )
        )


def test_task_turn_limit_is_optional():
    metadata = {"facts": "K", "pauses": [{"before_turn": 9, "seconds": 1}]}
    task = parse_task(item(metadata=metadata))
    assert task.metadata.turn_limit is None
    assert task.max_turns(20) == 20
    assert parse_task(item()).max_turns(20) == 3
    # Without turn_limit, pauses are checked against EVAL_MAX_TURNS when the run starts.
    assert parse_tasks([item(metadata=metadata)], max_turns=9)
    with pytest.raises(TaskError):
        parse_tasks([item(metadata=metadata)], max_turns=8)
    # With it, against turn_limit only.
    limited = {**metadata, "turn_limit": 10}
    assert parse_tasks([item(metadata=limited)], max_turns=2)


@pytest.mark.parametrize("turn_limit", [True, "3", 2.0, 3.0], ids=repr)
def test_task_turn_limit_must_be_an_integer(turn_limit):
    with pytest.raises(TaskError):
        parse_task(item(metadata={"facts": "K", "turn_limit": turn_limit}))


@pytest.mark.parametrize("before_turn", [True, "3", 3.0], ids=repr)
def test_task_pause_turn_must_be_an_integer(before_turn):
    metadata = {"facts": "K", "pauses": [{"before_turn": before_turn, "seconds": 1}]}
    with pytest.raises(TaskError):
        parse_task(item(metadata=metadata))


def test_task_pause_seconds_may_be_a_whole_or_fractional_number():
    for seconds in (600, 0.5):
        metadata = {"facts": "K", "pauses": [{"before_turn": 2, "seconds": seconds}]}
        assert parse_task(item(metadata=metadata)).pause_before(2) == seconds


@pytest.mark.parametrize("seconds", [float("inf"), "inf", float("nan"), True], ids=repr)
def test_task_pause_seconds_must_be_a_finite_number(seconds):
    metadata = {"facts": "K", "pauses": [{"before_turn": 2, "seconds": seconds}]}
    with pytest.raises(TaskError):
        parse_task(item(metadata=metadata))


@pytest.mark.parametrize("criteria", [[""], ["   "], ["gives totals", "\t\n"]], ids=repr)
def test_task_judge_criteria_must_hold_text(criteria):
    with pytest.raises(TaskError):
        parse_task(item(expected_output={"judge_criteria": criteria}))


def test_an_optional_setting_has_its_default_when_unset_and_rejects_an_invalid_value():
    # Every optional number setting is read with _number or _whole.
    for unset in ({}, {"X": ""}, {"X": "  "}):
        assert _number(unset, "X", 7.5) == 7.5
        assert _whole(unset, "X", 7, minimum=0) == 7
    assert _number({"X": "0.5"}, "X", 7.5) == 0.5
    assert _whole({"X": "3"}, "X", 7, minimum=0) == 3
    for value in ("many", "nan", "inf", "0", "-1"):
        with pytest.raises(SettingsError, match="X"):
            _number({"X": value}, "X", 7.5)
    for value in ("many", "1.5", "0"):
        with pytest.raises(SettingsError, match="X"):
            _whole({"X": value}, "X", 7, minimum=1)


def test_batch_is_read_stripped_and_empty_when_unset():
    assert load_settings(settings_env()).batch == ""
    assert load_settings(settings_env(EVAL_BATCH="  ")).batch == ""
    assert load_settings(settings_env(EVAL_BATCH=" 2026-10-22T20:07:57Z ")).batch == (
        "2026-10-22T20:07:57Z"
    )


def test_task_deliverables_are_image_or_zarr():
    task = parse_task(item(expected_output={"deliverables": ["image", "zarr"]}))
    assert task.expected.deliverables == ["image", "zarr"]
    assert parse_task(item()).expected.deliverables == []
    with pytest.raises(TaskError):
        parse_task(item(expected_output={"deliverables": ["png"]}))


def test_task_needs_a_goal():
    with pytest.raises(TaskError):
        parse_task(item(expected_output={}))


def test_task_rejects_artifact_checks():
    with pytest.raises(TaskError):
        parse_task(item(expected_output={"artifact_checks": [{"path": "*.png"}]}))


def presented_paths(content: str) -> list[str]:
    """The paths a reply presents whatever the chat's file listing holds."""
    return reply_files(content, 1).presented(())


def test_presented_paths_come_from_successful_tool_calls_and_relative_links():
    content = "".join(
        [
            "Here is the map ![map](./maps/rain.png) and [notes](notes.txt 'n').\n",
            display_image("/maps/rain.png"),
            tool_call("display_image", {"path": "missing.png"}, "Image not found: `missing.png`"),
            tool_call(
                "send_email",
                {"to": ["a@example.invalid"], "attachments": "out.zarr"},
                "Email sent to a@example.invalid with reply-to `a@example.invalid`.",
            ),
            tool_call(
                "send_email",
                {"to": ["a@example.invalid"], "attachments": ["x.png"]},
                "Email send failed: refused",
            ),
            '<details type="reasoning" done="true">\n[draft](draft.png)\n</details>\n',
            "[site](https://example.invalid/a.png) [top](#top) [cached](/cache/images/1.png)",
            " ![inline](data:image/png;base64,AAAA) [folder](plots/) ![m](my%20map.png)",
        ]
    )
    assert presented_paths(content) == [
        "maps/rain.png",
        "notes.txt",
        "out.zarr",
        "plots",
        "my map.png",
    ]
    assert presented_paths("No files here.") == []


def test_link_query_and_fragment_are_dropped():
    assert presented_paths("[d](out.zarr?x=1)") == ["out.zarr"]
    assert presented_paths("[m](rain.png#top)") == ["rain.png"]


@pytest.mark.parametrize(
    "attachments, paths",
    [
        (123, []),
        (True, []),
        ({"path": "x.png"}, []),
        (None, []),
        ([1, None, "x.png", {"p": "y.png"}], ["x.png"]),
    ],
)
def test_emailed_attachments_that_are_not_paths_are_ignored(attachments, paths):
    content = tool_call(
        "send_email",
        {"to": ["a@example.invalid"], "attachments": attachments},
        "Email sent to a@example.invalid with reply-to `a@example.invalid`.",
    )
    assert presented_paths(content) == paths


def test_a_successful_html_display_presents_its_path_and_a_failed_one_nothing():
    content = "".join(
        [
            display_page("plots/sst.html"),
            tool_call("display_image", {"path": "nope.html"}, "File not found: `nope.html`"),
            tool_call(
                "display_image",
                {"path": "data.csv"},
                "Unsupported type. Use a .png, .jpg, .jpeg, .gif, or .webp image, or an .html "
                "page (got `.csv`).",
            ),
        ]
    )
    assert presented_paths(content) == ["plots/sst.html"]


def test_backtick_text_outside_tool_blocks_is_a_candidate_in_order():
    content = (
        "Saved `kenya.zarr` from [the map](maps/rain.png) with variable `precip`.\n"
        + tool_call("list_artifacts", {}, "- `listed.zarr` (zarr)")
        + '<details type="reasoning" done="true">\n`draft.zarr`\n</details>\n'
        + "Also `./out/rain.zarr/`, ``double.zarr``, ```fence.zarr``` and `maps/rain.png`."
    )
    files = reply_files(content, 3)
    assert files == ReplyFiles(
        3,
        ["kenya.zarr", "maps/rain.png", "precip", "out/rain.zarr"],
        frozenset({"kenya.zarr", "precip", "out/rain.zarr"}),
    )


def test_backticks_inside_a_fenced_code_block_are_not_candidates():
    content = (
        "Saved `kenya.zarr`:\n"
        "```python\n"
        "ds = xr.open_zarr('x')  # see `inside.zarr`\n"
        "print(`also_inside.zarr`)\n"
        "```\n"
        "~~~\n`tilde.zarr`\n~~~\n"
        "Then `after.zarr`.\n"
        "```\n`unclosed.zarr`\n"
    )
    assert reply_files(content, 1).paths == ["kenya.zarr", "after.zarr"]


@pytest.mark.parametrize(
    "listed, presented",
    [
        ({"kenya.zarr"}, ["kenya.zarr", "maps/rain.png"]),
        ({"out/kenya.zarr", "missing.zarr"}, ["maps/rain.png"]),
        (set(), ["maps/rain.png"]),
    ],
    ids=["listed", "basename-only", "not-listed"],
)
def test_a_candidate_is_presented_only_when_listed_with_exactly_that_path(listed, presented):
    files = reply_files("Here: `kenya.zarr` and `precip`.\n" + display_image("maps/rain.png"), 1)
    assert files.presented(listed) == presented


def test_split_usage_moves_cache_counts_out_of_input():
    assert split_usage(
        {
            "input": 127037,
            "output": 1711,
            "cache_read_input_tokens": 120583,
            "cache_creation_input_tokens": 6452,
        }
    ) == (2, 120583, 6452, 1711)
    # Broadcast: counts in flat dotted metadata keys, as strings.
    broadcast_metadata = {
        "attributes.gen_ai.usage.input_tokens.cached": "800",
        "attributes.gen_ai.usage.input_tokens.cache_write": "50",
    }
    assert split_usage({"input": 900, "output": 3}, broadcast_metadata) == (50, 800, 50, 3)
    # App-recorded counts take precedence over metadata.
    assert split_usage(
        {"input": 900, "output": 3, "cache_read_input_tokens": 100}, broadcast_metadata
    ) == (800, 100, 0, 3)
    # Guessed key spellings are not read.
    assert split_usage({"input": 900, "output": 3, "input_cache_read": 800}) == (900, 0, 0, 3)
    assert split_usage({"input": 10, "output": 1}) == (10, 0, 0, 1)
    assert split_usage({"input": 10, "output": 1}, {}) == (10, 0, 0, 1)
    assert split_usage({"input": 10, "output": 1}, None) == (10, 0, 0, 1)


def test_trace_metrics_reads_broadcast_cache_counts_of_a_real_generation():
    # Langfuse generation c7346b686bb6f406: 100853 input = 4 uncached + 329 write + 100520 read.
    metrics = trace_metrics(
        [
            {
                "trace_id": "t",
                "type": "GENERATION",
                "usage_details": {"input": 100853, "output": 240, "total": 101093},
                "metadata": {
                    "attributes.gen_ai.usage.input_tokens.cached": 100520,
                    "attributes.gen_ai.usage.input_tokens.cache_write": 329,
                },
            }
        ]
    )
    assert metrics.tokens_uncached_input == 4
    assert metrics.tokens_cache_read == 100520
    assert metrics.tokens_cache_write == 329
    assert metrics.tokens_output == 240
    assert metrics.cache_miss_calls == 0


def test_trace_metrics_falls_back_to_total_cost():
    metrics = trace_metrics(
        [
            {
                "trace_id": "t",
                "type": "GENERATION",
                "usage_details": {"input": 5, "output": 1},
                "cost_details": {"total": None, "input": 0.1},
                "total_cost": 0.3,
            }
        ]
    )
    assert metrics.cost_usd == pytest.approx(0.3)
    assert metrics.calls_without_cost == 0


def test_trace_metrics_counts_failed_model_calls_and_missing_cost():
    metrics = trace_metrics(
        [
            {
                "trace_id": "t",
                "type": "GENERATION",
                "usage_details": {"input": 5, "output": 1},
                "cost_details": {},
                "total_cost": None,
            },
            {"trace_id": "t", "type": "SPAN", "name": "llm", "level": "ERROR"},
            {"trace_id": "t", "type": "TOOL", "level": "DEFAULT"},
        ]
    )
    assert metrics.model_calls == 2
    assert metrics.calls_without_cost == 1
    assert metrics.tool_calls == 1 and metrics.tool_failures == 0


def test_tool_with_ok_false_output_is_a_failure():
    metrics = trace_metrics(
        [
            {"trace_id": "t", "type": "TOOL", "level": "DEFAULT", "ok": False},
            {"trace_id": "t", "type": "TOOL", "level": "DEFAULT", "ok": True},
            # Output not a JSON object: not counted unless at level ERROR.
            {"trace_id": "t", "type": "TOOL", "level": "DEFAULT", "ok": None},
            {"trace_id": "t", "type": "TOOL", "level": "ERROR", "ok": None},
        ]
    )
    assert metrics.tool_calls == 4
    assert metrics.tool_failures == 2


def test_highest_skill_versions_compares_numerically():
    tools = [
        {
            "id": "a",
            "name": "s",
            "meta": {"manifest": {"kind": "skill", "skill_name": "S", "version": "1.9"}},
        },
        {
            "id": "b",
            "name": "s",
            "meta": {"manifest": {"kind": "skill", "skill_name": "s", "version": "1.10"}},
        },
        {"id": "c", "name": "other", "meta": {}},
    ]
    assert highest_skill_versions(tools) == ["b", "c"]


def test_skill_versions_follow_semver_with_prereleases_below_releases():
    assert version_key("v2.0.0") == version_key("2.0.0")
    assert version_key("2.0.0-rc.1") < version_key("2.0.0")
    assert version_key("2.0.0-alpha") < version_key("2.0.0-beta")
    assert version_key("2.0.0-rc.2") < version_key("2.0.0-rc.10")
    assert version_key("1.10.0") > version_key("1.9.5")

    def skill(tool_id, version, name="s"):
        return {
            "id": tool_id,
            "name": name,
            "meta": {"manifest": {"kind": "skill", "skill_name": name, "version": version}},
        }

    assert highest_skill_versions([skill("a", "2.0.0"), skill("b", "v2.1.0-rc.1")]) == ["b"]
    assert highest_skill_versions([skill("a", "2.1.0"), skill("b", "v2.1.0-rc.1")]) == ["a"]
    # Skill tools without a name cannot be matched up, so each is kept.
    assert highest_skill_versions([skill("a", "1", ""), skill("b", "2", "")]) == ["a", "b"]


def test_a_duration_leaves_out_each_part_that_is_0():
    assert runner.duration(252) == "4m12s"
    assert runner.duration(1080) == "18m"
    assert runner.duration(3605) == "1h5s"
    assert runner.duration(7322.4) == "2h2m2s"
    assert runner.duration(59.6) == "1m"
    assert runner.duration(0.4) == "0s"


def test_visible_text_drops_tool_blocks():
    content = 'Fetching.\n<details type="tool_calls" done="true">\n<summary>x</summary>\n</details>\nDone.'
    assert visible_text(content) == "Fetching.\n\nDone."
