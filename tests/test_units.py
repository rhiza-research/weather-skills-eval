import pytest
from fakes import display_image, tool_call

from weather_skills_eval.app_client import highest_skill_versions, version_key, visible_text
from weather_skills_eval.checks import presented_paths
from weather_skills_eval.config import ConfigError, pack_name, parse_config
from weather_skills_eval.scoring import split_usage, trace_metrics
from weather_skills_eval.settings import SettingsError, load_settings
from weather_skills_eval.simulated_user import SYSTEM_PROMPT
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
    assert result.port == "8080"


def test_config_pull_defaults_to_true_and_accepts_booleans():
    assert parse_config(config(), default_name="x", environ=ENVIRON).pull is True
    assert parse_config(config(pull=True), default_name="x", environ=ENVIRON).pull is True
    assert parse_config(config(pull=False), default_name="x", environ=ENVIRON).pull is False


@pytest.mark.parametrize("value", ["no", "false", 0, 1, None])
def test_config_pull_must_be_a_boolean(value):
    with pytest.raises(ConfigError) as error:
        parse_config(config(pull=value), default_name="x", environ=ENVIRON)
    assert error.value.field == "pull"


def test_config_reference_to_unset_variable_names_the_field():
    raw = config(env={**BASE_ENV, "LANGFUSE_SECRET_KEY": "${MISSING}"})
    with pytest.raises(ConfigError) as error:
        parse_config(raw, default_name="x", environ=ENVIRON)
    assert error.value.field == "env.LANGFUSE_SECRET_KEY"


APP_IMAGE = "weather-skills-chat:40be9ee"


@pytest.mark.parametrize(
    "image, environ",
    [
        ("${APP_IMAGE}", {"APP_IMAGE": APP_IMAGE}),
        ("weather-skills-chat:${APP_TAG}", {"APP_TAG": "40be9ee"}),
        (APP_IMAGE, {}),
    ],
    ids=["whole-reference", "partial-reference", "literal"],
)
def test_config_image_resolves_references(image, environ):
    result = parse_config(config(image=image), default_name="x", environ={**ENVIRON, **environ})
    assert result.image == APP_IMAGE


@pytest.mark.parametrize(
    "image, name, environ",
    [
        ("${APP_IMAGE}", "APP_IMAGE", {}),
        ("${APP_IMAGE}", "APP_IMAGE", {"APP_IMAGE": ""}),
        ("weather-skills-chat:${APP_TAG}", "APP_TAG", {}),
        ("weather-skills-chat:${APP_TAG}", "APP_TAG", {"APP_TAG": ""}),
    ],
    ids=["whole-unset", "whole-empty", "partial-unset", "partial-empty"],
)
def test_config_image_reference_to_unset_or_empty_variable_names_the_image(image, name, environ):
    with pytest.raises(ConfigError) as error:
        parse_config(config(image=image), default_name="x", environ={**ENVIRON, **environ})
    assert error.value.field == "image"
    assert f"${{{name}}}" in str(error.value)


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
        "EVAL_JUDGE_MODEL": "j",
        "EVAL_DOCKER_NETWORK": "n",
    }
    env.update(overrides)
    return env


def test_settings_refuse_the_production_environment():
    with pytest.raises(SettingsError, match="LANGFUSE_TRACING_ENVIRONMENT"):
        load_settings(settings_env(LANGFUSE_TRACING_ENVIRONMENT="production"))


def test_settings_name_the_missing_variable():
    env = settings_env()
    del env["EVAL_JUDGE_MODEL"]
    with pytest.raises(SettingsError, match="EVAL_JUDGE_MODEL"):
        load_settings(env)


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
    with pytest.raises(TaskError, match="metadata.difficulty"):
        parse_task(item(metadata={"facts": "Kenya", "turn_limit": 3, "difficulty": "x"}))


def test_task_rejects_pause_past_turn_limit():
    with pytest.raises(TaskError, match="turn limit"):
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
    with pytest.raises(TaskError, match="i1: metadata.pauses: pause before turn 9 is past"):
        parse_tasks([item(metadata=metadata)], max_turns=8)
    # With it, against turn_limit only.
    limited = {**metadata, "turn_limit": 10}
    assert parse_tasks([item(metadata=limited)], max_turns=2)


def test_simulated_user_prompt_holds_the_done_give_up_and_correction_rules():
    for rule in (
        "You are done when the assistant has presented everything you asked for, as you asked.",
        "Give up only when the assistant says it cannot do what you asked",
        "Compare what you received, the answer's text and any files, with what you asked for",
        "say so plainly and ask for the fix",
        '"done": true when the assistant has presented everything you asked for',
    ):
        assert rule in SYSTEM_PROMPT.replace("\n", " ")
    assert "rovenance" not in SYSTEM_PROMPT


def test_simulated_user_prompt_reveals_the_next_step_when_stuck_or_past_it_and_skips_done_steps():
    prompt = SYSTEM_PROMPT.replace("\n", " ")
    for rule in (
        "The next hidden step is the first one below that you have not revealed and that is "
        "not done.",
        "A step the conversation or the files you were shown already show as done counts as "
        "used: skip it and consider the next one.",
        "The assistant is stuck: it repeats itself, says it cannot go on, or asks you how to "
        "proceed.",
        "The assistant has presented a result, or moved on past where that step belongs, "
        "without doing that step.",
        "a progress reply that has not reached that step yet is not a reason",
        "Reveal at most one per reply",
    ):
        assert rule in prompt
    assert "only mention one when the assistant is stuck" not in prompt


def test_max_turns_defaults_to_20():
    assert load_settings(settings_env()).max_turns == 20
    assert load_settings(settings_env(EVAL_MAX_TURNS="50")).max_turns == 50
    with pytest.raises(SettingsError, match="EVAL_MAX_TURNS"):
        load_settings(settings_env(EVAL_MAX_TURNS="0"))


def test_task_deliverables_are_image_or_zarr():
    task = parse_task(item(expected_output={"deliverables": ["image", "zarr"]}))
    assert task.expected.deliverables == ["image", "zarr"]
    assert parse_task(item()).expected.deliverables == []
    with pytest.raises(TaskError, match="expected_output.deliverables"):
        parse_task(item(expected_output={"deliverables": ["png"]}))


def test_task_needs_a_goal():
    with pytest.raises(TaskError, match="expected_output") as error:
        parse_task(item(expected_output={}))
    assert "needs at least one deliverable or judge criterion" in str(error.value)


def test_task_rejects_artifact_checks():
    with pytest.raises(TaskError, match="expected_output.artifact_checks"):
        parse_task(item(expected_output={"artifact_checks": [{"path": "*.png"}]}))


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


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "many"])
def test_model_max_tokens_must_be_a_whole_number_of_at_least_1(value):
    with pytest.raises(SettingsError, match="EVAL_MODEL_MAX_TOKENS"):
        load_settings(settings_env(EVAL_MODEL_MAX_TOKENS=value))


def test_model_max_tokens_defaults_to_8192():
    assert load_settings(settings_env()).model_max_tokens == 8192
    assert load_settings(settings_env(EVAL_MODEL_MAX_TOKENS="1024")).model_max_tokens == 1024


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


def test_visible_text_drops_tool_blocks():
    content = 'Fetching.\n<details type="tool_calls" done="true">\n<summary>x</summary>\n</details>\nDone.'
    assert visible_text(content) == "Fetching.\n\nDone."
