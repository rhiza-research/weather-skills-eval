import pytest

from weather_skills_eval.app_client import highest_skill_versions, version_key, visible_text
from weather_skills_eval.config import ConfigError, parse_config
from weather_skills_eval.scoring import split_usage, trace_metrics
from weather_skills_eval.settings import SettingsError, load_settings
from weather_skills_eval.tasks import TaskError, parse_task

BASE_ENV = {
    "WEBUI_SECRET_KEY": "${S}",
    "BOOTSTRAP_ADMIN_EMAIL": "eval@example.invalid",
    "BOOTSTRAP_ADMIN_PASSWORD": "${P}",
    "BOOTSTRAP_ADMIN_API_KEY": "${K}",
}
ENVIRON = {"S": "secret", "P": "password", "K": "sk-key"}


def config(**overrides):
    raw = {"image": "img:tag", "model": "m", "dataset": "d", "env": dict(BASE_ENV)}
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
    raw = config(env={**BASE_ENV, "SKILL_SANDLOCK": False})
    with pytest.raises(ConfigError) as error:
        parse_config(raw, default_name="x", environ=ENVIRON)
    assert error.value.field == "env.SKILL_SANDLOCK"


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


def test_task_needs_a_goal():
    with pytest.raises(TaskError, match="expected_output"):
        parse_task(item(expected_output={}))


def test_split_usage_moves_cache_counts_out_of_input():
    assert split_usage(
        {
            "input": 127037,
            "output": 1711,
            "cache_read_input_tokens": 120583,
            "cache_creation_input_tokens": 6452,
        }
    ) == (2, 120583, 6452, 1711)
    # Broadcast: counts in metadata attributes, as strings.
    broadcast_metadata = {
        "attributes": {
            "gen_ai.usage.input_tokens.cached": "800",
            "gen_ai.usage.input_tokens.cache_write": "50",
        }
    }
    assert split_usage({"input": 900, "output": 3}, broadcast_metadata) == (50, 800, 50, 3)
    # Guessed key spellings are not read.
    assert split_usage({"input": 900, "output": 3, "input_cache_read": 800}) == (900, 0, 0, 3)
    assert split_usage({"input": 10, "output": 1}) == (10, 0, 0, 1)


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
