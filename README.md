# weather-skills-eval

Measures what a change to weather-skills-chat (model, prompt, code, skills, settings) does to success, cost and behavior on real tasks.

`weather-skills-eval run <config>` starts a throwaway app container from a configuration file, plays every task of a Langfuse dataset through the app's chat API with a simulated user and a judge ([LangWatch Scenario](https://github.com/langwatch/scenario)), scores each play, and records the run as one Langfuse experiment. Two runs on the same dataset are compared side by side in Langfuse.

The harness talks to the app only over HTTP and Docker. Fetches use live data.

## Running

Everything runs in Docker.

```sh
cp .env.example .env        # fill in the harness settings and the values your config references
docker compose run --rm --build eval run configs/<name>.yaml
```

The configuration file must be inside the repository: the eval image is built from it. The eval service reaches the Docker daemon of the current Docker context through its socket, starts the app container on the `weather-skills-eval` network, and removes it when the run ends, also when the run fails.

Tests and lint:

```sh
docker compose run --rm --build test
```

## Configuration file

```yaml
name: production-agent          # experiment name; defaults to the file name without .yaml
image: ghcr.io/rhiza-research/weather-skills-chat:TODO
model: TODO                     # id of the catalog model the chats use
dataset: TODO                   # Langfuse dataset holding the tasks
env:                            # the app's whole environment; nothing else is set
  WEBUI_SECRET_KEY: ${APP_WEBUI_SECRET_KEY}
  BOOTSTRAP_ADMIN_EMAIL: TODO
  BOOTSTRAP_ADMIN_PASSWORD: ${APP_ADMIN_PASSWORD}
  BOOTSTRAP_ADMIN_API_KEY: ${APP_ADMIN_API_KEY}
  OPENAI_API_BASE_URLS: TODO
  OPENAI_API_KEYS: ${APP_OPENROUTER_API_KEY}
  SKILL_SANDLOCK: "false"
  CATALOG_MODELS: [...]         # mappings and lists are passed as JSON
  SKILL_PACKS: [...]
  LANGFUSE_PUBLIC_KEY: ${APP_LANGFUSE_PUBLIC_KEY}
  LANGFUSE_SECRET_KEY: ${APP_LANGFUSE_SECRET_KEY}
```

- The keys are `name`, `image`, `model`, `dataset` and `env`. Any other key stops the run.
- `env` uses the app's variable names (see the app wiki pages Deployment, Skill Packs and Tracing). `WEBUI_SECRET_KEY` and the three `BOOTSTRAP_ADMIN_*` variables above are required: the harness signs in with `BOOTSTRAP_ADMIN_API_KEY`. `SKILL_SANDLOCK` must be `false` where the Docker host has no Landlock.
- `${NAME}` is replaced with the harness's environment variable `NAME`. A variable whose name contains KEY, SECRET, PASSWORD or TOKEN must be exactly one `${NAME}` reference; a literal value stops the run.
- Values are strings. Mappings and lists are written to the variable as JSON.
- The app listens on `PORT` from `env`, 8080 when unset.

## Harness settings

`.env.example` lists them. The simulated user and judge models are harness settings so that they stay fixed across the runs being compared. Their calls are traced in Langfuse with user id `simulated-user` and `judge`, session id = the chat id, in the environment set by `LANGFUSE_TRACING_ENVIRONMENT`, which must not be `production`.

## Task format (one Langfuse dataset item per task)

```json
{
  "input": "I need next week's rain totals for western Kenya as a map",
  "expected_output": {
    "artifact_checks": [
      {"path": "*.png", "kind": "image"},
      {"path": "*.zarr", "kind": "zarr", "variables": ["precip"], "dims": ["time"]},
      {"path": "notes.txt", "contains": "mm"}
    ],
    "judge_criteria": ["The assistant presents weekly rain totals for western Kenya"]
  },
  "metadata": {
    "facts": "Region: western Kenya. Period: the coming week. Units: mm.",
    "hidden_steps": ["Ask it to use the weekly aggregate instead of daily values"],
    "pauses": [{"before_turn": 3, "seconds": 600}],
    "turn_limit": 6,
    "tags": ["map", "kenya"]
  }
}
```

- `input` is the opening message, sent as the first turn.
- `artifact_checks` run against the chat's files after every reply. A check passes when one file whose chat path matches the `path` glob meets all of its conditions (`kind`, zarr `variables` and `dims`, text `contains`).
- `judge_criteria` are judged by Scenario's judge once every artifact check passes. The goal is reached when all checks and all criteria pass, and the play ends there.
- `facts` is what the simulated user answers from. `hidden_steps` are revealed one at a time, only when the assistant is stuck. The simulated user may give up.
- `pauses` wait before the given turn (turn 1 is the opening message, so the first possible pause is before turn 2).
- `turn_limit` is the most user messages a play sends.

## Scores

Each item of the experiment gets these scores. When the goal is not reached, the "to goal" scores cover the whole play, and the score comment says how the play ended (`turn_limit`, `user_gave_up`, `chat_error`).

| Score | Meaning |
|-|-|
| `goal_reached` | 1 when the goal was reached, else 0 |
| `turns_to_goal` | User messages sent |
| `model_calls_to_goal` | The app's model calls, including failed ones |
| `clarifying_questions` | Assistant replies the simulated user classified as a clarifying question |
| `hints_used` | Hidden steps the simulated user revealed |
| `answer_quality` | 1 to 5, rated by the judge model |
| `cost_to_goal_usd` | Sum of the generations' `cost_details` total |
| `tokens_uncached_input`, `tokens_cache_read`, `tokens_cache_write`, `tokens_output` | From the generations' `usage_details`; cache counts are moved out of `input` |
| `tool_calls`, `tool_failures` | Tool observations in the app's traces, and those at level ERROR |
| `cache_miss_calls` | Model calls that wrote at least 50,000 tokens of context (uncached input plus cache write) |
| `time_to_answer_seconds` | Time the app spent answering, from sending each message until its reply was done; pauses are excluded |

Trace-based scores are read from the app's Langfuse traces of the chat (one trace per turn, session id = chat id) once they stop changing. Each item's output holds the chat id and the app's trace ids.

The experiment's metadata holds the resolved configuration: image digest, the app's environment variable names, installed skill packs with their commits, the chat model, and the simulated user and judge models.

A run plays each task once: Langfuse expects each dataset item to appear once per experiment (Langfuse issue #5855).
