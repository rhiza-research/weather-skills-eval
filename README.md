# weather-skills-eval

Measures what a change to weather-skills-chat (model, prompt, code, skills, settings) does to success, cost and behavior on real tasks.

`weather-skills-eval run <config>` starts a throwaway app container from a configuration file, plays every task of a Langfuse dataset through the app's chat API with a simulated user and a judge ([LangWatch Scenario](https://github.com/langwatch/scenario)), scores each play, and records the run as one Langfuse experiment. Two runs on the same dataset are compared side by side in Langfuse.

The harness talks to the app only over HTTP and Docker. Fetches use live data.

## Running

Everything runs in Docker.

```sh
cp .env.example .env        # fill in the harness settings and the values your config references
cp configs/example.yaml configs/<name>.yaml   # replace every TODO
HARNESS_COMMIT=$(git rev-parse HEAD) docker compose run --rm --build eval run configs/<name>.yaml
```

- The eval service mounts `./configs` read-only at `/app/configs`, so configuration files live in `configs/` and editing one does not rebuild the image. A bind mount needs the files on the Docker host, so with a remote Docker context the `configs/` folder must exist at the same path on that host.
- `HARNESS_COMMIT` is baked into the image and recorded in each experiment's metadata; without it the metadata says `unrecorded`.
- The eval service reaches the Docker daemon of the current Docker context through its socket. It pulls the configured image, starts the app container on the `weather-skills-eval` network, and removes it when the run ends, also when the run fails or the harness receives SIGTERM. When a run starts it also removes containers left by an earlier run (label `weather-skills-eval=app`), so two runs must not share one Docker daemon at the same time.

Tests and lint, in a container without network access:

```sh
docker compose run --rm --build test
```

### Exit codes

| Code | Meaning |
|-|-|
| 0 | The run finished and the experiment was recorded |
| 1 | The run failed for any other reason |
| 2 | The run did not start: a harness setting, the configuration file, the dataset or a dataset item is invalid, or the dataset is missing or empty |
| 3 | The app did not start (image pull, container start or `/health`), or Docker failed |
| 4 | The app failed while the harness read its tools and skill packs |
| 5 | Langfuse returned an error |
| 143 | The harness was stopped with SIGTERM; the app container was removed |

## Configuration file

`configs/example.yaml` is a complete example. Its keys:

- `name`: the experiment name; defaults to the file name without `.yaml`.
- `image`: the app image. It is pulled before every run, and the experiment records its digest.
- `model`: id of the catalog model the chats use.
- `dataset`: the Langfuse dataset holding the tasks.
- `env`: the app's whole environment; nothing else is set.

Rules:

- Any other top-level key stops the run.
- `env` uses the app's variable names (see the app wiki pages Deployment, Skill Packs and Tracing). `WEBUI_SECRET_KEY` and the three `BOOTSTRAP_ADMIN_*` variables are required: the harness signs in with `BOOTSTRAP_ADMIN_API_KEY`. `SKILL_SANDLOCK` must be `false` where the Docker host has no Landlock.
- `${NAME}` is replaced with the harness's environment variable `NAME`; a reference to an unset or empty variable stops the run.
- A variable, or a key inside a mapping value, whose name has the word key, secret, password or token in it (singular or plural; `tokens` alone, as in `max_tokens`, does not count) must be exactly one `${NAME}` reference. A literal value stops the run.
- Values are strings. Mappings and lists are written to the variable as JSON, after their references are replaced.
- The app listens on `PORT` from `env`, 8080 when unset.
- The app and the harness must use the same Langfuse project: the harness reads the app's traces with its own keys. The example passes the harness's Langfuse keys and address to the app.

## Harness settings

`.env.example` lists them. The simulated user and judge models are harness settings so that they stay fixed across the runs being compared. Each of their calls is a Langfuse trace with user id `simulated-user` or `judge`, session id = the chat id, in the environment set by `LANGFUSE_TRACING_ENVIRONMENT`, which must not be `production`.

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

- `input` is the opening message, sent as the first turn. It must not be empty. Item ids must be unique.
- `artifact_checks` run against the chat's files after every reply. A check passes when one file whose chat path matches `path` meets all of its conditions (`kind`, zarr `variables` and `dims`, text `contains`). `path` is an `fnmatch` pattern on the whole path, so `*` also matches `/`: `*.png` matches `maps/rain.png`.
- `judge_criteria` are judged by Scenario's judge once every artifact check passes. The goal is reached when all checks and all criteria pass, and the play ends there.
- `facts` is what the simulated user answers from. `hidden_steps` are revealed one at a time, only when the assistant is stuck. The simulated user may give up.
- `pauses` wait before the given turn (turn 1 is the opening message, so the first possible pause is before turn 2).
- `turn_limit` is the most user messages a play sends.
- `tags` are labels for grouping tasks in Langfuse. The harness does not read them; Langfuse shows them with the item's metadata.

## Outcomes

Each play ends with one outcome, given in the item output and in the `goal_reached` score comment:

| Outcome | Meaning |
|-|-|
| `goal_reached` | Every artifact check and every judge criterion passed |
| `turn_limit` | The turn limit passed without the goal |
| `user_gave_up` | The simulated user gave up |
| `chat_error` | The app returned an error for a turn, or a reply the harness cannot read |
| `error` | Something else stopped the play, such as a harness model reply that cannot be used |
| `not_reached` | The play ended without the goal for none of the reasons above |

For `turn_limit`, `user_gave_up` and `not_reached`, the reasoning ends with the judge's last check.

## Scores

Each item of the experiment gets these scores. When the goal is not reached, the "to goal" scores cover the whole play.

| Score | Meaning |
|-|-|
| `goal_reached` | 1 when the goal was reached, else 0 |
| `turns_to_goal` | User messages sent |
| `model_calls_to_goal` | The app's model calls, including failed ones |
| `clarifying_questions` | Assistant replies the simulated user classified as a clarifying question |
| `hints_used` | Hidden steps the simulated user revealed, at most the number of hidden steps |
| `answer_quality` | 1 to 5, rated by the judge model; not rated when no turn got a reply |
| `cost_to_goal_usd` | Sum of each generation's `cost_details` total, or its total cost when that is missing |
| `tokens_uncached_input`, `tokens_cache_read`, `tokens_cache_write`, `tokens_output` | Per generation, `input` is the whole prompt; the cache counts are moved out of it. App-recorded generations carry them in `usage_details` (`cache_read_input_tokens`, `cache_creation_input_tokens`); OpenRouter Broadcast generations carry them in metadata `attributes` (`gen_ai.usage.input_tokens.cached`, `gen_ai.usage.input_tokens.cache_write`) |
| `tool_calls`, `tool_failures` | Tool observations in the app's traces, and those at level ERROR |
| `cache_miss_calls` | Model calls that wrote at least 50,000 tokens of context (uncached input plus cache write) |
| `time_to_answer_seconds` | Time the app spent on turns that got a reply, from sending each message until its reply was done; pauses and failed turns are excluded |

Trace-based scores (model calls, cost, tokens, tools, cache misses) are read from the app's Langfuse traces of the chat: traces named `weather-skills-chat` with session id = chat id, one per turn the app accepted, once they stop changing. The item output's `trace_scores` says whether they are `complete`, `partial` (traces were still arriving at the timeout) or `missing` (with the reason, such as a Langfuse error or no traces arriving); when they are missing, the `goal_reached` comment says so too and the other scores are still recorded. Each item's output holds the chat id and the app's trace ids, and the item's trace carries the chat id as its session id.

The experiment's metadata holds the resolved configuration: image and its digest, the app's environment variable names, installed skill packs with their commits, the tool ids the chats used, the chat model, the simulated user and judge models, the harness commit, and the dataset with its version and item count.

A run plays each task once: Langfuse expects each dataset item to appear once per experiment (Langfuse issue #5855).
