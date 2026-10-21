# weather-skills-eval

Measures what a change to weather-skills-chat (model, prompt, code, skills, settings) does to success, cost and behavior on real tasks.

`weather-skills-eval run <config>` starts a throwaway app container from a configuration file, plays every task of a Langfuse dataset through the app's chat API with a simulated user and a judge ([LangWatch Scenario](https://github.com/langwatch/scenario)), scores each play, and records the run as one Langfuse experiment. Two runs on the same dataset are compared side by side in Langfuse.

The harness talks to the app only over HTTP and Docker. Fetches use live data.

## Running

Everything runs in Docker.

```sh
cp .env.example .env        # fill in the harness settings and the values your config references
cp configs/example.yaml configs/<name>.yaml   # replace every TODO
HARNESS_COMMIT=$(git rev-parse HEAD) docker compose build eval
docker compose run --rm -T eval run - < configs/<name>.yaml
```

- `run -` reads the configuration from stdin, so it works with a remote Docker context, where a bind mount from this machine would not reach the daemon. A configuration read from stdin must set `name`. `run <path>` reads a file inside the container instead.
- `HARNESS_COMMIT` is baked into the image and recorded in each experiment's metadata; without it the metadata says `unrecorded`.
- The eval service reaches the Docker daemon of the current Docker context through its socket. It pulls the configured image (or, with `pull: false`, uses the image already on that daemon), starts the app container on the `weather-skills-eval` network, and removes it when the run ends, also when the run fails or the harness receives SIGTERM. Each app container is labeled with its run id (`weather-skills-eval.run`), and a run only touches its own container, so several runs can share one Docker daemon. A harness killed with SIGKILL cannot clean up: remove its container by that label by hand.

### Exit codes

| Code | Meaning |
|-|-|
| 0 | The run finished and the experiment was recorded |
| 1 | The run failed for any other reason |
| 2 | The run did not start: a harness setting, the configuration file, the dataset or a dataset item is invalid, the dataset is missing or empty, or the dataset needs a skill pack the configuration does not turn on |
| 3 | The app did not start (image pull, image not on the Docker host, container start or `/health`), or Docker failed |
| 4 | The app failed while the harness read its catalog model, tools and skill packs, or reported no base model for the catalog model |
| 5 | Langfuse returned an error |
| 143 | The harness was stopped with SIGTERM; the app container was removed |

## Configuration file

`configs/example.yaml` is a complete example. Its keys:

- `name`: the experiment name; defaults to the file name without `.yaml`, and is required when the configuration is read from stdin.
- `image`: the app image. Unless `pull` is `false`, it is pulled before every run, and the experiment records its digest.
- `pull`: `true` (the default) or `false`. With `false` the image is not pulled: the run uses the image already on the daemon of the current Docker context, such as one built there and never pushed, and stops when it is not there. With a remote context, the image must be built on that remote daemon. Its digest is recorded when the image has one for its repository, and is empty otherwise.
- `model`: id of the catalog model the chats use (an `id` in `CATALOG_MODELS`).
- `dataset`: the Langfuse dataset holding the tasks. The dataset's metadata may name the skill packs its tasks need as `skill_packs`, a list of pack names, such as `{"skill_packs": ["chc-skills"]}`. Without metadata, without `skill_packs`, or with an empty list, the dataset needs no pack. Metadata that is not an object, or a `skill_packs` that is not a list of non-empty names, stops the run. After reading the dataset and before starting the app, the harness checks that `SKILL_PACKS` in `env` turns on every pack the dataset names, and otherwise stops the run, naming each pack the configuration does not turn on. Every run of a dataset plays every item, so tasks that need a pack belong in a dataset of their own.
- `env`: the app's whole environment; nothing else is set.

Rules:

- Any other top-level key stops the run.
- `env` uses the app's variable names (see the app wiki pages Deployment, Skill Packs and Tracing). `WEBUI_SECRET_KEY` and the three `BOOTSTRAP_ADMIN_*` variables are required: the harness signs in with `BOOTSTRAP_ADMIN_API_KEY`. `SKILL_SANDLOCK` must be `false` where the Docker host has no Landlock.
- `${NAME}` is replaced with the harness's environment variable `NAME`; a reference to an unset or empty variable stops the run.
- A variable, or a key inside a mapping value, whose name has the word key, secret, password or token in it (singular or plural; `tokens` alone, as in `max_tokens`, does not count) must be exactly one `${NAME}` reference. A literal value stops the run.
- Values are strings. Mappings and lists are written to the variable as JSON, after their references are replaced.
- The app listens on `PORT` from `env`, 8080 when unset.
- `OPENAI_API_CONFIGS`, when set, must map each connection to an object whose `enable` is a boolean, `prefix_id` a string and `model_ids` a list of strings. The harness reads these three keys to name the base model (see [Scores](#scores)).
- `SKILL_PACKS`, when set, must be a list of objects, each with a non-empty string `git_url` and, when present, a boolean `enabled_by_default`. A pack is named by the last path segment of its `git_url`, ignoring surrounding whitespace and trailing slashes, without a trailing `.git` (`https://github.com/rhiza-research/chc-skills.git` is `chc-skills`), and is on when its `enabled_by_default` is `true` or absent. The harness reads these to check the dataset's `skill_packs`.
- The app and the harness must use the same Langfuse project: the harness reads the app's traces with its own keys. The example passes the harness's Langfuse keys and address to the app.

## Harness settings

`.env.example` lists them. The simulated user and judge models are harness settings so that they stay fixed across the runs being compared. `EVAL_SIMULATED_USER_MODEL` must accept image input: the simulated user is shown the presented images. Each of their calls is a Langfuse trace with user id `simulated-user` or `judge`, session id = the chat id, in the environment set by `LANGFUSE_TRACING_ENVIRONMENT`, which must not be `production`.

Every simulated-user and judge call, including the answer-quality rating, sends `max_tokens` = `EVAL_MODEL_MAX_TOKENS` (a whole number, at least 1, default 8192) unless the call sets its own. Without a limit, OpenRouter reserves the model's whole output window against the account balance and refuses calls with HTTP 402 when the balance is low. A simulated-user or judge reply cut off at that limit stops the play as `error`, naming the limit; a cut-off answer-quality rating does not stop the play, and `answer_quality` is recorded as not rated.

`EVAL_MAX_TURNS` (a whole number, at least 1, default 20) is the turn cap: the most user messages a play sends, to stop a play where the assistant or the simulated user loops. A task's `turn_limit` overrides it. Plays run until the simulated user is done, gives up or reaches the turn cap.

## Task format (one Langfuse dataset item per task)

```json
{
  "input": "I need next week's rain totals for western Kenya as a map",
  "expected_output": {
    "deliverables": ["image"],
    "judge_criteria": ["The assistant presents weekly rain totals for western Kenya"]
  },
  "metadata": {
    "facts": "Region: western Kenya. Period: the coming week. Units: mm.",
    "hidden_steps": ["Ask it to use the weekly aggregate instead of daily values"],
    "pauses": [{"before_turn": 3, "seconds": 600}],
    "tags": ["map", "kenya"]
  }
}
```

- `input` is the opening message, sent as the first turn. It must not be empty. Item ids must be unique.
- `deliverables` are the kinds of file the result must include, each `image` or `zarr`. Empty or absent means the task expects no file. An item needs at least one deliverable or judge criterion. Any other key in `expected_output` stops the run.
- `judge_criteria` are judged by Scenario's judge once the play has ended, as described under [Judging](#judging). The goal is reached when the simulated user is done and the judge passes every criterion.
- `facts` is what the simulated user answers from. `hidden_steps` are revealed one at a time, only when the assistant is stuck. See [Simulated user](#simulated-user).
- `pauses` wait before the given turn (turn 1 is the opening message, so the first possible pause is before turn 2).
- `turn_limit` is optional: the task's turn cap, in place of `EVAL_MAX_TURNS`. A pause past it stops the run; without it, a pause past `EVAL_MAX_TURNS` stops the run when it starts.
- `tags` are labels for grouping tasks in Langfuse. The harness does not read them; Langfuse shows them with the item's metadata.

## Simulated user

The simulated user's first message is the task's `input`. On every later turn it gets the transcript as the user sees it (each reply's text, without tool calls) and, when a reply has presented files, what a real user sees of the latest reply that presents files (see [Judging](#judging) for how replies present files): each presented file's path and kind, and each image, attached when its media type is `image/png`, `image/jpeg`, `image/gif` or `image/webp` and otherwise shown with the note `(not shown as an image)`; a listed image that could not be downloaded is shown as `(could not be opened)`; a presented path that is not in the chat's files, or every presented path when the listing fails, is shown with a note that it could not be opened. When a path is not listed yet, or the listing or a download failed, the listing and the failed downloads are read again on the next turn; images already downloaded for the same presenting reply are reused by path, not downloaded again. It never gets provenance or the harness's notes on why a result cannot be judged, so it judges what it received from the file list and the images, as a user would ("I asked for the data file and only got a picture"). It compares what it received, the answer's text and any files, with what it asked for and its facts. A reply that is done or gives up never counts a hint. The files are read once per presenting reply, except as above. It also gets the task's `facts` and `hidden_steps`, never the judge criteria.

Each of its replies says whether it is `done` or gives up. A reply that is not a JSON object, has an unknown classification, has a `done`, `give_up` or `revealed_hidden_step` that is not `true` or `false`, has `done` and `give_up` both true, has a `message` that is neither text nor null, or has an empty message while neither is true (except in the read at the turn cap, whose message is never sent) cannot be used: the call is made once more, shown that reply and a message saying why, and a second such reply stops the play as `error`. A model call that fails is retried by LiteLLM's `completion` (`num_retries`; checked on 1.105.1, the floor in `pyproject.toml`): every API error (any HTTP status, including 400, 402 and 429), timeout and connection error is retried at once, with no wait, up to `EVAL_MODEL_RETRIES` more times; when it still fails, the play stops as `error`. A reply cut off at `EVAL_MODEL_MAX_TOKENS` is not made again: it stops the play as `error`. An unusable reply is made once more, as above.

A play ends only when the simulated user is done, gives up, or the turn cap (`turn_limit`, else `EVAL_MAX_TURNS`) passes. At the cap the simulated user reads the last reply once more, without anything more being sent to the app, so a play whose assistant finishes on the last allowed reply can still reach the goal. Only that read's done or give-up counts (with its reasoning); it records no classification and no hint.

## Judging

The judge rules once, after the play has ended (for any outcome but `chat_error` and `error`), on the final result. A judge pass never ends a play. Scenario's judge is told the conversation reached the turn cap only when the outcome is `turn_cap`. The judge sees the transcript as the user sees it: each reply's text, without tool calls.

A task with no `deliverables` is judged on the transcript and its `judge_criteria`.

When a task has `deliverables`, the result is every file the assistant presented during the play, never the other files of the chat, which hold intermediate results; the criteria decide which of them the user wanted at the end:

1. A reply presents the `path` of a successful `display_image` call, the `attachments` of a successful `send_email` call, and the relative markdown image and link targets in its text (`![map](maps/rain.png)`, `[data](out.zarr)`). The result holds each presented path once, at the latest turn that presented it, in that order.
2. Each file's kind and provenance come from its entry in the chat's file listing. Every kind in `deliverables` must be among the presented files. Each file of the last reply that presents files must have provenance: the app reads the `weather_skills_history` chain of PNG images and zarr stores into the entry's `provenance_detail`.
3. Each presented image of type `image/png`, `image/jpeg`, `image/gif` or `image/webp` is downloaded once the play has ended, so a path presented again is judged on its latest version; a failed listing or download, or a listing that lacks a file of the last reply that presents files, is read once more, after waiting `EVAL_POLL_INTERVAL_SECONDS`, before it counts as a problem; when that extra listing fails, the earlier one is used and the problem also names the listing error. Any other image is given to the judge without its picture, with the note `(not shown as an image)`, which does not block. Nothing else is downloaded.
4. The judge rules on the `judge_criteria` plus a fixed criterion: each presented file's provenance (data source, region, period, aggregation, units) matches the user's request and what was agreed in the chat. Besides the transcript, its call holds one message of harness evidence: each file's path, kind and the turn that presented it, its `provenance_detail` as JSON, and each image.

When no reply presented a file, a kind is missing among all presented files, or a file of the last reply that presents files is not in the listing, has no provenance or cannot be downloaded, the judge model is not called, and the reason is the verdict. Such a problem on a file presented earlier does not stop the judge: the file is given to it with a note naming the problem, next to the turn that presented it.

## Outcomes

Each play ends with one outcome, given in the item output and in the `goal_reached` score comment:

| Outcome | Meaning |
|-|-|
| `goal_reached` | The simulated user was done and the judge passed every criterion, including the provenance criterion when the task has `deliverables` |
| `chat_error` | The app returned an error for a turn, or a reply the harness cannot read |
| `error` | Something else stopped the play, such as a harness model reply that cannot be used, or the final judge call raised or returned no verdict (the error then starts with `The final judge call failed:`) |
| `user_done` | The simulated user was done, and the judge's verdict failed a criterion or no result could be judged |
| `user_gave_up` | The simulated user gave up |
| `turn_cap` | The play sent as many user messages as the turn cap (`turn_limit`, else `EVAL_MAX_TURNS`) allows, and the simulated user was neither done nor gave up on the last reply |
| `not_reached` | Scenario's run ended for none of the reasons above |

For `user_done`, `user_gave_up`, `turn_cap` and `not_reached`, the reasoning ends with the judge's verdict, as `(final check passed: ...)` or `(final check failed: ...)`: the judge model's reasoning (`passed` or `no reasoning given` when it gave none), or why no result could be judged (it names the file, and the download error when there is one). For `goal_reached` the reasoning is the judge model's reasoning, or `passed` when it gave none.

## Scores

Each item of the experiment gets these scores. The "to goal" scores cover the whole play.

| Score | Meaning |
|-|-|
| `goal_reached` | 1 when the goal was reached, else 0 |
| `turns_to_goal` | User messages sent |
| `model_calls_to_goal` | The app's model calls, including failed ones |
| `clarifying_questions` | Assistant replies the simulated user classified as a clarifying question |
| `hints_used` | Hidden steps the simulated user revealed, at most the number of hidden steps |
| `answer_quality` | 1 to 5, rated by the judge model; not rated when no turn got a reply, or when the rating fails, such as a rating cut off at `EVAL_MODEL_MAX_TOKENS` |
| `cost_to_goal_usd` | Sum of each generation's `cost_details` total, or its total cost when that is missing |
| `tokens_uncached_input`, `tokens_cache_read`, `tokens_cache_write`, `tokens_output` | Per generation, `input` is the whole prompt; the cache counts are moved out of it. App-recorded generations carry them in `usage_details` (`cache_read_input_tokens`, `cache_creation_input_tokens`); OpenRouter Broadcast generations carry them as top-level metadata keys whose names contain the dots (`attributes.gen_ai.usage.input_tokens.cached`, `attributes.gen_ai.usage.input_tokens.cache_write`) |
| `tool_calls`, `tool_failures` | Tool observations in the app's traces, and those that failed: at level ERROR, or whose output is a JSON object with `"ok": false` (the app records a failed skill run at level DEFAULT with that output). Only tool observations' inputs and outputs are read |
| `cache_miss_calls` | Model calls that wrote at least 50,000 tokens of context (uncached input plus cache write) |
| `time_to_answer_seconds` | Time the app spent on turns that got a reply, from sending each message until its reply was done; pauses and failed turns are excluded |

Trace-based scores (model calls, cost, tokens, tools, cache misses) are read from the app's Langfuse traces of the chat: traces named `weather-skills-chat` with session id = chat id, one per turn the app accepted, once they stop changing. They have settled only once every tool observation's output has been read. A tool's output is read once the tool has ended, also when its output is null; a tool still running is read again on a later poll. When the traces have otherwise settled and the only unread tools have no end time (such as a span a turn timeout left open), their outputs are read once as they are; the traces count as complete when that read returned every one of them, and as partial (`the outputs of tools left open could not be read: <count>`) otherwise. When that read found new traces or observations, polling goes on, without the outputs it read for tools still open. A Langfuse read that fails with a transport error, a timeout, or a 429 or 5xx response is retried at `EVAL_POLL_INTERVAL_SECONDS` until `EVAL_TRACE_TIMEOUT_SECONDS`; the failed read does not restart the settle time, which only a change in the observations does. Any other error (such as a 400, 401, 403 or 404, or a Langfuse error without a status code) makes the trace scores missing at once when no read has succeeded yet, and partial, from the last successful read, when one has; a timeout before any read succeeded makes them missing too. The item output's `trace_scores` says whether they are `complete`, `partial` at the timeout (`partial: Langfuse reads failed: <error>` when the last read failed, `partial: traces were still arriving` when traces were missing or still changing, else `partial: tool outputs not read by the timeout: <count>`) or `missing` (with the reason, such as a Langfuse error, the last read's error when the reads that succeeded found no traces, or no traces arriving); when they are missing, the `goal_reached` comment says so too and the other scores are still recorded. Each item's output holds the chat id, the app's trace ids and `provider_models`, and the item's trace carries the chat id as its session id. `provider_models` lists the distinct `model` names of the app's generations in those traces: the model OpenRouter reports for calls it records with Broadcast, the model the app requested for calls the app records. It is empty when the trace scores are missing.

The experiment's per-item cost in Langfuse is the app's cost: the item's trace holds one generation named `app-cost-rollup` whose cost is the item's `cost_to_goal_usd`, with no model and no token counts. It is recorded whenever that score is, and leaves out the simulated user and judge. The rollup repeats the app's cost: a cost total over the chat's session, or across the `sdk-experiment` environment (where Langfuse records the item traces) and the app's, counts it twice unless generations named `app-cost-rollup` are left out.

The experiment's metadata holds the resolved configuration: image with its digest and image id and whether it was pulled, the app's environment variable names, installed skill packs with their commits, the tool ids the chats used, the chat model as `model` and `catalog_model`, the simulated user and judge models, the turn cap `EVAL_MAX_TURNS` as `max_turns`, the harness commit, and the dataset with its version, item count and the skill packs it needs (`dataset_skill_packs`, comma-separated, empty when none). `catalog_model` is the configured catalog id; `model` is the base model the app reports for it, without the `prefix_id` of the connection the app serves it from: like the app, the harness takes the last connection in `OPENAI_API_CONFIGS` that is enabled and whose `model_ids` allow the model (`open-router-.anthropic/claude-haiku-5.5` becomes `anthropic/claude-haiku-5.5`). `model` is also the experiment's description. A base model can be a floating alias such as `~anthropic/claude-opus-latest`; the metadata is written before the plays, so the models that served the calls are only in each item's `provider_models`.

A run plays each task once: Langfuse expects each dataset item to appear once per experiment (Langfuse issue #5855).

## Development

Tests and lint, in a container without network access:

```sh
docker compose run --rm --build test
```

Install the pre-commit hooks:

```sh
uv run pre-commit install
```
