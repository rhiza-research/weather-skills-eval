# weather-skills-eval

Measures what a change to weather-skills-chat (model, prompt, code, skills, settings) does to success, cost and behavior on real tasks.

`weather-skills-eval run` starts a throwaway app container from its settings (see [Harness settings](#harness-settings)) and one model of a configs repo commit (see [Configs repo](#configs-repo)), plays every task of a Langfuse dataset through the app's chat API with a simulated user and a panel of judges ([LangWatch Scenario](https://github.com/langwatch/scenario)), scores each play, and records the run as one Langfuse experiment (see [Experiment name and served model](#experiment-name-and-served-model)). Two runs on the same dataset are compared side by side in Langfuse.

The harness talks to the app only over HTTP and Docker. The app's skills fetch live data from the data providers.

## Terms

- **Harness**: the program in this repository, `weather-skills-eval`.
- **Run**: one `weather-skills-eval run`. It plays every task of the dataset once with one model of one configs repo commit, and records one Langfuse experiment.
- **Batch**: the runs one launch of `scripts/run-eval.py` starts: each picked model once per repetition.
- **Task**: one item of the Langfuse dataset (see [Task format](#task-format)).
- **Play**: one task played once, as one chat with the app.
- **Turn**: one user message and the app's reply to it. Turn 1 is the opening message.
- **Configs repo**: the repository that defines the app's models, skill packs, connections and settings (weather-skills-configs).
- **Model id**: the `model.id` of a file in the configs repo's `models/`, such as `weather-agent-opus`. A run plays one model id.
- **Base model**: the model the app reports for a model id, such as `anthropic/claude-haiku-5.5`.
- **Served model**: the model OpenRouter names in its reply to a request for the base model.
- **Variant**: the `eval.variant` of a model file. It is appended to the experiment name in parentheses.
- **Docker host**: the host of the current Docker context.
- **Default mode**: `scripts/run-eval.py` without `--background`. It asks its questions on your machine and launches the batch (see [The default mode](#the-default-mode)).
- **Background mode**: `scripts/run-eval.py --background`. It asks nothing and runs the batch in the batch container (see [Background mode](#background-mode)).
- **Run id**: 12 hexadecimal characters the harness generates for each run. It is in the names and the label of the run's containers and volume. It is not recorded in Langfuse.
- **Run name**: the run name Langfuse returns for the run's experiment, printed in the run summary. It is not the run id.

## Before you start

You need:

- `uv`, which runs `scripts/run-eval.py`.
- `git` with read access to the configs repo.
- For a launch on a new VM, which is the built-in `EVAL_LAUNCH`: the `gcloud` CLI, logged in with `gcloud auth login`, with an account that can create and delete instances in the project `rhiza-shared`.
- For a launch on the Docker host: the `docker` CLI. `gcloud auth application-default login` is needed only when a secret is empty in both the shell and `.env` (see [Secrets](#secrets)).

Requirements:

- `EVAL_JUDGE_MODELS` names at least one judge model, and `LANGFUSE_BASE_URL`, `LANGFUSE_TRACING_ENVIRONMENT`, `EVAL_CONNECTION_KEYS`, `EVAL_DATASET` and `EVAL_ADMIN_EMAIL` are set, each by flag, in the shell or in `.env`. `.env.example` sets all six.
- The configs repo is a public GitHub repository, given as `https://github.com/<owner>/<repo>`: the batch downloads its files without a token.
- `EVAL_SIMULATED_USER_MODEL` accepts image input: the simulated user is shown the presented images. Each judge model accepts image input when a task has an `image` deliverable: each judge's verdict call holds the presented images (see [What a judge sees](#what-a-judge-sees)).
- `LANGFUSE_TRACING_ENVIRONMENT` is not `production`.
- For a launch on a new VM, every secret is in Secret Manager. A VM launch never uses a secret from the shell or `.env`: the VM reads every secret from Secret Manager. The VM's service account can read only the secrets the infrastructure repository grants it one by one, in `terraform/20-gke-cluster/weather-skills-eval.tf`. A variable newly named in `EVAL_CONNECTION_KEYS` needs its secret and its grant there.
- CI publishes the eval image for `linux/amd64` only.

## Running an eval

An eval tests one build of the chat app with one or more models from the configs repo. Each model's run plays every task of the dataset.

1. Copy the settings file once: `cp .env.example .env`. Leave a secret empty to have it read from Secret Manager (see [Secrets](#secrets)).
2. Run `uv run scripts/run-eval.py`. It asks for each of the answers (see [Batch values](#batch-values)), such as the models and the app image to test, prefilled with a default. It prints the estimated cost (see [Cost estimate](#cost-estimate)), and last asks where to run the batch: on a VM it creates for this batch, on the Docker host, or neither, which only prints the commands.
3. On a VM, it prints the VM's name and two commands: one to view the batch's output, and one to stop the VM. The VM deletes itself when the batch ends. The batch keeps running when you close your machine. On the Docker host, it prints the commands that follow and stop the batch (see [Following a batch](#following-a-batch)).
4. Each run's results are a Langfuse experiment (see [Reading the results](#reading-the-results)). The runs of one batch record the same `batch` in their metadata.

To change a default, set its value in `.env`, such as `APP_IMAGE`, `RUNNER_IMAGE` or `EVAL_MODELS` (see [Batch values](#batch-values)). To start one run by hand while debugging, see [Debugging one run](#debugging-one-run). [How a batch and a run work](#how-a-batch-and-a-run-work) describes each step in detail.

### Cost estimate

`scripts/run-eval.py` estimates a batch's cost from `.costs.json` and OpenRouter's prices. `.costs.json`, at the repository root and not checked in, holds the agent cost of each model's latest run, by model id:

```json
{"weather-agent-opus": {"served_model": "anthropic/claude-opus-5.5", "agent_cost_usd": 1.23}}
```

`scripts/run-eval.py` is the only program that reads or writes it. When a run ends, it saves the agent cost on the run's cost line (see [Run summary](#run-summary)) as that model's latest cost. Only a run that prints its model's cost line updates the saved cost, and a run prints its cost line only when it exits 0. A failed run leaves that model's saved cost as it was.

The estimate for the batch is printed as one line, with the agents' cost and the judges' and simulated user's cost apart:

```
Estimated cost: $4.29 (agents $3.00, judges and simulated user $1.29)
```

- Agents: each picked model's saved agent cost, times the repetitions.
- Judges and simulated user, per run of a model: each picked judge model's price times 102,000 input and 14,000 output tokens, plus the simulated user model's price times 48,000 input and 15,000 output tokens, token counts measured over the 2026-10-22 runs. This is multiplied by the number of picked models and the repetitions. The prices are `pricing.prompt` and `pricing.completion` (USD per token) from OpenRouter's public models list, `https://openrouter.ai/api/v1/models`, read without a key, for the model name without `openrouter/`.
- A picked model with no saved cost is left out of the total and named on `Agent cost unknown, not counted: <model ids>`; a judge or simulated user model the list does not price, or prices as absent, unparsable or negative, is left out and named on `No OpenRouter price, not counted: <models>`.

With a `.costs.json` that does not exist, the estimate is printed and every picked model is named on the `Agent cost unknown, not counted: <model ids>` line. A `.costs.json` that cannot be read, or a models list that cannot be fetched, prints `Cost estimate unavailable: <reason>` and the script goes on; the estimate never stops a batch. A cost line is not saved into a `.costs.json` that cannot be read, and a failed write prints `Cost not saved: .costs.json could not be written: <reason>`.

## Following a batch

A batch's output is the batch container's output:

- On a VM: `gcloud compute instances get-serial-port-output <VM> --zone us-central1-a --project rhiza-shared`, which the launch prints. The instance template turns on no logging of the serial console, so the output is not kept once the VM is deleted. After that, the runs that reached Langfuse are found by `metadata.batch` (see [Reading the results](#reading-the-results)); a run that failed before its experiment was created leaves no record.
- On the Docker host: `docker logs -f weather-skills-eval-batch`. The batch container is started with `--rm`, so it is removed when the batch ends, and its output is not kept after that.

The batch prints `Batch <batch id>`, the batch id being the UTC time it started its runs (`2026-10-22T20:07:57Z`). It prints each run's stdout and stderr, line by line. When `EVAL_PARALLEL_RUNS` is above 1, each line is prefixed with `[<model> run <round>] `. When a run ends, the batch prints its exit code (`<model> run <round>: exit <code>`). The batch goes on after a run exits non-zero, and the batch itself exits 1 when any run failed.

### Progress

While a run lasts, `run` prints its progress on stderr, one message per line. The first progress line is printed once the dataset is read. Every error that exits 2 prints `Run not started: <error>` on stderr. A harness setting error, a configs repo error and a dataset that is missing, empty or unreadable print it before any progress line. Invalid dataset metadata, a skill pack the configs repo commit does not turn on, and an invalid dataset item print it after the dataset line.

The main steps of starting the run come first, in this order:

- The dataset read, with its item count: `Dataset weather-eval-tasks read: 12 items`.
- The app image pull as it begins and ends: `Pulling image <image>`, `Image <image> pulled`. With `EVAL_PULL_APP_IMAGE=false`: `Image <image> found on the Docker host`.
- The runner image pull: `Pulling runner image <image>`, `Runner image <image> pulled`.
- The containers: `Runner container <name> started`, `App container <name> started`.
- The wait for `/health` and its end: `Waiting up to <EVAL_APP_START_TIMEOUT_SECONDS>s for <app container> to answer /health`, `App container <name> answered /health`.
- The served model: `OpenRouter serves <base model> as <served model>`.
- The experiment: `Experiment <experiment name> started: <number of tasks> tasks, concurrency <EVAL_CONCURRENCY>`.

Then each task when it starts and when it finishes, numbered out of the total:

```
Task 3 of 12 started: wind
Task 3 of 12 finished: wind, goal_reached, 4m12s; 5 of 12 finished, 18m elapsed
```

- The task's name on these lines is its Langfuse item id.
- The finished line gives the task's outcome (see [Outcomes](#outcomes)), the time from the task's start, how many tasks have finished, and the time since the run started. For `chat_error` and `error`, the outcome is followed by the first line of the error (`chat_error: <error>`).
- Times are in whole seconds, leaving out each part that is 0 (`18m`, `1h5s`).
- Each run plays the tasks in a random order drawn for that run. This makes runs going at the same time less likely to reach the same data providers with the same task at the same time. Tasks are numbered in the order they start.
- The finished line of a lost task (see [Sandbox runner](#sandbox-runner)) gives `not scored, the sandbox runner restarted during it` in place of the outcome.
- Any other task that raised gives `raised <error type>: <first line of the error>` in place of the outcome, or `raised <error type>` when the error has no message.
- A task not played because the sandbox runner had already failed logs `Task <n> of <total> skipped: <task id>, the sandbox runner failed` instead of a started and a finished line. A skipped task takes a number and is never counted as finished.

After the last task, the harness logs `Waiting for the runner after the last task` when it waits for the runner then (see [Sandbox runner](#sandbox-runner)). It logs `Removing the run's containers and volume` before it removes them, also when the run fails after the app image was read. The harness's warnings, such as that it is waiting for the runner, are printed on stderr the same way.

### Run summary

A run that exits 0 prints its summary on stdout, in this order:

1. `Experiment <experiment name> (<model id>), run <run name>`.
2. `Goal reached on <x> of <y> tasks`.
3. When tasks were lost: `<n> tasks not scored because the sandbox runner restarted during them: <task ids>`, the ids separated by `, `. For one lost task: `1 task not scored because the sandbox runner restarted during it: <task id>`.
4. The experiment URL, when Langfuse returned one.
5. The cost line.

A task that raised, a lost task included, has no scores: its item records the error. Langfuse leaves a task that raised out of the experiment's results, so `<y>` is the number of tasks that returned a result and does not count it.

The cost line is `Cost ` followed by JSON with the run's model id as `config_name`, the served model as `served_model`, without the ` (<variant>)` suffix the experiment name carries, and the sum of the items' `cost_to_goal_usd` as `agent_cost_usd`, rounded to 6 decimal places (`Cost {"config_name": "weather-agent-opus", "served_model": "anthropic/claude-opus-5.5", "agent_cost_usd": 1.23}`). An item without that score, such as one whose trace scores are missing, adds nothing.

### Exit codes of one run

| Code | Meaning |
|-|-|
| 0 | The run finished and the experiment was recorded, also when tasks were lost (see [Sandbox runner](#sandbox-runner)) |
| 1 | The run failed for any other reason |
| 2 | The run did not start: a harness setting is invalid, a configs repo file cannot be read or parsed or is of another schema version, a connection at the commit has no key (see [Configs repo](#configs-repo)), the dataset or a dataset item is invalid, the dataset is missing, empty or could not be read from Langfuse (any error, a Langfuse error included), or the dataset needs a skill pack the configs repo commit does not turn on |
| 3 | The app or the sandbox runner did not start (an app image without the sandbox emulator, image pull, a pulled image with no digest for its repository, image not on the Docker host, container start or `/health`), the runner was not running steadily by `EVAL_APP_START_TIMEOUT_SECONDS` before a task, or after the last task when no task waited for it after it was lost, the runner was found exited with code 0, `dead`, disappeared or unreadable before or after a task, or Docker failed |
| 4 | The app's skill state could not be made to match the configs repo, or the app failed while the harness read the model's entry, tools and skill packs, or reported no base model for the model id, or, after the dataset was read, a network error (an `httpx` error, such as a connection error or timeout) reaching the app or Langfuse stopped the run |
| 5 | Langfuse returned an error response after the dataset was read |
| 143 | The harness was stopped with SIGTERM; the app and runner containers and the volume were removed |

A run that stops with exit 1, 3, 4, 5 or 143 after its experiment started leaves a partly recorded experiment in Langfuse, which `delete` removes (see [Deleting a run](#deleting-a-run)). A partly recorded experiment has no printed URL: the URL is printed only on exit 0. Find it in Langfuse by the experiment name on the run's `Experiment <experiment name> started` progress line.

### Stopping a batch and cleanup

- On a VM: the `gcloud compute instances delete <VM> --zone us-central1-a --project rhiza-shared` command the launch prints deletes the VM. The runs in flight when the VM is deleted leave partly recorded experiments (see [Deleting a run](#deleting-a-run)).
- On the Docker host: `docker stop weather-skills-eval-batch` stops the batch and removes the container.

`docker stop` sends SIGTERM, which stops the batch. The batch starts no further run. It sends SIGTERM to each `docker compose run` still going, so each run removes its containers, and it waits for them. Then it prints `Stopped. Results so far:` with the exit code of each run that ended before the SIGTERM, and exits 143. `--stop-timeout 120` gives the batch 120 seconds before Docker kills it.

The eval service reaches the Docker daemon of the Docker host through its socket. It pulls the app image (or, with `EVAL_PULL_APP_IMAGE=false`, uses the image already on that daemon) and the runner image, starts a runner container and then the app container on the `weather-skills-eval` network (see [Sandbox runner](#sandbox-runner)), and removes both containers and the run's volume when the run ends, also when the run fails or the harness receives SIGTERM. Each container and volume is labeled with its run id (`weather-skills-eval.run=<run id>`), and a run only touches its own, so several runs can share one Docker daemon.

A harness killed with SIGKILL cannot clean up. Its containers and volume keep the label and are removed by hand. The label is on every container the run creates (the app, the runner, and the emulator and volume-setup containers) and on its volume. The run id is the last part of the app container's name, `weather-skills-eval-app-<run id>`, which the progress lines print. These commands list what one run left:

```sh
docker ps -a --filter label=weather-skills-eval.run=<run id>
docker volume ls --filter label=weather-skills-eval.run=<run id>
```

When the run id is not known, the filter without a value lists the containers and volumes of every run:

```sh
docker ps -a --filter label=weather-skills-eval.run
docker volume ls --filter label=weather-skills-eval.run
```

`docker rm -f <container>` and `docker volume rm <volume>` remove what these commands list.

## Reading the results

Compare the scores of two runs only when their experiments record the same `comparability_hash` (see [Comparing runs](#comparing-runs)).

Each run is one Langfuse experiment. Filtering Langfuse experiments on `metadata.batch` shows the runs of one batch. A run plays each task once: Langfuse expects each dataset item to appear once per experiment (Langfuse issue #5855). Several plays per task are `EVAL_REPETITIONS` (see [Batch values](#batch-values)): each repetition is a run of its own, with its own experiment.

### Outcomes

Each item's output holds these keys:

- `chat_id`: the app's chat id.
- `outcome` and `goal_reached` (`true` or `false`).
- `reasoning` and `error`.
- `judges`: each judge's `model`, `state`, `passed` and `reasoning` (see [The panel and its majority](#the-panel-and-its-majority)); empty when the panel did not rule.
- `trace_scores`, `app_trace_ids` and `provider_models` (see [Trace-based scores](#trace-based-scores)).
- `paused_seconds`: the seconds the play waited in the task's `pauses`.
- `scores`: each score's `name`, `value` and `comment` (see [Scores](#scores)).

A judge is named as `EVAL_JUDGE_MODELS` names it in `reasoning`, `error` and `judges`, and without `openrouter/` in score names and score comments.

Each play ends with one outcome, given in the item output and in the `goal_reached` score comment:

| Outcome | Meaning |
|-|-|
| `goal_reached` | The simulated user was done and the judge panel passed every criterion, including the provenance criterion when the task has `deliverables` |
| `chat_error` | The app returned an error for a turn, or a reply the harness cannot read, a request to the app failed without a response (any `httpx` error, such as a connection error or a timeout), or the reply was not done within `EVAL_TURN_TIMEOUT_SECONDS` (see [Harness settings](#harness-settings)) |
| `error` | Something else stopped the play, such as a harness model reply that cannot be used, or every judge's call failed or returned no verdict (the error then reads `No judge ruled: <model> errored: <error>`, one per judge, separated by `; `) |
| `user_done` | The simulated user was done, and the panel failed a criterion or no result could be judged |
| `user_gave_up` | The simulated user gave up |
| `turn_cap` | The play sent as many user messages as the turn cap (`turn_limit`, else `EVAL_MAX_TURNS`) allows, and the simulated user was neither done nor gave up on the last reply |
| `not_reached` | Scenario's run ended for none of the reasons in the other rows |

For `user_done`, `user_gave_up`, `turn_cap` and `not_reached`, the item's `reasoning` ends with the panel's verdict and reasoning, as `(final check passed: <panel reasoning>)` or `(final check failed: <panel reasoning>)`. For `goal_reached` the item's `reasoning` is the panel's reasoning.

The panel's reasoning is the judges' reasoning when they all gave the same verdict and the same reasoning. Otherwise it gives each judge's, as `<model> passed: <reasoning>`, `<model> failed: <reasoning>` or `<model> errored: <error>`, separated by `; `.

A judge's reasoning is its model's reasoning. When the model gave none, it is `passed` for a judge that passed and `no reasoning given` for one that failed. When no result could be judged (see [What a judge sees](#what-a-judge-sees)), every judge's reasoning is the problem, one of:

- `No result presented: no reply presented a file`.
- `The chat's files could not be listed: <error>`.
- `No presented file is a <kind>; presented: <path> (<kind>), ...` (`an` before `image`).
- `Presented files are not in the chat's files: <paths>`, followed by `; the listing read again failed: <error>` when the extra listing failed.
- `Presented files have no provenance: <paths>`.
- `<path> could not be read: <error>`, with the download error.

### Scores

Each item of the experiment gets these scores. The "to goal" scores cover the whole play.

| Score | Meaning |
|-|-|
| `goal_reached` | 1 when the goal was reached, else 0 |
| `turns_to_goal` | User messages sent |
| `model_calls_to_goal` | The app's model calls, including failed ones |
| `clarifying_questions` | Assistant replies the simulated user classified as a clarifying question |
| `hints_used` | Hidden steps the simulated user revealed, at most the number of hidden steps |
| `answer_quality:<model>` | 1 to 5, rated by that judge model, named without its `openrouter/` prefix (`answer_quality:openai/gpt-6.1-sol`); not rated when no turn got a reply, or when the rating fails, such as a rating cut off at `EVAL_MODEL_MAX_TOKENS` |
| `answer_quality` | The mean of the judges' `answer_quality:<model>` scores; its comment says how many of the judges rated and names each that did not with its reason (`not rated by <model>: <reason>`), separated by `; `; not recorded when none rated |
| `judges_agree` | 1 when every judge that ruled gave the same verdict, else 0; its comment gives each judge's verdict, separated by `; `, as `<model> errored: <error>` for an errored judge; not recorded when the panel did not rule, fewer than two judges ruled, or no judge model was called (no result could be judged) |
| `cost_to_goal_usd` | Sum of each generation's `cost_details` total, or its total cost when that is missing; a generation with neither adds nothing, and the comment then reads `<n> model calls carried no cost` |
| `tokens_uncached_input`, `tokens_cache_read`, `tokens_cache_write`, `tokens_output` | Per generation, `input` is the whole prompt; the cache counts are moved out of it. App-recorded generations carry them in `usage_details` (`cache_read_input_tokens`, `cache_creation_input_tokens`); OpenRouter Broadcast generations carry them as top-level metadata keys whose names contain the dots (`attributes.gen_ai.usage.input_tokens.cached`, `attributes.gen_ai.usage.input_tokens.cache_write`) |
| `tool_calls`, `tool_failures` | Tool observations in the app's traces, and those that failed: at level ERROR, or whose output is a JSON object with `"ok": false` (the app records a failed skill run at level DEFAULT with that output). Only tool observations' inputs and outputs are read |
| `cache_miss_calls` | Model calls that wrote at least 50,000 tokens of context (uncached input plus cache write) |
| `time_to_answer_seconds` | Time the app spent on turns that got a reply, from sending each message until the first poll of the chat that found its reply done, so each turn includes up to one `EVAL_POLL_INTERVAL_SECONDS` after the reply was done; pauses and failed turns are excluded |

`answer_quality` rates the text transcript against the opening message and the task's `judge_criteria`. The rater is not shown files, images or provenance.

### Trace-based scores

Trace-based scores (model calls, cost, tokens, tools, cache misses) are read from the app's Langfuse traces of the chat: traces named `weather-skills-chat` with session id = chat id, one per turn the app accepted. They are read once they stop changing: there is one trace per accepted turn, every tool observation's output has been read, and the observations have not changed for `EVAL_TRACE_SETTLE_SECONDS`. A tool's output is read once the tool has ended, also when its output is null; a tool still running is read again on a later poll.

A Langfuse read that fails with a transport error, a timeout, or a 429 or 5xx response is retried at `EVAL_POLL_INTERVAL_SECONDS` until `EVAL_TRACE_TIMEOUT_SECONDS`. The failed read does not restart the settle time, which only a change in the observations does. Any other error (such as a 400, 401, 403 or 404, or a Langfuse error without a status code) makes the trace scores missing at once when no read has succeeded yet, and partial, from the last successful read, when one has. A timeout before any read succeeded makes them missing too.

The item output's `trace_scores` is one of:

- `complete`.
- `partial: Langfuse reads failed: <error>`: the last read failed.
- `partial: traces were still arriving`: at the timeout, traces were missing or still changing.
- `partial: tool outputs not read by the timeout: <count>`.
- `partial: the outputs of tools left open could not be read: <count>`: tools without an end time, such as a span a turn timeout left open.
- `missing: <reason>`: the reason is a Langfuse error, the last read's error when the reads that succeeded found no traces, no traces arriving, or no turn reaching the app.

When the trace scores are missing, the `goal_reached` comment says so too and the other scores are still recorded.

Each item's output holds the chat id, the app's trace ids and `provider_models`, and the item's trace carries the chat id as its session id. `provider_models` lists the distinct `model` names of the app's generations in those traces: the model OpenRouter reports for calls it records with Broadcast, the model the app requested for calls the app records. It is empty when the trace scores are missing.

### Cost in Langfuse

The experiment's per-item cost in Langfuse is the app's cost: the item's trace holds one generation named `app-cost-rollup` whose cost is the item's `cost_to_goal_usd`, with no model and no token counts. It is recorded whenever that score is, and leaves out the simulated user and judge. The rollup repeats the app's cost: a cost total over the chat's session, or across the `sdk-experiment` environment (where Langfuse records the item traces) and the app's, counts it twice unless generations named `app-cost-rollup` are left out.

### Experiment metadata

The experiment's metadata holds the resolved configuration. All values are strings.

| Key | Meaning |
|-|-|
| `config_name` | The model id |
| `served_model` | The served model (see [Experiment name and served model](#experiment-name-and-served-model)) |
| `image` | `APP_IMAGE` |
| `image_digest` | The app image's digest (see [Harness settings](#harness-settings)); empty when it has none |
| `image_id` | The app image's id |
| `image_pulled` | `true` or `false`: whether the run pulled the app image |
| `runner_image` | `RUNNER_IMAGE` |
| `runner_image_digest` | The runner image's digest |
| `runner_image_id` | The runner image's id |
| `model` | The base model, without its connection's `prefix_id` |
| `catalog_model` | The model id |
| `env_var_names` | The names of the app's environment variables the harness set from its settings and the configs repo, sorted, comma-separated |
| `skill_packs` | The installed skill packs with their commits, as JSON: each pack's `git_url`, `git_ref` and `commit_sha` |
| `tool_ids` | The tool ids the chats used, comma-separated |
| `simulated_user_model` | `EVAL_SIMULATED_USER_MODEL` |
| `judge_models` | The judge models, sorted, comma-separated |
| `comparability_hash` | See [Comparing runs](#comparing-runs) |
| `max_turns` | The turn cap `EVAL_MAX_TURNS` |
| `harness_commit` | The harness commit: `HARNESS_COMMIT` is baked into the eval image; without it the value is `unrecorded` |
| `batch` | `EVAL_BATCH`, the batch id; empty when unset |
| `app_image_revision`, `app_image_pr_number` | The app image's revision and PR number labels, as the batch read them; empty when unset |
| `configs_repo`, `configs_commit` | `EVAL_CONFIGS_REPO` and `EVAL_CONFIGS_COMMIT`, as given |
| `dataset` | `EVAL_DATASET` |
| `dataset_version` | The dataset's version, as an ISO date and time, when the dataset Langfuse returned has one; else `latest` |
| `dataset_items` | The item count |
| `dataset_skill_packs` | The skill packs the dataset needs, comma-separated; empty when none |

The harness reads the dataset without a version, so the run played the dataset as it was when the run read it.

`model` is the base model the app reports for the model id, without the `prefix_id` of the connection the app serves it from. Like the app, the harness takes the last connection in `OPENAI_API_CONFIGS` that is enabled and whose `model_ids` allow the model (`open-router-.anthropic/claude-haiku-5.5` becomes `anthropic/claude-haiku-5.5`). A connection without `enable` is enabled, and one with no or empty `model_ids` allows every model. When no connection is taken, `model` is the base model the app reports, with surrounding whitespace removed. `model` is also the experiment's description.

### Comparing runs

`comparability_hash` is the first 16 hex characters of the SHA-256 of a JSON object (keys sorted, no whitespace) holding what grades a run:

- The judge models, sorted, so their order does not change it.
- The simulated user model.
- Every dataset item's `id`, `input`, `expected_output` and `metadata`, in order of id, so the order a run plays them in does not change it.
- The grading version, which is raised by hand when grading changes in a way the prompt texts do not show.
- One SHA-256 computed over all of the following together:
  - Four judge texts: the scenario description template, the start-time line, the provenance criterion and the evidence header.
  - Five simulated-user texts: the system prompt, the conversation header, the files header, the reply request and the retry request.
  - The answer-quality prompt.
  - The installed `langwatch-scenario` version, as a version string. Scenario's judge system prompt is in that package.
- `EVAL_MAX_TURNS`, `EVAL_MODEL_MAX_TOKENS` and `EVAL_TURN_TIMEOUT_SECONDS`.

Two runs with the same hash have all of these in common, so their scores can be compared; a different hash means one of these differs. Of the harness settings, the hash holds only `EVAL_JUDGE_MODELS`, `EVAL_SIMULATED_USER_MODEL`, `EVAL_MAX_TURNS`, `EVAL_MODEL_MAX_TOKENS` and `EVAL_TURN_TIMEOUT_SECONDS`. Two runs with the same hash can differ in every other harness setting, such as `EVAL_CONCURRENCY`, `EVAL_APP_START_TIMEOUT_SECONDS`, `EVAL_POLL_INTERVAL_SECONDS`, `EVAL_TRACE_TIMEOUT_SECONDS`, `EVAL_TRACE_SETTLE_SECONDS`, `EVAL_MODEL_RETRIES`, `EVAL_RATE_LIMIT_RETRIES` and `EVAL_RATE_LIMIT_BACKOFF_SECONDS`. What is under test is left out too: the model id and its base model, the app and runner images, the skill packs and the app's system prompt.

### Experiment name and served model

After reading the base model and before the plays, the harness sends one short request for it to OpenRouter with its own `OPENROUTER_API_KEY` (not the app's key). The request is retried as `EVAL_MODEL_RETRIES` says and, when rate-limited, as `EVAL_RATE_LIMIT_RETRIES` says. It is not traced in Langfuse. The base model must therefore be an OpenRouter model name.

The model named in the reply is the served model. The harness records it in the metadata as `served_model`, without a variant suffix, and names the experiment after it, followed by ` (<variant>)` when the model's `eval.variant` is set (`z-ai/glm-5.3 (reasoning_high)`).

A floating alias such as `~anthropic/claude-opus-latest` resolves this way to the model it points to, such as `anthropic/claude-opus-5.5`, as it was when the run started. When that request fails, or its reply names no model or only the alias, the run stops before the experiment is created (exit 1). The models that served each play's calls are in each item's `provider_models`.

## Debugging one run

To debug, one run of one model can be started by hand, without `scripts/run-eval.py`:

```sh
HARNESS_COMMIT=$(git rev-parse HEAD) docker compose build eval
git -C <configs repo checkout> archive <commit SHA> | docker compose run --rm -T --entrypoint sh \
  -e EVAL_CONFIGS_REPO=<configs repo URL> -e EVAL_CONFIGS_COMMIT=<commit SHA> \
  -e EVAL_CONFIGS_DIR=/configs-repo -e EVAL_MODEL=<model id> \
  -e APP_IMAGE=<image> -e RUNNER_IMAGE=<image> [-e EVAL_CONCURRENCY=<n>] \
  eval -c 'mkdir -p /configs-repo && tar -x -f - -C /configs-repo && exec weather-skills-eval run'
```

The configs repo files reach the run as a tar on stdin, which the container unpacks at `/configs-repo` before the harness starts. This works with a remote Docker context, where a bind mount from your machine would not reach the daemon.

compose gives the eval service every variable of `.env` (`env_file` in `compose.yaml`), so a harness setting set in `.env`, such as `APP_IMAGE`, `RUNNER_IMAGE`, `EVAL_CONCURRENCY`, `EVAL_JUDGE_MODELS` or `EVAL_SIMULATED_USER_MODEL`, reaches the run without `-e`. A value passed with `-e` is used in place of the one in `.env`. `EVAL_BATCH` is optional and passed the same way. `.env.example` sets neither `APP_IMAGE` nor `RUNNER_IMAGE`. The secrets come from `.env` or `-e` only (see [Secrets](#secrets)).

With `EVAL_PULL_APP_IMAGE=true`, as `.env.example` sets it, the run pulls the app image before it starts.

## Deleting a run

`delete` removes one run's experiment from Langfuse, such as a partly recorded one. Build the eval image once:

```sh
docker compose build eval
```

Then run one of two commands. The first prints what it found and asks for `yes` before it deletes:

```sh
docker compose run --rm -it eval delete <experiment-id>
```

The second deletes without asking:

```sh
docker compose run --rm eval delete --yes <experiment-id>
```

The experiment id is the `baseline=` value in the experiment's Langfuse URL. Langfuse has no call that deletes an experiment; the experiment goes away once its traces are deleted. `delete` lists the experiment's items and finds each item's trace plus every trace in the session whose id is the item output's `chat_id`: the app's `weather-skills-chat` traces, the `simulated-user` and `judge` traces, and the item trace. For an item whose output has no chat id, it reads the item trace's observations and takes their session ids as the item's chats: the item trace carries the chat when the play returned and the runner check that follows the play failed. A play that itself raised links nothing, so `delete` removes only its item trace: that chat's `weather-skills-chat`, `simulated-user` and `judge` traces stay. Any other item without a chat has only its own trace too. No other trace is deleted.

`delete` prints the experiment name, item count and trace count, then asks for `yes` on stdin, which needs `-it`. It deletes only on `yes`; any other answer, or end of input, deletes nothing. `--yes` skips the question and deletes after printing the counts, so it works without `-it`. After deleting, it prints the experiment name and the number of items and traces it deleted.

A Langfuse error stops the delete (exit 5). After any interruption (a Langfuse error, a network error or Ctrl-C), running `delete` again deletes the rest.

`delete` needs only `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL`, which the Langfuse SDK reads (see [Secrets](#secrets)). It reads no other harness setting.

| Code | Meaning |
|-|-|
| 0 | The traces were deleted, the experiment has no items (nothing to delete), or the answer was not `yes` (nothing deleted) |
| 1 | The delete failed for any other reason |
| 2 | No experiment has that id; nothing was deleted |
| 5 | Langfuse returned an error |

## Settings

`scripts/run-eval.py` takes each value from its flag, else the shell, else `.env`, else its built-in default. A variable `NAME` has the flag `--<name>` and the VM metadata key `<name>`, where `<name>` is the variable's name in lowercase with dashes for underscores (`EVAL_IMAGE_TAG` is `--eval-image-tag` and `eval-image-tag`). There is one exception to that order: in the default mode, `EVAL_CONFIGS_COMMIT` comes from its flag only. Without the flag, the default mode resolves it from the configs ref (see [Questions](#questions)).

### Batch values

A batch is described by its batch values. They are:

- The answers: the values the default mode asks for, and `EVAL_CONFIGS_COMMIT`, which it does not ask for but resolves from the configs ref it asks for. The table in this section lists them.
- The non-secret settings `.env.example` sets: `LANGFUSE_BASE_URL`, `LANGFUSE_TRACING_ENVIRONMENT`, `EVAL_CONNECTION_KEYS`, `EVAL_DATASET` and `EVAL_ADMIN_EMAIL`. They are never asked.
- Ten optional harness settings, which `.env.example` leaves commented out: `EVAL_APP_START_TIMEOUT_SECONDS`, `EVAL_TURN_TIMEOUT_SECONDS`, `EVAL_POLL_INTERVAL_SECONDS`, `EVAL_TRACE_TIMEOUT_SECONDS`, `EVAL_TRACE_SETTLE_SECONDS`, `EVAL_MODEL_RETRIES`, `EVAL_RATE_LIMIT_RETRIES`, `EVAL_RATE_LIMIT_BACKOFF_SECONDS`, `EVAL_MODEL_MAX_TOKENS` and `EVAL_MAX_TURNS`. Each is a batch value only when it is set. They are never asked.

| Answer | Meaning | Built-in default |
|-|-|-|
| `EVAL_CONFIGS_REPO` | See [Harness settings](#harness-settings) | None |
| `EVAL_CONFIGS_COMMIT` | The commit SHA (see [Harness settings](#harness-settings)) | None |
| `EVAL_MODELS` | The model ids to run, comma-separated | None |
| `EVAL_JUDGE_MODELS` | See [Harness settings](#harness-settings) | None |
| `EVAL_SIMULATED_USER_MODEL` | See [Harness settings](#harness-settings) | None |
| `APP_IMAGE` | See [Harness settings](#harness-settings) | `ghcr.io/rhiza-research/weather-skills-chat:prod` |
| `RUNNER_IMAGE` | See [Harness settings](#harness-settings) | `ghcr.io/rhiza-research/weather-skills-sandbox-runner:prod` |
| `EVAL_CONCURRENCY` | See [Harness settings](#harness-settings) | Blank, for the harness's default |
| `EVAL_REPETITIONS` | How many rounds of the models the batch runs: a whole number of at least 1 | 1 |
| `EVAL_PARALLEL_RUNS` | How many runs the batch keeps going at the same time: a whole number of at least 1 | 1 |
| `EVAL_PULL_APP_IMAGE` | `true` or `false`: whether the batch pulls `APP_IMAGE` once, before the runs. With `false`, the batch ends when the app image is not on the Docker host, as on a new batch VM, whose disk starts from the Container-Optimized OS image | `true` |
| `EVAL_IMAGE_TAG` | The eval image tag the batch runs in | `main` |

The default mode also takes two values that are not batch values: `EVAL_CONFIGS_REF`, the configs ref, with the built-in default `main`, and `EVAL_LAUNCH`, the launch (`vm`, `docker` or `none`), with the built-in default `vm`.

An invalid flag value, such as `--eval-repetitions 0`, or a value of only whitespace where one is required, exits 2, naming the flag, before anything runs. Secrets are never flags (see [Secrets](#secrets)).

`EVAL_CONCURRENCY` sets how many tasks each run plays at the same time, so up to `EVAL_PARALLEL_RUNS` × `EVAL_CONCURRENCY` tasks are in flight at the same time.

### Secrets

Every run needs `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `OPENROUTER_API_KEY`, `APP_WEBUI_SECRET_KEY`, `APP_ADMIN_PASSWORD`, `APP_ADMIN_API_KEY`, and each variable `EVAL_CONNECTION_KEYS` names.

`scripts/run-eval.py` reads a secret from Secret Manager as the latest version of the secret `weather-skills-eval-<name>`, where `<name>` is the variable's name in lowercase with dashes for underscores (`APP_OPENROUTER_API_KEY` is `weather-skills-eval-app-openrouter-api-key`). It reads with the application default credentials. The project is `rhiza-shared`: it is fixed in the script and is not a setting.

Where each secret comes from depends on how the run starts:

- A launch on a new VM: the VM reads every secret from Secret Manager, through its service account (see [Before you start](#before-you-start)). Neither `gcloud` command of the launch reads a secret.
- A launch on the Docker host with `EVAL_LAUNCH` `docker`: the default mode takes each secret from the shell, else from `.env`. It reads each one left empty from Secret Manager; run `gcloud auth application-default login` first. A secret that cannot be read exits 1, naming it, before the batch starts. The default mode passes each secret to `docker run` by name only (`-e NAME`), with the value in the environment of `docker run`, so no secret value is on a command line.
- A printed `docker run` command that you run by hand: `docker run` takes each secret the command names from the shell it runs in. Background mode, in the container, reads each secret left empty from Secret Manager. That read works only on a batch VM: the container has no Google login elsewhere. So set every secret in that shell.
- A run started by hand (see [Debugging one run](#debugging-one-run)) and `delete`: the secrets come from `.env` or `-e` only, never from Secret Manager. Only `scripts/run-eval.py` reads Secret Manager.

### Harness settings

A run reads the harness settings from its environment. `scripts/run-eval.py` gives them to each run of a batch. The model, the skill packs, the connections and the tracked settings (`settings.yaml`) come from the configs repo (see [Configs repo](#configs-repo)). `.env.example` lists the harness settings, except `APP_IMAGE`, `RUNNER_IMAGE` and `EVAL_CONCURRENCY`, which it names only in a comment, and the ones `scripts/run-eval.py` sets per run. A missing or invalid harness setting stops the run before it starts (exit 2), except an `EVAL_MODEL` that is no model's `id`, which exits 1 (see [Configs repo](#configs-repo)).

Each entry says whether the setting is required or gives its default, and whether the default mode of `scripts/run-eval.py` asks for it.

Images:

- `APP_IMAGE`: the app image. The harness requires it; `scripts/run-eval.py` has a built-in default for it (see [Batch values](#batch-values)). Asked. A run pulls it before it starts unless `EVAL_PULL_APP_IMAGE` is `false`. A batch pulls it once, before the runs, and gives each run `EVAL_PULL_APP_IMAGE=false`. So each run of a batch logs `Image <image> found on the Docker host` and records `image_pulled` as `false`. The experiment records its digest: for a digest reference (`<repository>@sha256:...`), the digest it pins; otherwise the first digest the image lists for its repository. A pulled image that lists no digest for its repository stops the run (exit 3).
- `RUNNER_IMAGE`: the sandbox runner image skills run on, `ghcr.io/rhiza-research/weather-skills-sandbox-runner:<tag>`. The harness requires it; `scripts/run-eval.py` has a built-in default for it (see [Batch values](#batch-values)). Asked. The tag prod runs is `image.tag` in the infrastructure repository's `charts/weather-skills-sandbox/values.yaml`. The infrastructure repository's image workflow sets `:prod` from that pin when it runs on `main`, so `:prod` follows the pin after that workflow runs. `RUNNER_IMAGE` is a published image and is pulled before every run, whatever `EVAL_PULL_APP_IMAGE` says. The experiment records its digest the same way as for `APP_IMAGE`; a runner image that lists no digest for its repository stops the run (exit 3).
- `EVAL_PULL_APP_IMAGE`: `true` or `false`, for `APP_IMAGE` only. Default `true`. Asked ("Pull the app image?"). With `false` the app image is not pulled: the run uses the image already on the Docker host's daemon, such as one built there and never pushed, and stops when it is not there. With a remote context, the image must be built on that remote daemon. For a digest reference, the digest it pins is recorded; otherwise the first digest the image lists for its repository, or empty when it has none. The harness takes an unset or empty value as `true`, and every other value but `true` (in any casing) as `false`. Only `scripts/run-eval.py` refuses a value that is neither `true` nor `false`.

Dataset and app:

- `EVAL_DATASET`: the Langfuse dataset holding the tasks. Required. Not asked. The dataset's metadata may name the skill packs its tasks need as `skill_packs`, a list of pack names, such as `{"skill_packs": ["chc-skills"]}`. Without metadata, without `skill_packs`, or with an empty list, the dataset needs no pack. Metadata that is not an object, or a `skill_packs` that is not a list of non-empty names, stops the run. After reading the dataset and before starting the app, the harness checks that the commit's `SKILL_PACKS` turns on every pack the dataset names, and otherwise stops the run, naming each pack it does not turn on. A pack is turned on when its `skill-packs.yaml` entry has `enabled_by_default: true`; `is_active` is not part of that check. A pack is named by the last path segment of its `git_url`, without a trailing `.git` (`https://github.com/rhiza-research/chc-skills.git` is `chc-skills`). Every run of a dataset plays every item, so tasks that need a pack belong in a dataset of their own.
- `EVAL_ADMIN_EMAIL`: the email of the app's admin, which the harness signs in as. Required. Not asked.
- `LANGFUSE_BASE_URL`: the address of the Langfuse server. Required. Not asked. The harness gives it to the app.
- `LANGFUSE_TRACING_ENVIRONMENT`: the Langfuse environment of the simulated user's and judges' traces. Required. Not asked. It must not be `production`. Each simulated-user and judge call is a Langfuse trace with user id `simulated-user` or `judge`, session id = the chat id, in this environment.

Configs repo:

- `EVAL_CONFIGS_REPO`: the configs repo's URL. Required. Asked. A batch needs `https://github.com/<owner>/<repo>`.
- `EVAL_CONFIGS_COMMIT`: the commit SHA. Required. Not asked: the default mode resolves it from the configs ref it asks for, or takes it from the `--eval-configs-commit` flag.
- `EVAL_CONFIGS_DIR`: the folder that holds a copy of the configs repo at that commit. Required. Not asked: `scripts/run-eval.py` sets it to `/configs-repo`.
- `EVAL_MODEL`: the run's model id. Required. Not asked: `scripts/run-eval.py` sets it for each run from `EVAL_MODELS`.
- `EVAL_CONNECTION_KEYS`: the variable that holds each connection's key, by the connection's base URL: `<base URL>=<VARIABLE>`, comma-separated. Default empty in the harness, which stops a run whose commit has a connection it does not name (see [Configs repo](#configs-repo)). `scripts/run-eval.py` requires it. Not asked.

The harness records `EVAL_CONFIGS_REPO` and `EVAL_CONFIGS_COMMIT` as given, and never compares them with the files in `EVAL_CONFIGS_DIR`.

Simulated user and judges:

- `EVAL_SIMULATED_USER_MODEL`: the simulated user's model. Required. Asked.
- `EVAL_JUDGE_MODELS`: the judge panel: one or more models, comma-separated. Required. Asked, as a checklist. Spaces around a name are ignored. An empty name or a model named twice stops a run (exit 2), as in a run started by hand; `scripts/run-eval.py` drops empty and repeated names before it starts the runs.
- `EVAL_MODEL_MAX_TOKENS`: a whole number, at least 1. Default 16384. Not asked. Every simulated-user and judge call, including the answer-quality rating, sends `max_tokens` = `EVAL_MODEL_MAX_TOKENS` unless the call sets its own `max_tokens` or `max_completion_tokens`; a call that sets `max_completion_tokens` and no `max_tokens` is sent without `max_tokens`. Without a limit, OpenRouter reserves the model's whole output window against the account balance and refuses calls with HTTP 402 when the balance is low. A simulated-user reply cut off at that limit stops the play as `error`, naming the limit. A judge's verdict cut off at that limit makes that judge errored, naming the limit, and the play ends as `error` only when no judge ruled (see [The panel and its majority](#the-panel-and-its-majority)). A cut-off answer-quality rating does not stop the play, and `answer_quality` is recorded as not rated.
- `EVAL_MODEL_RETRIES`: a whole number, at least 0. Default 2. Not asked. A simulated-user, judge or answer-quality call, or the served-model request, that fails is retried by LiteLLM's `completion` (`num_retries`): every API error (any HTTP status, including 400, 402 and 429), timeout and connection error is retried at once, with no wait, up to `EVAL_MODEL_RETRIES` more times.
- `EVAL_RATE_LIMIT_BACKOFF_SECONDS`: the first wait before a rate-limit retry, described in the `EVAL_RATE_LIMIT_RETRIES` entry. A finite number greater than 0. Default 15. Not asked.
- `EVAL_RATE_LIMIT_RETRIES`: the most rate-limit retries of one call. A whole number, at least 0. Default 6. Not asked. A simulated-user, judge or answer-quality call, or the served-model request (see [Experiment name and served model](#experiment-name-and-served-model)), that still fails with a rate limit (LiteLLM's `RateLimitError`, HTTP 429) once LiteLLM's own retries are spent is made again after a wait, at most `EVAL_RATE_LIMIT_RETRIES` times. The first wait is `EVAL_RATE_LIMIT_BACKOFF_SECONDS` and it doubles for each retry, so the defaults wait 15, 30, 60, 120, 240 and 480 seconds. When the error's `Retry-After` header (seconds or an HTTP date, a date without a time zone taken as UTC) asks for longer, that is waited instead, but no wait is longer than the last one of that schedule (480 seconds with the defaults). Each rate-limit retry is itself one LiteLLM call with up to `EVAL_MODEL_RETRIES` immediate retries, so a rate-limited call is tried up to (`EVAL_RATE_LIMIT_RETRIES` + 1) × (`EVAL_MODEL_RETRIES` + 1) times. Each wait is logged on stderr (`<model> rate-limited (HTTP 429), retry <n> of <EVAL_RATE_LIMIT_RETRIES> in <seconds>s`). A call still rate-limited after the last retry fails as any failed call does, and `0` makes a rate limit fail at once. Other errors are not waited on. A traced call stays one generation in Langfuse however many attempts it took. The app's own model calls are not the harness's and are not retried this way.

Plays, app start, polls and traces:

- `EVAL_CONCURRENCY`: how many tasks a run plays at the same time. A whole number, at least 1. Default 1; empty or absent means the default. Asked.
- `EVAL_MAX_TURNS`: the turn cap: the most user messages a play sends, to stop a play where the assistant or the simulated user loops. A whole number, at least 1. Default 20. Not asked. A task's `turn_limit` overrides it. [Simulated user](#simulated-user) lists when a play ends.
- `EVAL_TURN_TIMEOUT_SECONDS`: how long the harness waits for each reply, from sending the message. A finite number greater than 0. Default 1800. Not asked. A reply not done by then stops the play as `chat_error`.
- `EVAL_APP_START_TIMEOUT_SECONDS`: how long the harness waits for the app to answer `/health`, and for the sandbox runner to be running steadily (see [Sandbox runner](#sandbox-runner)). A finite number greater than 0. Default 600. Not asked.
- `EVAL_POLL_INTERVAL_SECONDS`: the time between two polls of the app's `/health`, of the sandbox runner, of a chat for its reply, and of the app's Langfuse traces. A finite number greater than 0. Default 2. Not asked.
- `EVAL_TRACE_TIMEOUT_SECONDS`: how long the harness waits for a play's Langfuse traces (see [Trace-based scores](#trace-based-scores)). A finite number greater than 0. Default 600. Not asked.
- `EVAL_TRACE_SETTLE_SECONDS`: how long a play's traces must stay unchanged before they are read as complete. A finite number greater than 0. Default 60. Not asked. It must be less than `EVAL_TRACE_TIMEOUT_SECONDS`, else the run does not start (exit 2). A play whose traces are read as complete therefore waits at least this long for them.

Set by `scripts/run-eval.py` for each run of a batch, and optional for a run started by hand:

- `EVAL_BATCH`: the batch id, recorded as `batch`.
- `EVAL_APP_IMAGE_REVISION` and `EVAL_APP_IMAGE_PR_NUMBER`: the app image's labels, recorded as `app_image_revision` and `app_image_pr_number`.

### LangWatch

The harness never sends run data to LangWatch. Before Scenario starts, `run` removes every casing of `LANGWATCH_API_KEY` from the harness's environment, so Scenario neither posts its run events nor exports its spans to LangWatch.

### The app environment the harness sets

The harness sets these variables on the app from its settings: `WEBUI_SECRET_KEY` from `APP_WEBUI_SECRET_KEY`, `BOOTSTRAP_ADMIN_EMAIL` from `EVAL_ADMIN_EMAIL`, `BOOTSTRAP_ADMIN_PASSWORD` from `APP_ADMIN_PASSWORD`, `BOOTSTRAP_ADMIN_API_KEY` from `APP_ADMIN_API_KEY` (the harness signs in with it), and `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` as the harness's own, since the harness reads the app's traces with its own keys.

The harness adds the variables the configs repo defines (see [Configs repo](#configs-repo)). A `settings.yaml` key replaces a variable of the same name from the harness settings or from the other configs repo files. Last, the harness adds the sandbox runner variables (see [Sandbox runner](#sandbox-runner)), which replace both.

The harness calls the app on the port that `PORT` names in `settings.yaml`, else on 8080.

## How a task is played and judged

### Task format

One Langfuse dataset item is one task.

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

- `input` is the opening message, sent as the first turn. It must hold text other than whitespace. Item ids must be unique. The item id is the task's name on the progress lines and in the run summary.
- `deliverables` are the kinds of file the result must include, each `image` or `zarr`. Empty or absent means the task expects no file. An item needs at least one deliverable or judge criterion, and each judge criterion must hold text other than whitespace. Any other key in `expected_output` stops the run.
- `judge_criteria` are judged by the judge panel once the play has ended, as described under [Judging](#judging). The goal is reached when the simulated user is done and the panel passes every criterion. The panel passes by a majority of the judges that ruled.
- `facts` is what the simulated user answers from, and `hidden_steps` are steps it may reveal. See [Simulated user](#simulated-user). `facts` is required and must not be empty; unlike `input`, a `facts` of only whitespace is accepted.
- `pauses` wait before the given turn (turn 1 is the opening message, so the first possible pause is before turn 2). A pause before turn 1 or earlier, a `before_turn` that is not a whole number (`"3"` or `3.0` stops the run), a pause whose `seconds` is not a finite number greater than 0 (`true` stops the run), or two pauses before the same turn stop the run.
- `turn_limit` in `metadata` is optional: the task's turn cap, a whole number of at least 1 (`true`, `"3"` or `2.0` stops the run), in place of `EVAL_MAX_TURNS`. A pause past `turn_limit` stops the run. Without `turn_limit`, a pause past `EVAL_MAX_TURNS` stops the run when the run starts.
- `tags` are labels for grouping tasks in Langfuse, which shows them with the item's metadata. A `tags` that is not a list of strings stops the run. No play uses them, but they are part of the item's `metadata` in `comparability_hash` (see [Comparing runs](#comparing-runs)), so a changed tag changes the hash.
- Any other key in `metadata` stops the run.

### Presented files

A reply presents a file in one of three ways:

- The `path` of a successful `display_image` call, of an image or of an `.html` page.
- The `attachments` of a successful `send_email` call.
- A relative markdown image or link target in its text outside fenced code blocks (`![map](maps/rain.png)`, `[data](out.zarr)`).

A reply names a file in passing when two conditions hold. The reply's text holds the file's path in single backticks, outside fenced code blocks (`` `kenya_rain.zarr` ``). And the chat's file listing has an entry with exactly that path, and the entry is not a folder. A zarr store is listed as kind `zarr` and counts as a file. A file named in passing joins the result like a presented one, and counts as one of a reply's files; [What a judge sees](#what-a-judge-sees) says where it differs. Backtick text that is not a listed path, such as a variable name or a bare file name of a file in a folder (`` `rain.zarr` `` for `out/rain.zarr`), names nothing. Text inside tool calls and reasoning blocks names nothing.

A fenced code block starts at a line that begins, after any spaces or tabs, with three or more backticks or three or more tildes (`~~~`). It runs to a line that begins the same way with that fence, or to the end of the text. A backtick fence opens a block only when no backtick follows on that line, so an inline span such as ```` ```x``` ```` opens none.

A reply with no presented file and no file named in passing does not change which reply is the last one that presents files.

The result holds each path once, ordered by the turn each path is placed at. A path some reply presents outright is placed at the latest turn that presented it outright; a later mention in passing does not move it. Any other path is placed at the latest turn that named it.

### Simulated user

The simulated user's first message is the task's `input`. On every later turn it is shown:

- The transcript as the user sees it: each reply's text, without tool calls.
- The task's opening message, `facts` and `hidden_steps`. It never gets the judge criteria.
- When a reply has presented files, what a real user sees of the latest reply that presents files (see [Presented files](#presented-files)): each presented file's path and kind, and each image.
  - An image is attached when its media type is `image/png`, `image/jpeg`, `image/gif` or `image/webp`, and otherwise shown with the note `(not shown as an image)`.
  - A listed image that could not be downloaded is shown as `(could not be opened)`.
  - A presented path that is not in the chat's files, or every presented path when the listing fails, is shown with a note that it could not be opened.

The files are read once per reply that presents a file or holds text in single backticks. There is one exception: when a path is not listed yet, or the listing or a download failed, the listing and the failed downloads are read again on the next turn. Images already downloaded for the same presenting reply are reused by path, not downloaded again.

The simulated user never gets provenance or the harness's notes on why a result cannot be judged. The file list and the images are all it is shown of the files, so it checks the result as a user would ("I asked for the data file and only got a picture").

It is told:

- That the assistant's data files are Zarr stores, the standard format for this kind of data, and that a Zarr store is a data file.
- To compare what it received, the answer's text and any files, with what it asked for and its facts.
- To answer only what the assistant asked, from its facts, and to say it does not know or has no preference when its facts do not cover a question.
- To raise another fact only when a result lacks it or contradicts it.
- To check a plan the assistant proposes against what it asked for and its facts before agreeing, the same way it checks results.
- That it is done when the assistant has presented everything it asked for, as it asked.
- To ask for anything still wrong or missing before it is done or gives up, unless it already asked for that exact fix.
- To give up only when the assistant says it cannot do what was asked, or fails again at a step the simulated user already asked it to fix.

A hint is a hidden step the simulated user reveals. About hidden steps it is told:

- That the next hidden step is the first one it has not revealed and that is not done. A step the conversation or the files it was shown already show as done counts as used, and it skips to the next one.
- To mention the next hidden step, at most one per reply, in either of two cases: the assistant is stuck (it repeats itself, says it cannot go on, or asks how to proceed), or the assistant has presented a result, or moved on past where that step belongs, without doing that step.
- Not to mention it while the assistant is still working toward it.

A hint counts in `hints_used` only when its message is sent. A reply that is done or gives up never counts a hint.

Each simulated-user reply is a JSON object. It classifies the assistant's last reply as `clarifying_question`, `progress`, `answer` or `other`, and says whether the simulated user is `done` or gives up. Each simulated-user call sends `temperature` 0 and `response_format` `json_object`.

A reply cannot be used when it is not a JSON object, has an unknown classification, has a `done`, `give_up` or `revealed_hidden_step` that is not `true` or `false`, has `done` and `give_up` both true, has a `message` that is neither text nor null, or has an empty message while neither `done` nor `give_up` is true. The last rule does not apply to the read at the turn cap, whose message is never sent. After a reply that cannot be used, the harness makes the call once more, with that reply and a message saying why added to the call's messages. A second such reply stops the play as `error`.

A model call that fails is retried as `EVAL_MODEL_RETRIES` and `EVAL_RATE_LIMIT_RETRIES` say (see [Harness settings](#harness-settings)); when it still fails, the play stops as `error`. A call whose reply is cut off at `EVAL_MODEL_MAX_TOKENS` is not made again: the cut-off reply stops the play as `error`.

A play ends when the simulated user is done, gives up, or the turn cap (`turn_limit`, else `EVAL_MAX_TURNS`) passes, or when an error stops it as `chat_error` or `error` (see [Outcomes](#outcomes)). At the cap the simulated user reads the last reply once more, without anything more being sent to the app, so a play whose assistant finishes on the last allowed reply can still reach the goal. Only that read's done or give-up counts (with its reasoning); it records no classification and no hint.

### Judging

#### The panel and its majority

The judge panel rules once, after the play has ended (for any outcome but `chat_error` and `error`), on the final result. Each model in `EVAL_JUDGE_MODELS` rules through its own Scenario judge, at the same time as the others, on the same transcript and the same harness evidence, which is read once for the whole panel.

A judge whose call still fails after LiteLLM's retries and the rate-limit retries (see [Harness settings](#harness-settings)), or returns no verdict, is errored and left out. The panel passes when more than half of the judges that ruled pass; a tie fails. When no judge ruled, the play ends as `error`.

Each judge's `model`, `state` (`passed`, `failed` or `error`), `passed` (true, false, or null for an errored judge) and `reasoning` (for an errored judge, the error) are in the item output's `judges`. A judge pass never ends a play.

#### Refused verdict calls

When the provider refuses a judge's verdict call with a refusal Scenario's `provider_compat` recognizes, the call is sent once more adapted. That judge's later verdict calls in the same play are sent adapted. A refusal of the adapted call makes the judge errored. The answer-quality rating is never adapted: it sends `temperature` 0 and `response_format` `json_object` as they are.

#### What a judge sees

A judge sees the transcript as the user sees it: each reply's text, without tool calls. Its scenario description holds the task's opening message and `facts`. A judge never gets `hidden_steps`. Scenario's judge is told the conversation reached the turn cap only when the outcome is `turn_cap`.

The judge's scenario description (after the facts) and the answer-quality rating prompt (before the user's request) hold the UTC date and time the play started, given as harness context, not as something the user said (`Harness context, not said by the user: the conversation started on <YYYY-MM-DD> at <HH:MM> UTC.`). The simulated user is not given it.

A task with no `deliverables` is judged on the transcript and its `judge_criteria`.

When a task has `deliverables`, the result is every file the assistant presented during the play (see [Presented files](#presented-files)), never the other files of the chat, which hold intermediate results; the criteria decide which of them the user wanted at the end:

1. Each file's kind and provenance come from its entry in the chat's file listing. Every kind in `deliverables` must be among the presented files. Each PNG image, `.html` page and zarr store of the last reply that presents files must have provenance, except a file named only in passing. The app reads the `weather_skills_history` chain of those three kinds, and no other, into the entry's `provenance_detail`. A file of another kind without provenance, and a file named only in passing with any problem, are given to the judge with a note; a path named only in passing never blocks the judge. A file named only in passing counts as presented when its kind is in `deliverables` and no file of that kind was presented outright, however many such files there are; otherwise it stays named in passing.
2. Each presented image of type `image/png`, `image/jpeg`, `image/gif` or `image/webp` is downloaded once the play has ended, so a path presented again is judged on its latest version. A failed listing or download is read once more, after waiting `EVAL_POLL_INTERVAL_SECONDS`, before it counts as a problem. So is a listing that lacks a file of the last reply that presents files, or a backtick path of the last reply that names any file. When that extra listing fails, the earlier one is used, and when a file of the last reply that presents files is then missing from it the problem also names the listing error. Any other image is given to the judge without its picture, with the note `(not shown as an image)`, which does not block. Nothing else is downloaded.
3. The judge rules on the `judge_criteria` plus a fixed criterion: each presented file's provenance (data source, region, period, aggregation, units) matches the user's request and what was agreed in the chat; the criterion says a file named in passing and not presented is not held to it. Besides the transcript, its call holds one message of harness evidence: each file's path, kind and the turn that presented it (for a file named in passing, `named in passing on turn <n> and not presented`), its `provenance_detail` as JSON, and each image.

No result can be judged when the chat's files could not be listed on either read, no reply presented a file, a kind is missing among all presented files, or a file of the last reply that presents files is not in the listing, has no provenance where step 1 requires it, or cannot be downloaded. Then no judge model is called: every judge's verdict is failed, and its reasoning is the problem (see [Outcomes](#outcomes)). Backtick text that is not a listed path is never a file missing from the listing. Such a problem on a file presented earlier does not stop the judge: the file is given to it with a note naming the problem, next to the turn that presented it.

## How a batch and a run work

`scripts/run-eval.py` runs a batch of runs for the models you pick from one configs repo commit. It has two modes, which take the same flags, except `--eval-launch` and `--eval-configs-ref`, which only the default mode takes. The default mode, `uv run scripts/run-eval.py`, runs on your machine: it asks for each answer that has no flag, prints two ways to launch the batch, on a new VM or on the Docker host, and runs the launch that `EVAL_LAUNCH` names. Background mode, `--background`, asks nothing: it runs in the eval image `ghcr.io/rhiza-research/weather-skills-eval` and runs the batch on the Docker host whose socket it has. `uv run scripts/run-eval.py --help` lists the flags.

Once launched on a VM, a batch depends on nothing on your machine, which can close. A batch launched on the Docker host runs on that host, so your machine can close only when the Docker host is another machine.

### The default mode

A flag skips its question. Each answer without a flag is asked, except `EVAL_CONFIGS_COMMIT` (see [Questions](#questions)), prefilled from the shell, else `.env`, else its built-in default (see [Batch values](#batch-values)). The other batch values are never asked. A prefill from the shell or `.env` that its flag would refuse, such as `EVAL_PULL_APP_IMAGE=yes`, prints `<NAME> from the shell or .env: <why>` and exits 2 before asking anything. With every answer and `EVAL_LAUNCH` as a flag, nothing is asked. A `--eval-judge-models` flag is trimmed as `EVAL_JUDGE_MODELS` in `.env` is (spaces trimmed, empty names and repeats dropped); a flag that names no model prints `--eval-judge-models names no judge model` and exits 2.

#### Questions

1. When `EVAL_JUDGE_MODELS` has no flag and names no judge model in the shell or `.env`, the script prints `EVAL_JUDGE_MODELS names no judge model; set it in the shell or .env` and exits 2 before asking anything. When one of the non-secret settings `.env.example` sets (see [Batch values](#batch-values)) has no flag and is set in neither the shell nor `.env`, it prints `Not set by flag, in the shell or .env: <names>` and exits 2.
2. It asks for `EVAL_CONFIGS_REPO`, which must be `https://github.com/<owner>/<repo>`. Any other URL prints `Configs repo <url> is not https://github.com/<owner>/<repo>` and exits 2.
3. It asks for the ref (`EVAL_CONFIGS_REF`): a branch, a tag or a full commit SHA. It fetches that ref with `git fetch --depth 1` into a new repository in a temporary folder, with your machine's git access, checks it out there, and prints `Configs <repo> at <ref> is commit <SHA>`. A failed git command prints its error and exits 1. A `--eval-configs-commit` flag skips the ref question: the commit is used as is, and fetched the same way, so a commit the repo lacks exits 1 before the launch.
4. It offers the ids of the commit's model files as a checklist, each ticked when `EVAL_MODELS` in the shell or `.env` names it. An id in `EVAL_MODELS` that the commit lacks is not offered. Picking none exits 0. A `--eval-models` flag that names an id that is not one of the commit's prints `Not a model of commit <SHA>: <ids>` and exits 2. The script removes the temporary folder once it has asked its questions.
5. It offers the judge models that `EVAL_JUDGE_MODELS` names in the shell, else in `.env` (comma-separated; spaces trimmed, empty names and repeats dropped), as a checklist with each ticked. At least one must be picked.
6. It asks for `EVAL_SIMULATED_USER_MODEL`, `APP_IMAGE`, `RUNNER_IMAGE` and `EVAL_CONCURRENCY`. A blank simulated user model is not accepted.
7. It asks for the number of repetitions (`EVAL_REPETITIONS`) and "Runs at once:" (`EVAL_PARALLEL_RUNS`), each a whole number of at least 1.
8. It prints the batch's estimated cost (see [Cost estimate](#cost-estimate)), then asks "Pull the app image?" (`EVAL_PULL_APP_IMAGE`) and for `EVAL_IMAGE_TAG`.
9. It prints the commands of both launches, with every batch value, and asks last for `EVAL_LAUNCH`: "On a new VM" (`vm`), "On the current Docker context's host" (`docker`) or "Neither" (`none`). With `none` it only prints the commands. When every one of `;`, `|`, `#` and `~` appears in a batch value, it prints `No VM launch: every metadata delimiter ;|#~ is in a batch value` and exits 2 before it prints the commands, whatever the launch.

#### VM launch

The VM launch is two commands:

- `gcloud compute instances create weather-skills-eval-<UTC time> --zone us-central1-a --project rhiza-shared --source-instance-template weather-skills-eval`.
- `gcloud compute instances add-metadata weather-skills-eval-<UTC time> --zone us-central1-a --project rhiza-shared --metadata ^;^<key>=<value>;...`, with every batch value.

The create command has no `--metadata`, since that would replace the template's metadata, its cloud-init included; `add-metadata` adds to it. The metadata list is split on the first of `;`, `|`, `#` and `~` that no value holds (`gcloud topic escaping`).

With `vm`, the script runs the two commands. A failed command prints `Failed, exit <code>: <command>` after gcloud's own error, and the script exits with that code; when the first fails, the second is not run. When `add-metadata` fails, the VM already exists: the script prints `VM <VM> was created; delete it with:` and the delete command.

After both commands, the script prints the VM's name, `View its output with: gcloud compute instances get-serial-port-output <VM> --zone us-central1-a --project rhiza-shared` and `Stop it with: gcloud compute instances delete <VM> --zone us-central1-a --project rhiza-shared`.

#### Launch on the Docker host

The launch on the Docker host is `docker run --detach --rm --pull always --name weather-skills-eval-batch --stop-timeout 120 --volume /var/run/docker.sock:/var/run/docker.sock -e <secret>... ghcr.io/rhiza-research/weather-skills-eval:<EVAL_IMAGE_TAG> /opt/run-eval/bin/python /app/scripts/run-eval.py --background --<flag>=<value>...`. The socket is the Docker host's own, so no path from your machine is mounted and a remote Docker context works. With `docker`:

1. When the container `weather-skills-eval-batch` exists on the Docker host, the script prints `A batch is running: container weather-skills-eval-batch` and `Stop it with: docker stop weather-skills-eval-batch` and exits 1, starting nothing. One batch runs at a time on a Docker host. The script checks with `docker container inspect weather-skills-eval-batch`, and takes any failure of that command, an unreachable Docker daemon included, as no batch running.
2. It takes each secret as [Secrets](#secrets) describes.
3. It runs the `docker run` command. A failed `docker run` prints `Failed, exit <code>: docker run`, and the script exits with that code. On success it prints `Batch started in container weather-skills-eval-batch`, `Follow it with: docker logs -f weather-skills-eval-batch` and the stop command.

### Background mode

Background mode runs in the eval image, and removes the configs repo files and their tar when it ends. A launch on the Docker host puts every secret in the container's environment; a batch VM puts no secret there. Background mode asks nothing. It exits 2, naming the value, before anything runs, for a required batch value that has no flag, no shell or `.env` value and no built-in default, and for a shell or `.env` value that its flag would refuse. `EVAL_CONCURRENCY` may be empty, and so may each commented-out setting, which is then left out. `--eval-launch` and `--eval-configs-ref`, of the default mode only, exit 2, naming them. Background mode:

1. Takes each secret from its environment, and reads each one left empty from Secret Manager (see [Secrets](#secrets)). A secret that cannot be read ends the batch, naming it, before any run starts.
2. Downloads the files of `EVAL_CONFIGS_REPO`, which must be `https://github.com/<owner>/<repo>`, at `EVAL_CONFIGS_COMMIT` through GitHub's tarball API, without git or a token, so the repo must be public. A failed download ends the batch.
3. Pulls `APP_IMAGE` with `docker pull` when `EVAL_PULL_APP_IMAGE` is `true`, and reads the app image's labels `org.opencontainers.image.revision` and `weather-skills-chat.pr-number` (absent on an image not built for a PR) with `docker image inspect`. A failed command ends the batch.
4. Queues the models in rounds, each model once per round, as many rounds as the repetitions. It starts the runs in that order and keeps `EVAL_PARALLEL_RUNS` of them going: when one ends, the next starts.
5. Starts each run as `docker compose run --rm -T --entrypoint sh` of the eval service, in the eval image of `EVAL_IMAGE_TAG` (compose's `EVAL_IMAGE`), already on the Docker host, with the command `-c 'mkdir -p /configs-repo && tar -x -f - -C /configs-repo && exec weather-skills-eval run'`. Each run gets a tar of the configs repo files on stdin, so no host path is mounted. The tar is written once.
6. Gives each run, with `-e`, every batch value but `EVAL_MODELS`, `EVAL_REPETITIONS`, `EVAL_PARALLEL_RUNS` and `EVAL_IMAGE_TAG`. It sets `EVAL_PULL_APP_IMAGE=false`, `EVAL_BATCH`, `EVAL_CONFIGS_DIR=/configs-repo`, `EVAL_APP_IMAGE_REVISION` and `EVAL_APP_IMAGE_PR_NUMBER` (empty when the label is absent), and `EVAL_MODEL`, the run's model id. Each secret is passed by name only.
7. Prints each run's output and exit code (see [Following a batch](#following-a-batch)), and saves each run's agent cost (see [Cost estimate](#cost-estimate)).

SIGTERM stops the batch (see [Stopping a batch and cleanup](#stopping-a-batch-and-cleanup)).

### A batch VM

The infrastructure repository's instance template `weather-skills-eval` (Container-Optimized OS) runs a batch with no SSH session. The VM runs the batch in background mode, in the eval image of the tag in its metadata key `eval-image-tag`, as the container `weather-skills-eval-batch`, with each batch value in its metadata as a flag `--<key>=<value>`.

The VM deletes itself when the batch container exits for any reason, with any exit code. A VM whose eval image pull fails, such as for a tag that does not exist, deletes itself the same way. The template's maximum run duration, 24 hours, deletes a VM whose batch is still running then, or whose `eval-image-tag` is never set. The runs in flight at that time leave partly recorded experiments.

The batch's output is on the serial console while the VM lives. Several batch VMs can run at the same time: nothing in the template is shared between them.

The rest of the mechanism is in the infrastructure repository's `terraform/20-gke-cluster/weather-skills-eval.tf` and `terraform/20-gke-cluster/weather-skills-eval-cloud-init.yaml.tftpl`.

### The eval image

CI publishes `ghcr.io/rhiza-research/weather-skills-eval` on each push to `main`, tagged `main` and with the full commit SHA, which is also its `HARNESS_COMMIT`. It holds the harness, `compose.yaml`, the scripts, with `run-eval.py`'s dependencies in `/opt/run-eval`, and the Docker CLI and compose plugin.

### Configs repo

A run takes the app's definition from a copy of the configs repo (weather-skills-configs) at one commit, in `EVAL_CONFIGS_DIR` (see [Harness settings](#harness-settings)).

The harness reads `schemas/model.json`, `schemas/skill-packs.json`, `schemas/connections.json` and `schemas/settings.json`, and refuses to run (exit 2) unless each has `x-schema-version` 1 and each data file has `schema_version` 1. It checks nothing else about the files, so a data file that is not a mapping, or lacks a key the harness reads, fails the run (exit 1). A model file without an `eval` mapping fails every run at that commit this way, with `Run failed: KeyError: 'eval'`. Then:

- `CATALOG_MODELS` is the `model` of every file in `models/`, so every model at the commit is given to the app. `EVAL_MODEL` must be the `id` of one; the run plays that model. An `EVAL_MODEL` that is no model's `id` exits 1 with `Run failed: KeyError: '<id>'`.
- `SKILL_PACKS` is each pack's `git_url`, `ref` and `enabled_by_default` from `skill-packs.yaml`.
- `OPENAI_API_BASE_URLS` is the keys of `connections.yaml`, in file order, joined with `;`. `OPENAI_API_CONFIGS` maps each index, from `"0"`, to the connection's entry.
- `OPENAI_API_KEYS` is, in the same order, the value of the variable that `EVAL_CONNECTION_KEYS` names for each base URL. A connection with no variable named, or whose variable is unset or empty, stops the run (exit 2).
- Each key of `settings.yaml` is set as the app's variable of that name. A value that is not a string is set as its JSON text.

Once the app answers `/health`, the harness reads `GET /api/v1/skills/` as a platform admin (`X-Organization-Id: platform`), and matches each pack listed in `skill-packs.yaml` by `git_url` and each skill listed there by name. A listed pack or skill missing from the app stops the run (exit 4). Where a pack's or skill's `is_active` differs, the harness calls the pack's or skill's toggle route. That route flips the state, so the harness calls it only where the state differs. Where a skill's `enabled_by_default` differs, the harness posts the value to the skill's `enabled-by-default` route. A skill not listed in `skill-packs.yaml` is not checked.

### Sandbox runner

Skills run on a runner container, as in production and staging, not in the app container. Per run, the harness:

1. Reads the sandbox emulator from the app image under test: creates the container `weather-skills-eval-emulator-<run id>` from the app image without starting it, reads `/app/sandbox/emulator/emulator.py` and `/app/sandbox/emulator/local_runtime.py` with `get_archive`, and removes that container. An app image without them stops the run before the runner starts (exit 3), naming the image and the missing path.
2. Pulls `RUNNER_IMAGE`, whatever `EVAL_PULL_APP_IMAGE` says. A failed pull stops the run before the app starts (exit 3), naming the runner image.
3. Creates a Docker volume `weather-skills-eval-data-<run id>`, and creates its `artifacts` folder: creates the container `weather-skills-eval-volume-setup-<run id>` from the app image with the whole volume at the app's data directory (`/app/backend/data`), without starting it, copies the empty folder `artifacts` into `/app/backend/data` with `put_archive`, and removes that container. That container's mount is the volume's first, so Docker fills the volume from the app image's data directory, as the app's own mount would. A folder that cannot be copied stops the run before the runner starts (exit 3).
4. Creates the runner container `weather-skills-eval-runner-<run id>` from `RUNNER_IMAGE`, as user `0`, with only the volume's `artifacts` folder mounted, at `/data/artifacts` (a volume mount with the subpath `artifacts`), and the command `python3 -u /opt/sandbox/emulator/emulator.py`. The runner sees the chat files and nothing else of the app's data directory, such as its database or uploads, as on the cluster. The subpath mount needs Docker Engine API 1.45, docker-py's default API version. Its environment is `ARTIFACTS_DIR=/data/artifacts`, `SKILL_K8S_EMULATOR_HOST=<runner name>`, `SKILL_K8S_EMULATOR_API_PORT=18080`, `SKILL_K8S_BIND_PORT=9090`, `SKILL_K8S_EMULATOR_START_DELAY=0` and `SKILL_K8S_BIND_TOKEN`, a random value generated for the run.
5. Copies the two emulator files into the runner at `/opt/sandbox/emulator` with `put_archive`, then starts it. No file reaches a container through a bind mount, so a remote Docker context works.
6. Starts the app with the volume at its data directory (`/app/backend/data`) and, added to its environment, `SKILL_SANDLOCK=false`, `SKILL_SANDBOX_BACKEND=k8s`, `SKILL_K8S_SANDBOX_ENABLED=true`, `SKILL_K8S_EMULATOR=true`, `SKILL_K8S_APISERVER=http://<runner name>:18080` and the same `SKILL_K8S_BIND_TOKEN`.

The emulator serves the claim, bind and exec APIs the app's `k8s` sandbox backend calls on the cluster, and runs each chat's skills with the runner image's `sandbox_runtime.py` in the chat's folder under `/data/artifacts`, which is the app's artifacts folder on the shared volume. The runner is created with the Docker restart policy `on-failure`, so a runner that dies is restarted, as the cluster replaces a dead runner pod. While the harness waits for the app's `/health`, a runner that exits or is restarting stops the run (exit 3) with the runner's logs.

The runner is running steadily when it is running on two polls in a row with the same Docker restart count. Before each task is played, the harness polls the runner every `EVAL_POLL_INTERVAL_SECONDS` until it is running steadily, logging that it is waiting when it finds it not running, and then plays the task. A runner not running steadily within `EVAL_APP_START_TIMEOUT_SECONDS` stops the run (exit 3) with the runner's logs; it is polled at least twice, so a start timeout shorter than the poll interval does not fail a running runner. A runner that exited with code 0 or is `dead`, which Docker does not restart, stops the run at once with its logs. A runner that has disappeared, or that Docker cannot read, stops the run at once without logs.

After each task, the harness checks the runner again. When the runner is not running, or has restarted since the task began, that task is not scored, so the runner's failure does not count as an agent failure. Its item records the runner error, with the runner's logs, and no scores, and its item trace still carries its chat id as its session id. A task is lost when it is not scored this way and Docker restarts the runner, so that the run goes on. What happens next depends on the runner's state:

- Running again with a new restart count, `restarting`, or exited with a non-zero code: Docker restarts it, so the task is lost and the run goes on. The next task waits until the runner is running steadily again, and is then played. The app makes a new claim on the restarted runner on its next skill run. A loss needs no wait at the end when a task started waiting for the runner after it and found the runner running steadily. When no such task follows the latest loss, as when the loss was on the last tasks, the harness waits for the runner the same way once all tasks have finished. A runner not running steadily again within `EVAL_APP_START_TIMEOUT_SECONDS`, before the next task or after the last one, stops the run (exit 3) with its logs.
- Exited with code 0, `dead` or in any other state, disappeared, or unreadable: the run stops at once (exit 3), with the logs when the runner can still be read. No further task is played.

A runner that Docker keeps restarting during tasks (a crash loop) does not stop the run by itself. Each task that is playing when the runner restarts is lost. The run stops (exit 3) only when the runner is not running steadily again in time, before the next task or after the last one. A run that finishes exits 0 however many tasks were lost, and its summary lists them (see [Run summary](#run-summary)). With `EVAL_CONCURRENCY` above 1, when the run stops, tasks already playing finish and are checked after the play, and a task still waiting for the runner is not played.

## Development

Tests and lint, in a container without network access:

```sh
docker compose run --rm --build test
```

The tests of `scripts/run-eval.py` are in the script and run on the host:

```sh
uv run --isolated --no-project --with-requirements scripts/run-eval.py --with pytest pytest scripts/run-eval.py
```

Install the pre-commit hooks, which run `ruff-format`, `ruff-check --fix` and `gitleaks`:

```sh
uv run pre-commit install
```
