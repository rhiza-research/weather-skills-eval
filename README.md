# weather-skills-eval

Measures what a change to weather-skills-chat (model, prompt, code, skills, settings) does to success, cost and behavior on real tasks.

`weather-skills-eval run <config>` starts a throwaway app container from a configuration file, plays every task of a Langfuse dataset through the app's chat API with a simulated user and a judge ([LangWatch Scenario](https://github.com/langwatch/scenario)), scores each play, and records the run as one Langfuse experiment, named after the model that answered. Two runs on the same dataset are compared side by side in Langfuse.

The harness talks to the app only over HTTP and Docker. Fetches use live data.

The harness never sends run data to LangWatch. Before Scenario starts, `run` removes every casing of `LANGWATCH_API_KEY` from the harness's environment, so Scenario neither posts its run events nor exports its spans to LangWatch. A configuration still resolves `${LANGWATCH_API_KEY}` from the harness's environment.

## Running

Everything runs in Docker.

```sh
cp .env.example .env        # fill in the harness settings and the values your config references
cp configs/example.yaml configs/<name>.yaml   # replace every TODO
HARNESS_COMMIT=$(git rev-parse HEAD) docker compose build eval
docker compose run --rm -T -e APP_IMAGE=<image> -e RUNNER_IMAGE=<image> [-e EVAL_CONCURRENCY=<n>] eval run - < configs/<name>.yaml
```

The configs set `pull: false`, so run `docker pull <APP_IMAGE>` first when running by hand.

`APP_IMAGE`, `RUNNER_IMAGE` and `EVAL_CONCURRENCY` are chosen per run with `-e`, as is `EVAL_BATCH`, optional. An empty or absent `EVAL_CONCURRENCY` means the harness's default.

`uv run scripts/run-eval.py` does the same for the configs you pick:

1. It lists `configs/*.yaml` other than `example.yaml` as a checklist.
2. It asks for `APP_IMAGE`, `RUNNER_IMAGE` and `EVAL_CONCURRENCY`, each prefilled from the shell, else from `.env`, else from its default: `ghcr.io/rhiza-research/weather-skills-chat:prod`, `ghcr.io/rhiza-research/weather-skills-sandbox-runner:prod`, and blank for the concurrency.
3. It asks for the number of repetitions, a whole number of at least 1, default 1.
4. It asks "Pull the app image?", default yes.
5. It builds the eval image once with `HARNESS_COMMIT` set to `git rev-parse HEAD`, then, on yes, runs `docker pull <APP_IMAGE>`. A failed build or pull exits with its exit code.
6. It prints the batch id, the UTC time it reached this step (`Batch 2026-10-22T20:07:57Z`).
7. It runs the picked configs in rounds, each config once per round, as many rounds as the repetitions, with the values and `EVAL_BATCH=<batch id>` passed as `-e`. It prints each run's exit code (`<config> run <round>: exit <code>`) and exits 1 if any failed.

Ctrl-C prints the exit codes of the runs that finished and exits 130.

Each run records the batch id in its experiment's metadata as `batch`, so filtering Langfuse experiments on `metadata.batch` shows the runs of one invocation.

- `run -` reads the configuration from stdin, so it works with a remote Docker context, where a bind mount from this machine would not reach the daemon. A configuration read from stdin must set `name`. `run <path>` reads a file inside the container instead.
- `HARNESS_COMMIT` is baked into the image and recorded in each experiment's metadata; without it the metadata says `unrecorded`.
- The eval service reaches the Docker daemon of the current Docker context through its socket. It pulls the configured app image (or, with `pull: false`, uses the image already on that daemon) and the runner image, starts a runner container and then the app container on the `weather-skills-eval` network (see [Sandbox runner](#sandbox-runner)), and removes both containers and the run's volume when the run ends, also when the run fails or the harness receives SIGTERM. Each container and volume is labeled with its run id (`weather-skills-eval.run`), and a run only touches its own, so several runs can share one Docker daemon. A harness killed with SIGKILL cannot clean up: remove its containers and volume by that label by hand.

### Sandbox runner

Skills run on a runner container, as in production and staging, not in the app container. Per run, the harness:

1. Reads the sandbox emulator from the app image under test: creates the container `weather-skills-eval-emulator-<run id>` from the app image without starting it, reads `/app/sandbox/emulator/emulator.py` and `/app/sandbox/emulator/local_runtime.py` with `get_archive`, and removes that container. An app image without them stops the run before the runner starts (exit 3), naming the image and the missing path.
2. Pulls `runner_image`, whatever `pull` says. A failed pull stops the run before the app starts (exit 3), naming the runner image.
3. Creates a Docker volume `weather-skills-eval-data-<run id>`, and creates its `artifacts` folder: creates the container `weather-skills-eval-volume-setup-<run id>` from the app image with the whole volume at the app's data directory (`/app/backend/data`), without starting it, copies the empty folder `artifacts` into `/app/backend/data` with `put_archive`, and removes that container. Being the volume's first mount, it fills the volume from the app image's data directory, as the app's own mount would. A folder that cannot be copied stops the run before the runner starts (exit 3).
4. Creates the runner container `weather-skills-eval-runner-<run id>` from `runner_image`, as user `0`, with only the volume's `artifacts` folder mounted, at `/data/artifacts` (a volume mount with the subpath `artifacts`), and the command `python3 -u /opt/sandbox/emulator/emulator.py`. The runner sees the chat files and nothing else of the app's data directory, such as its database or uploads, as on the cluster. The subpath mount needs Docker Engine API 1.45, docker-py's default API version. Its environment is `ARTIFACTS_DIR=/data/artifacts`, `SKILL_K8S_EMULATOR_HOST=<runner name>`, `SKILL_K8S_EMULATOR_API_PORT=18080`, `SKILL_K8S_BIND_PORT=9090`, `SKILL_K8S_EMULATOR_START_DELAY=0` and `SKILL_K8S_BIND_TOKEN`, a random value generated for the run.
5. Copies the two emulator files into the runner at `/opt/sandbox/emulator` with `put_archive`, then starts it. No file reaches a container through a bind mount, so a remote Docker context works.
6. Starts the app with the volume at its data directory (`/app/backend/data`) and, added to `env`, `SKILL_SANDLOCK=false`, `SKILL_SANDBOX_BACKEND=k8s`, `SKILL_K8S_SANDBOX_ENABLED=true`, `SKILL_K8S_EMULATOR=true`, `SKILL_K8S_APISERVER=http://<runner name>:18080` and the same `SKILL_K8S_BIND_TOKEN`.

The emulator serves the claim, bind and exec APIs the app's `k8s` sandbox backend calls on the cluster, and runs each chat's skills with the runner image's `sandbox_runtime.py` in the chat's folder under `/data/artifacts`, which is the app's artifacts folder on the shared volume. The runner is created with the Docker restart policy `on-failure`, so a runner that dies is restarted, as the cluster replaces a dead runner pod. While the harness waits for the app's `/health`, a runner that exits or is restarting stops the run (exit 3) with the runner's logs. Before each task is played, the harness polls the runner every `EVAL_POLL_INTERVAL_SECONDS` until it is running on two polls in a row with the same Docker restart count, logging that it is waiting when it finds it not running, and then plays the task. A runner not running steadily within `EVAL_APP_START_TIMEOUT_SECONDS` stops the run (exit 3) with the runner's logs; it is polled at least twice, so a start timeout shorter than the poll interval does not fail a running runner. One that exited with code 0 or is `dead`, which Docker does not restart, stops the run at once with its logs; one that has disappeared, or that Docker cannot read, stops it at once without logs. After each task, the harness checks the runner again. When it is not running, or has restarted since the task began, that task is not scored, so it is not scored as an agent failure: its item records the runner error, with the runner's logs, and no scores, and its item trace still carries its chat id as its session id. What happens next depends on the runner's state:

- Running again with a new restart count, `restarting`, or exited with a non-zero code: Docker restarts it, so the task is counted as lost and the run goes on. The next task waits for the runner as above and is played once it is running steadily again; the app makes a new claim on the restarted runner on its next skill run. When no task started waiting for the runner after the latest such loss and found it running steadily, as when the loss was on the last tasks, the harness waits for the runner the same way once all tasks have finished; a loss followed by a task that waited and played needs no wait at the end. A runner not back steadily within `EVAL_APP_START_TIMEOUT_SECONDS`, before the next task or after the last one, stops the run (exit 3) with its logs.
- Exited with code 0, `dead` or in any other state, disappeared, or unreadable: the run stops at once (exit 3), with the logs when the runner can still be read. No further task is played.

A runner Docker keeps restarting during tasks (a crash loop) does not stop the run by itself: each task it restarts during is counted as lost, and the run exits 0 unless the runner is not back steadily before the next task or after the last one. A run that finishes exits 0 however many tasks were lost this way, and its summary on stdout lists them on their own line (`<n> tasks not scored because the sandbox runner restarted during them: <task ids>`). Langfuse leaves a task that raised out of the experiment's results, so lost tasks are not counted in `Goal reached on <x> of <y> tasks`: `<y>` is the number of tasks that returned a result. With `EVAL_CONCURRENCY` above 1, when the run stops, tasks already playing finish and are checked after the play, and a task still waiting for the runner is not played. An exit 3 after tasks started leaves a partly recorded experiment in Langfuse, which `delete` removes (see [Deleting a run](#deleting-a-run)).

### Exit codes

| Code | Meaning |
|-|-|
| 0 | The run finished and the experiment was recorded, also when tasks were not scored because the sandbox runner restarted during them (see [Sandbox runner](#sandbox-runner)) |
| 1 | The run failed for any other reason |
| 2 | The run did not start: a harness setting, the configuration file, the dataset or a dataset item is invalid, the dataset is missing, empty or could not be read from Langfuse (any error, a Langfuse error included), or the dataset needs a skill pack the configuration does not turn on |
| 3 | The app or the sandbox runner did not start (an app image without the sandbox emulator, image pull, a pulled image with no digest for its repository, image not on the Docker host, container start or `/health`), the runner was not running steadily by `EVAL_APP_START_TIMEOUT_SECONDS` before a task, or after the last task when no task waited for it after it was lost, the runner was found exited with code 0, `dead`, disappeared or unreadable before or after a task, or Docker failed |
| 4 | The app failed while the harness read its catalog model, tools and skill packs, or reported no base model for the catalog model, or, after the dataset was read, a network error (an `httpx` error, such as a connection error or timeout) reaching the app or Langfuse stopped the run |
| 5 | Langfuse returned an error response after the dataset was read |
| 143 | The harness was stopped with SIGTERM; the app and runner containers and the volume were removed |

### Deleting a run

```sh
docker compose run --rm -it eval delete <experiment-id>
docker compose run --rm eval delete --yes <experiment-id>
```

The experiment id is the `baseline=` value in the experiment's Langfuse URL. Langfuse has no call that deletes an experiment; the experiment goes away once its traces are deleted. `delete` lists the experiment's items and finds each item's trace plus every trace in the session whose id is the item output's `chat_id`: the app's `weather-skills-chat` traces, the `simulated-user` and `judge` traces, and the item trace. For an item whose output has no chat id, it reads the item trace's observations and takes their session ids as the item's chats: the item trace carries the chat when the play returned and the runner check after it failed. A play that itself raised links nothing, so its item has only its own trace, as does any other item without a chat. No other trace is deleted.

It prints the experiment name, item count and trace count, then asks for `yes` on stdin, which needs `-it`. It deletes only on `yes`; any other answer, or end of input, deletes nothing. `--yes` skips the question and deletes after printing the counts, so it works without `-it`. After deleting, it prints the experiment name and the number of items and traces it deleted.

It sends the trace ids to Langfuse in batches of at most 1,000, the most one request takes ([Langfuse FAQ](https://langfuse.com/faq/all/deprecated-api-migration.md)), in order, with the item traces last, in requests of their own: no request holds an item trace and another trace. A Langfuse error on a batch stops the delete (exit 5); the batches sent before it stay deleted, and the failing batch may be partly deleted, as when a 5xx or timeout comes back after Langfuse deleted some of it. Because the item traces go last, after any interruption (a Langfuse error, a network error or Ctrl-C) the item traces still left keep the experiment listed, and running `delete` again deletes the rest.

It needs only `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL`, read by the Langfuse SDK, not the run's settings.

| Code | Meaning |
|-|-|
| 0 | The traces were deleted, the experiment has no items (nothing to delete), or the answer was not `yes` (nothing deleted) |
| 1 | The delete failed for any other reason |
| 2 | No experiment has that id; nothing was deleted |
| 5 | Langfuse returned an error |

## Configuration file

`configs/example.yaml` is a complete example. Its keys:

- `name`: the configuration's name, recorded in the experiment's metadata as `config_name`; defaults to the file name without `.yaml`, and is required when the configuration is read from stdin. The experiment itself is named after the model that answered (see [Scores](#scores)).
- `image`: the app image. It may hold `${NAME}` references, such as `${APP_IMAGE}` or `ghcr.io/rhiza-research/weather-skills-chat:${APP_TAG}`, so the image under test can change per run without editing the file. A variable `image` references reaches the harness from the eval service's `.env` or from `docker compose run -e NAME=value`; a variable exported in the host shell is not passed into the container. The experiment metadata records the resolved image, not the `${...}` text. Unless `pull` is `false`, it is pulled before every run, and the experiment records its digest: for a digest reference (`<repository>@sha256:...`), the digest it pins; otherwise the first digest the image lists for its repository. A pulled image that lists no digest for its repository stops the run (exit 3).
- `runner_image`: the sandbox runner image skills run on, `ghcr.io/rhiza-research/weather-skills-sandbox-runner:<tag>`; prod's tag is in the infrastructure repository's `charts/weather-skills-sandbox/values.yaml`. It is versioned separately from the app image, so the configurations set it as `${RUNNER_IMAGE}`, chosen per run like `${APP_IMAGE}`; it may hold `${NAME}` references like `image`. It is a published image and is pulled before every run, whatever `pull` says, and the experiment records its digest the same way as for `image`; a runner image that lists no digest for its repository stops the run (exit 3). Required.
- `pull`: `true` (the default) or `false`, for `image` only. With `false` the app image is not pulled: the run uses the image already on the daemon of the current Docker context, such as one built there and never pushed, and stops when it is not there. With a remote context, the image must be built on that remote daemon. For a digest reference, the digest it pins is recorded; otherwise the first digest the image lists for its repository, or empty when it has none.
- `model`: id of the catalog model the chats use (an `id` in `CATALOG_MODELS`).
- `dataset`: the Langfuse dataset holding the tasks. The dataset's metadata may name the skill packs its tasks need as `skill_packs`, a list of pack names, such as `{"skill_packs": ["chc-skills"]}`. Without metadata, without `skill_packs`, or with an empty list, the dataset needs no pack. Metadata that is not an object, or a `skill_packs` that is not a list of non-empty names, stops the run. After reading the dataset and before starting the app, the harness checks that `SKILL_PACKS` in `env` turns on every pack the dataset names, and otherwise stops the run, naming each pack the configuration does not turn on. Every run of a dataset plays every item, so tasks that need a pack belong in a dataset of their own.
- `env`: the app's environment; the harness adds only the sandbox runner variables (see [Sandbox runner](#sandbox-runner)).

Rules:

- Any other top-level key stops the run.
- `env` uses the app's variable names (see the app wiki pages Deployment, Skill Packs and Tracing). `WEBUI_SECRET_KEY` and the three `BOOTSTRAP_ADMIN_*` variables are required: the harness signs in with `BOOTSTRAP_ADMIN_API_KEY`. `SKILL_SANDLOCK`, `SKILL_SANDBOX_BACKEND`, `SKILL_K8S_SANDBOX_ENABLED`, `SKILL_K8S_EMULATOR`, `SKILL_K8S_APISERVER` and `SKILL_K8S_BIND_TOKEN` are set by the harness; setting one in `env` stops the run.
- `${NAME}` in `image`, `runner_image` or `env` is replaced with the harness's environment variable `NAME`; a reference to an unset or empty variable stops the run.
- A variable, or a key inside a mapping value, whose name has the word key, secret, password or token in it (singular or plural; `tokens` alone, as in `max_tokens`, does not count) must be exactly one `${NAME}` reference. A literal value stops the run. Inside a mapping, such a key may also be null, which carries no secret: an app connection entry can have an empty `key`.
- Each top-level value in `env` must be a string, a mapping or a list. Mappings and lists are written to the variable as JSON, after their references are replaced. Any other top-level value, such as a number, a boolean or null, stops the run, so numbers and booleans are quoted. Values inside a mapping or list are not limited to these types.
- The app listens on `PORT` from `env`, 8080 when unset. A `PORT` that is not a whole number from 1 to 65535 stops the run.
- `OPENAI_API_CONFIGS`, when set, must map each connection to an object. In each object, `enable`, when present, must be a boolean, `prefix_id` a string, and `model_ids` a list of strings; `prefix_id` and `model_ids` may also be absent or any empty or false value, such as null, `""`, `[]` or `false`, which counts as none. The harness reads these three keys to name the base model (see [Scores](#scores)).
- `SKILL_PACKS`, when set, must be a list of objects, each with a non-empty string `git_url` and, when present, a boolean `enabled_by_default`. A pack is named by the last path segment of its `git_url`, ignoring surrounding whitespace and trailing slashes, without a trailing `.git` (`https://github.com/rhiza-research/chc-skills.git` is `chc-skills`), and is on when its `enabled_by_default` is `true` or absent. The harness reads these to check the dataset's `skill_packs`.
- The app and the harness must use the same Langfuse project: the harness reads the app's traces with its own keys. The example passes the harness's Langfuse keys and address to the app.

## Harness settings

`.env.example` lists them. `EVAL_CONCURRENCY` is chosen per run (see [Running](#running)). The simulated user and judge models are harness settings so that they stay fixed across the runs being compared. `EVAL_SIMULATED_USER_MODEL` must accept image input: the simulated user is shown the presented images. `EVAL_JUDGE_MODEL` must accept image input when a task has an `image` deliverable: the judge's verdict call holds the presented images (see [Judging](#judging)). Each of their calls is a Langfuse trace with user id `simulated-user` or `judge`, session id = the chat id, in the environment set by `LANGFUSE_TRACING_ENVIRONMENT`, which must not be `production`.

Every simulated-user and judge call, including the answer-quality rating, sends `max_tokens` = `EVAL_MODEL_MAX_TOKENS` (a whole number, at least 1, default 8192) unless the call sets its own `max_tokens` or `max_completion_tokens`; a call that sets `max_completion_tokens` and no `max_tokens` is sent without `max_tokens`. Without a limit, OpenRouter reserves the model's whole output window against the account balance and refuses calls with HTTP 402 when the balance is low. A simulated-user or judge reply cut off at that limit stops the play as `error`, naming the limit; a cut-off answer-quality rating does not stop the play, and `answer_quality` is recorded as not rated.

`EVAL_MAX_TURNS` (a whole number, at least 1, default 20) is the turn cap: the most user messages a play sends, to stop a play where the assistant or the simulated user loops. A task's `turn_limit` overrides it. [Simulated user](#simulated-user) lists when a play ends.

`EVAL_TURN_TIMEOUT_SECONDS` (a finite number greater than 0, default 1800) is how long the harness waits for each reply, from sending the message; a reply not done by then stops the play as `chat_error`.

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

- `input` is the opening message, sent as the first turn. It must hold text other than whitespace. Item ids must be unique.
- `deliverables` are the kinds of file the result must include, each `image` or `zarr`. Empty or absent means the task expects no file. An item needs at least one deliverable or judge criterion, and each judge criterion must hold text other than whitespace. Any other key in `expected_output` stops the run.
- `judge_criteria` are judged by Scenario's judge once the play has ended, as described under [Judging](#judging). The goal is reached when the simulated user is done and the judge passes every criterion.
- `facts` is what the simulated user answers from, and `hidden_steps` are steps it may reveal. See [Simulated user](#simulated-user). `facts` is required and must not be empty; unlike `input`, a `facts` of only whitespace is accepted.
- `pauses` wait before the given turn (turn 1 is the opening message, so the first possible pause is before turn 2). A pause before turn 1 or earlier, a `before_turn` that is not a whole number (`"3"` or `3.0` stops the run), a pause whose `seconds` is not a finite number greater than 0 (`true` stops the run), or two pauses before the same turn stop the run.
- `turn_limit` in `metadata` is optional: the task's turn cap, a whole number of at least 1 (`true`, `"3"` or `2.0` stops the run), in place of `EVAL_MAX_TURNS`. A pause past it stops the run; without it, a pause past `EVAL_MAX_TURNS` stops the run when it starts.
- `tags` are labels for grouping tasks in Langfuse. The harness does not read them; Langfuse shows them with the item's metadata.
- Any other key in `metadata` stops the run.

## Simulated user

The simulated user's first message is the task's `input`. On every later turn it gets the transcript as the user sees it (each reply's text, without tool calls) and, when a reply has presented files, what a real user sees of the latest reply that presents files (see [Judging](#judging) for how replies present files): each presented file's path and kind, and each image, attached when its media type is `image/png`, `image/jpeg`, `image/gif` or `image/webp` and otherwise shown with the note `(not shown as an image)`; a listed image that could not be downloaded is shown as `(could not be opened)`; a presented path that is not in the chat's files, or every presented path when the listing fails, is shown with a note that it could not be opened. When a path is not listed yet, or the listing or a download failed, the listing and the failed downloads are read again on the next turn; images already downloaded for the same presenting reply are reused by path, not downloaded again. It never gets provenance or the harness's notes on why a result cannot be judged, so it judges what it received from the file list and the images, as a user would ("I asked for the data file and only got a picture"). It compares what it received, the answer's text and any files, with what it asked for and its facts. A reply that is done or gives up never counts a hint. The files are read once per reply that names a file in any of those ways, backtick text included, except as above. It also gets the task's `facts` and `hidden_steps`, never the judge criteria. It is told to answer only what the assistant asked, and to raise another fact only when a result lacks it or contradicts it, and to check a plan the assistant proposes against what it asked for and its facts before agreeing, the same way it checks results. It is told that the next hidden step is the first one it has not revealed and that is not done; a step the conversation or the files it was shown already show as done counts as used, and it skips to the next one. It is told to mention the next hidden step, at most one per reply, in either case: the assistant is stuck (it repeats itself, says it cannot go on, or asks how to proceed), or the assistant has presented a result, or moved on past where that step belongs, without doing that step. It is told not to mention it while the assistant is still working toward it. A hint counts in `hints_used` only when its message is sent.

Each of its replies says whether it is `done` or gives up. A reply that is not a JSON object, has an unknown classification, has a `done`, `give_up` or `revealed_hidden_step` that is not `true` or `false`, has `done` and `give_up` both true, has a `message` that is neither text nor null, or has an empty message while neither is true (except in the read at the turn cap, whose message is never sent) cannot be used: the call is made once more, shown that reply and a message saying why, and a second such reply stops the play as `error`. A model call that fails is retried by LiteLLM's `completion` (`num_retries`; checked on 1.105.1, the floor in `pyproject.toml`): every API error (any HTTP status, including 400, 402 and 429), timeout and connection error is retried at once, with no wait, up to `EVAL_MODEL_RETRIES` more times; when it still fails, the play stops as `error`. A reply cut off at `EVAL_MODEL_MAX_TOKENS` is not made again: it stops the play as `error`. An unusable reply is made once more, as above.

A play ends when the simulated user is done, gives up, or the turn cap (`turn_limit`, else `EVAL_MAX_TURNS`) passes, or when an error stops it as `chat_error` or `error` (see [Outcomes](#outcomes)). At the cap the simulated user reads the last reply once more, without anything more being sent to the app, so a play whose assistant finishes on the last allowed reply can still reach the goal. Only that read's done or give-up counts (with its reasoning); it records no classification and no hint.

## Judging

The judge rules once, after the play has ended (for any outcome but `chat_error` and `error`), on the final result. A judge pass never ends a play. Scenario's judge is told the conversation reached the turn cap only when the outcome is `turn_cap`. The judge sees the transcript as the user sees it: each reply's text, without tool calls. The judge's scenario description (after the facts) and the answer-quality rating prompt (before the user's request) hold the UTC date and time the play started, given as harness context, not as something the user said (`Harness context, not said by the user: the conversation started on <YYYY-MM-DD> at <HH:MM> UTC.`); the simulated user is not given it.

A task with no `deliverables` is judged on the transcript and its `judge_criteria`.

When a task has `deliverables`, the result is every file the assistant presented during the play, never the other files of the chat, which hold intermediate results; the criteria decide which of them the user wanted at the end:

1. A reply presents the `path` of a successful `display_image` call, of an image or of an `.html` page; the `attachments` of a successful `send_email` call; and the relative markdown image and link targets in its text outside fenced code blocks (`![map](maps/rain.png)`, `[data](out.zarr)`). A reply names a file in passing with text in single backticks in its text outside fenced code blocks (`` `kenya_rain.zarr` ``) when the chat's file listing has an entry with exactly that path that is not a folder (a zarr store is listed as kind `zarr` and counts). A file named in passing joins the result like a presented one, and in the rest of this section counts as one of a reply's files; steps 2 and 4 say where it differs. Backtick text that is not a listed path, such as a variable name or a bare file name of a file in a folder (`` `rain.zarr` `` for `out/rain.zarr`), names nothing, and text inside tool calls and reasoning blocks names nothing. A line that starts with three backticks opens a fenced code block only when no backtick follows on that line, so an inline span such as ```` ```x``` ```` opens none. A reply with no such file does not change which reply is the last one that presents files. The result holds each path once, in the order of the turns below: a path some reply presents outright is at the latest turn that presented it outright, a later backtick mention does not move it; any other path is at the latest turn that named it.
2. Each file's kind and provenance come from its entry in the chat's file listing. Every kind in `deliverables` must be among the presented files. Each PNG image, `.html` page and zarr store of the last reply that presents files must have provenance, unless no reply names its path other than in backticks: the app reads the `weather_skills_history` chain of those kinds, and no other, into the entry's `provenance_detail`. A file of another kind without provenance, and a file named only in backticks with any problem, are given to the judge with a note; a path named only in backticks never blocks the judge. A file no reply names other than in backticks counts as presented when its kind is in `deliverables` and no file of that kind was presented outright, however many such files there are; otherwise it stays named in passing.
3. Each presented image of type `image/png`, `image/jpeg`, `image/gif` or `image/webp` is downloaded once the play has ended, so a path presented again is judged on its latest version; a failed listing or download, or a listing that lacks a file of the last reply that presents files or a backtick path of the last reply that names any file, is read once more, after waiting `EVAL_POLL_INTERVAL_SECONDS`, before it counts as a problem; when that extra listing fails, the earlier one is used, and when a file of the last reply that presents files is then missing from it the problem also names the listing error. Any other image is given to the judge without its picture, with the note `(not shown as an image)`, which does not block. Nothing else is downloaded.
4. The judge rules on the `judge_criteria` plus a fixed criterion: each presented file's provenance (data source, region, period, aggregation, units) matches the user's request and what was agreed in the chat; the criterion says a file named in passing and not presented is not held to it. Besides the transcript, its call holds one message of harness evidence: each file's path, kind and the turn that presented it (for a file named in passing, `named in passing on turn <n> and not presented`), its `provenance_detail` as JSON, and each image.

When the chat's files could not be listed on either read, no reply presented a file, a kind is missing among all presented files, or a file of the last reply that presents files is not in the listing, has no provenance where step 2 requires it, or cannot be downloaded, the judge model is not called, and the reason is the verdict. Backtick text that is not a listed path is never a file missing from the listing. Such a problem on a file presented earlier does not stop the judge: the file is given to it with a note naming the problem, next to the turn that presented it.

## Outcomes

Each play ends with one outcome, given in the item output and in the `goal_reached` score comment:

| Outcome | Meaning |
|-|-|
| `goal_reached` | The simulated user was done and the judge passed every criterion, including the provenance criterion when the task has `deliverables` |
| `chat_error` | The app returned an error for a turn, or a reply the harness cannot read, a request to the app failed without a response (any `httpx` error, such as a connection error or a timeout), or the reply was not done within `EVAL_TURN_TIMEOUT_SECONDS` (see [Harness settings](#harness-settings)) |
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
| `time_to_answer_seconds` | Time the app spent on turns that got a reply, from sending each message until the first poll of the chat that found its reply done, so each turn includes up to one `EVAL_POLL_INTERVAL_SECONDS` after the reply was done; pauses and failed turns are excluded |

Trace-based scores (model calls, cost, tokens, tools, cache misses) are read from the app's Langfuse traces of the chat: traces named `weather-skills-chat` with session id = chat id, one per turn the app accepted, once they stop changing. They have settled only once every tool observation's output has been read. A tool's output is read once the tool has ended, also when its output is null; a tool still running is read again on a later poll. When the traces have otherwise settled and the only unread tools have no end time (such as a span a turn timeout left open), their outputs are read once as they are; the traces count as complete when that read returned every one of them, and as partial (`the outputs of tools left open could not be read: <count>`) otherwise. When that read found new traces or observations, polling goes on, without the outputs it read for tools still open. A Langfuse read that fails with a transport error, a timeout, or a 429 or 5xx response is retried at `EVAL_POLL_INTERVAL_SECONDS` until `EVAL_TRACE_TIMEOUT_SECONDS`; the failed read does not restart the settle time, which only a change in the observations does. Any other error (such as a 400, 401, 403 or 404, or a Langfuse error without a status code) makes the trace scores missing at once when no read has succeeded yet, and partial, from the last successful read, when one has; a timeout before any read succeeded makes them missing too. The item output's `trace_scores` says whether they are `complete`, `partial` at the timeout (`partial: Langfuse reads failed: <error>` when the last read failed, `partial: traces were still arriving` when traces were missing or still changing, else `partial: tool outputs not read by the timeout: <count>`) or `missing` (with the reason, such as a Langfuse error, the last read's error when the reads that succeeded found no traces, or no traces arriving); when they are missing, the `goal_reached` comment says so too and the other scores are still recorded. Each item's output holds the chat id, the app's trace ids and `provider_models`, and the item's trace carries the chat id as its session id. `provider_models` lists the distinct `model` names of the app's generations in those traces: the model OpenRouter reports for calls it records with Broadcast, the model the app requested for calls the app records. It is empty when the trace scores are missing.

The experiment's per-item cost in Langfuse is the app's cost: the item's trace holds one generation named `app-cost-rollup` whose cost is the item's `cost_to_goal_usd`, with no model and no token counts. It is recorded whenever that score is, and leaves out the simulated user and judge. The rollup repeats the app's cost: a cost total over the chat's session, or across the `sdk-experiment` environment (where Langfuse records the item traces) and the app's, counts it twice unless generations named `app-cost-rollup` are left out.

The experiment's metadata holds the resolved configuration: the configuration's `name` as `config_name`, image with its digest and image id and whether it was pulled, the runner image as `runner_image` with `runner_image_digest` and `runner_image_id`, the app's environment variable names, installed skill packs with their commits, the tool ids the chats used, the chat model as `model` and `catalog_model`, the simulated user and judge models, the turn cap `EVAL_MAX_TURNS` as `max_turns`, the harness commit, `EVAL_BATCH` as `batch` (empty when unset), and the dataset with its version (`dataset_version`, always `latest`: the harness reads the dataset without a version, so the run played the dataset as it was when the run read it), item count and the skill packs it needs (`dataset_skill_packs`, comma-separated, empty when none). `catalog_model` is the configured catalog id; `model` is the base model the app reports for it, without the `prefix_id` of the connection the app serves it from: like the app, the harness takes the last connection in `OPENAI_API_CONFIGS` that is enabled and whose `model_ids` allow the model (`open-router-.anthropic/claude-haiku-5.5` becomes `anthropic/claude-haiku-5.5`). A connection without `enable` is enabled, and one with no or empty `model_ids` allows every model. When no connection is taken, `model` is the base model the app reports, with surrounding whitespace removed. `model` is also the experiment's description.

After reading the base model and before the plays, the harness sends one short request for it to OpenRouter with its own `OPENROUTER_API_KEY` (not the app's key), retried as `EVAL_MODEL_RETRIES` says, and names the experiment after the model named in the reply, recorded in the metadata as `served_model`; the base model must therefore be an OpenRouter model id. A floating alias such as `~anthropic/claude-opus-latest` resolves this way to the model it points to, such as `anthropic/claude-opus-5.5`, as it was when the run started. When that request fails, or its reply names no model or only the alias, the run stops before the experiment is created (exit 1). The models that served each play's calls are in each item's `provider_models`.

A run plays each task once: Langfuse expects each dataset item to appear once per experiment (Langfuse issue #5855).

## Development

Tests and lint, in a container without network access:

```sh
docker compose run --rm --build test
```

The tests of `scripts/run-eval.py` are in the script and run on the host:

```sh
uv run --isolated --no-project --with-requirements scripts/run-eval.py --with pytest pytest scripts/run-eval.py
```

Install the pre-commit hooks:

```sh
uv run pre-commit install
```
