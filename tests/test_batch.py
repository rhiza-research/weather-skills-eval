"""The batch, through ``batch.main`` with fake Docker and a fake Secret Manager."""

import io
import json
import tarfile
import threading
from types import SimpleNamespace

from weather_skills_eval import batch as batch_module
from weather_skills_eval import cli
from weather_skills_eval.batch import (
    CONFIGS_REPO_DIR,
    DOCKER_SOCKET,
    HARNESS_LABEL_VALUE,
    NETWORK,
    PR_NUMBER_LABEL,
    REVISION_LABEL,
    main,
    secret_id,
)
from weather_skills_eval.instance import BATCH_LABEL, LABEL

BATCH = "2026-11-23T10:00:00Z"
EVAL_ID = "sha256:" + "e" * 64
APP_ID = "sha256:" + "a" * 64
APP_IMAGE = "ghcr.io/rhiza-research/weather-skills-chat:pr-42"
RUNNER_IMAGE = "ghcr.io/rhiza-research/weather-skills-sandbox-runner:prod"
CONFIGS_COMMIT = "c" * 40
OPENROUTER = "https://openrouter.ai/api/v1"

# What the launcher passes, plus the dotenv file's harness settings.
ENVIRON = {
    "EVAL_BATCH": BATCH,
    "EVAL_MODELS": "m-a,m-b",
    "EVAL_CONFIGS_REPO": "https://github.com/rhiza-research/weather-skills-configs",
    "EVAL_CONFIGS_COMMIT": CONFIGS_COMMIT,
    "EVAL_CONNECTION_KEYS": f"{OPENROUTER}=APP_OPENROUTER_API_KEY",
    "EVAL_IMAGE": EVAL_ID,
    "APP_IMAGE": APP_IMAGE,
    "RUNNER_IMAGE": RUNNER_IMAGE,
    "EVAL_JUDGE_MODELS": "openrouter/test/judge",
    "LANGFUSE_PUBLIC_KEY": "pk-test",
    "LANGFUSE_SECRET_KEY": "sk-lf-test",
    "OPENROUTER_API_KEY": "or-test",
    "APP_WEBUI_SECRET_KEY": "webui-secret",
    "APP_ADMIN_PASSWORD": "admin-password",
    "APP_ADMIN_API_KEY": "app-key",
    "APP_OPENROUTER_API_KEY": "or-app",
    "LANGFUSE_TRACING_ENVIRONMENT": "eval",
}


def write_repo(root, models, connection=OPENROUTER) -> None:
    """A copy of the configs repo with ``models`` and the ``connection``."""
    for name in ("model", "skill-packs", "connections", "settings"):
        path = root / "schemas" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"x-schema-version": 1}')
    for model in models:
        path = root / "models" / f"{model}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema_version": 1, "model": {"id": model}, "eval": {"variant": None}}
        path.write_text(json.dumps(data))
    for name, key, value in (
        ("skill-packs.yaml", "skill_packs", []),
        ("connections.yaml", "connections", {connection: {}}),
        ("settings.yaml", "settings", {}),
    ):
        (root / name).write_text(json.dumps({"schema_version": 1, key: value}))


class RunContainer:
    """A run's container: its ``stdout`` and ``stderr`` chunks, then ``code``.
    A held one's stdout ends only once ``release`` is set."""

    def __init__(self, kwargs, stdout=(), stderr=(), code=0, hold=False, on_wait=None):
        self.kwargs = kwargs
        self.name = self.id = kwargs["name"]
        self.model = kwargs["environment"]["EVAL_MODEL"]
        self.stdout, self.stderr, self.code = list(stdout), list(stderr), code
        self.release = threading.Event()
        if not hold:
            self.release.set()
        self.on_wait = on_wait
        self.archives = []
        self.started = self.removed = False

    def put_archive(self, path, data):
        self.archives.append((path, data))
        return True

    def start(self):
        self.started = True

    def logs(self, stream, follow, stdout, stderr):
        assert stream and follow and stdout != stderr
        return self._stdout() if stdout else iter(self.stderr)

    def _stdout(self):
        yield from self.stdout
        self.release.wait(10)

    def wait(self):
        if self.on_wait:
            self.on_wait()
        return {"StatusCode": self.code}

    def remove(self, force=False):
        self.removed = True

    def files(self) -> dict[str, str]:
        """The files copied in, by absolute path, as text."""
        [(path, data)] = self.archives
        assert path == "/"
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            return {
                "/" + m.name: tar.extractfile(m).read().decode()
                for m in tar.getmembers()
                if m.isfile()
            }


class FakeDocker:
    """Docker for a batch. ``run`` makes the container of each run from the
    create arguments."""

    def __init__(self, run=None, networks=(NETWORK,), app_labels=None):
        self.run = run or (lambda kwargs: RunContainer(kwargs, [b"done\n"]))
        self.runs: list[RunContainer] = []
        self.pulled, self.looked_up, self.networks_created = [], [], []
        images = {
            APP_IMAGE: SimpleNamespace(id=APP_ID, attrs={"Config": {"Labels": app_labels or {}}}),
        }
        outer = self

        class Images:
            def get(self, ref):
                outer.looked_up.append(ref)
                return images[ref]

            def pull(self, ref):
                outer.pulled.append(ref)
                return images[ref]

        class Containers:
            def create(self, image, **kwargs):
                container = outer.run({"image": image, **kwargs})
                outer.runs.append(container)
                return container

        class Networks:
            def list(self, names):
                # Docker's names filter matches part of a name too.
                return [SimpleNamespace(name=n) for n in networks if names[0] in n]

            def create(self, name, labels):
                outer.networks_created.append((name, labels))

        self.images, self.containers = Images(), Containers()
        self.networks = Networks()


def run_batch(
    tmp_path, fake, environ=None, secrets=None, models=("m-a", "m-b"), connection=OPENROUTER
):
    """``main`` with a configs repo copy holding ``models``; returns the exit
    code, stdout, stderr and the secrets read."""
    repo = tmp_path / "configs-repo"
    write_repo(repo, models, connection)
    read = []

    def reader(project, secret):
        read.append((project, secret))
        value = (secrets or {}).get(secret)
        if value is None:
            raise PermissionError("403 permission denied")
        return value

    out, err = io.StringIO(), io.StringIO()
    code = main(
        {**ENVIRON, **(environ or {})},
        docker_client_factory=lambda: fake,
        secret_reader=reader,
        configs_repo_dir=repo,
        out=out,
        err=err,
    )
    return code, out.getvalue(), err.getvalue(), read


def test_a_batch_runs_each_model_once_per_round(tmp_path):
    def run(kwargs):
        failed = kwargs["environment"]["EVAL_MODEL"] == "m-a"
        out = [b"Goal reached ", b"on 1 of 1 tasks\n"]
        return RunContainer(kwargs, out, [b"Task 1 of 1 started: t1\n"], code=2 if failed else 0)

    labels = {REVISION_LABEL: "8dd46ed4", PR_NUMBER_LABEL: "42"}
    fake = FakeDocker(run, app_labels=labels)

    code, out, err, read = run_batch(tmp_path, fake, {"EVAL_REPETITIONS": "2"})

    assert code == 1
    assert read == []
    # In rounds, from the eval image ID, each with the configs repo files copied in.
    assert [r.model for r in fake.runs] == ["m-a", "m-b", "m-a", "m-b"]
    for container in fake.runs:
        kwargs = container.kwargs
        assert kwargs["image"] == EVAL_ID
        assert kwargs["entrypoint"] == ["weather-skills-eval"]
        assert kwargs["command"] == ["run"]
        files = container.files()
        assert f"{CONFIGS_REPO_DIR}/models/m-b.yaml" in files
        assert f"{CONFIGS_REPO_DIR}/schemas/model.json" in files
        assert kwargs["labels"] == {LABEL: HARNESS_LABEL_VALUE, BATCH_LABEL: BATCH}
        assert kwargs["network"] == NETWORK
        assert kwargs["volumes"] == {DOCKER_SOCKET: {"bind": DOCKER_SOCKET, "mode": "rw"}}
        env = kwargs["environment"]
        assert env["EVAL_CONFIGS_DIR"] == CONFIGS_REPO_DIR
        assert env["EVAL_CONFIGS_COMMIT"] == CONFIGS_COMMIT
        assert env["EVAL_DOCKER_NETWORK"] == NETWORK
        assert env["EVAL_APP_IMAGE_REVISION"] == "8dd46ed4"
        assert env["EVAL_APP_IMAGE_PR_NUMBER"] == "42"
        assert env["EVAL_BATCH"] == BATCH and env["APP_ADMIN_API_KEY"] == "app-key"
        # A variable with no secret name reaches every run as it is.
        assert env["LANGFUSE_TRACING_ENVIRONMENT"] == "eval"
        assert container.started and container.removed
    assert fake.networks_created == []
    # One at once: unprefixed, each line whole across chunks.
    assert out.count("\nGoal reached on 1 of 1 tasks\n") == 4
    assert "m-a run 1: exit 2\n" in out
    assert "m-b run 1: exit 0\n" in out
    assert out.endswith(f"Batch {BATCH} finished: 2 of 4 runs exited 0\n")
    assert err == "Task 1 of 1 started: t1\n" * 4


def test_the_network_is_created_only_when_missing(tmp_path):
    present = FakeDocker()
    run_batch(tmp_path, present)
    assert present.networks_created == []

    # Only a network whose name contains it.
    missing = FakeDocker(networks=[f"{NETWORK}-old"])
    run_batch(tmp_path, missing)
    assert missing.networks_created == [(NETWORK, {LABEL: "network", BATCH_LABEL: BATCH})]


def test_runs_start_in_queue_order_with_at_most_the_runs_at_once(tmp_path):
    lock = threading.Lock()
    started, held = [], []
    going = {"now": 0, "most": 0}
    models = [f"m-{n}" for n in "abcde"]

    def ended():
        with lock:
            going["now"] -= 1

    def run(kwargs):
        container = RunContainer(kwargs, [b"working\n"], hold=True, on_wait=ended)
        with lock:
            started.append(container.model)
            held.append(container)
            going["now"] += 1
            going["most"] = max(going["most"], going["now"])
            # With every slot taken the oldest run ends; after the last start, all do.
            releasing = held if len(started) == len(models) else held[:1]
            if going["now"] == 3 or len(started) == len(models):
                for each in releasing:
                    each.release.set()
                del held[: len(releasing)]
        return container

    fake = FakeDocker(run)

    code, out, _, _ = run_batch(
        tmp_path,
        fake,
        {"EVAL_MODELS": ",".join(models), "EVAL_PARALLEL_RUNS": "3"},
        models=models,
    )

    assert code == 0
    assert started == models
    assert going["most"] == 3
    for model in models:
        assert f"{model} run 1: exit 0\n" in out


def test_with_runs_at_once_every_line_of_a_run_is_prefixed(tmp_path):
    def run(kwargs):
        model = kwargs["environment"]["EVAL_MODEL"]
        # The last line has no newline.
        return RunContainer(kwargs, [f"{model} one\n{model} t".encode(), b"wo"], [b"progress\n"])

    fake = FakeDocker(run)

    _, out, err, _ = run_batch(tmp_path, fake, {"EVAL_PARALLEL_RUNS": "2"})

    for model in ("m-a", "m-b"):
        lines = [line for line in out.splitlines() if line.startswith(f"[{model} run 1]")]
        assert lines == [f"[{model} run 1] {model} one", f"[{model} run 1] {model} two"]
        assert f"[{model} run 1] progress\n" in err


def test_a_secret_comes_from_the_environment_else_from_secret_manager(tmp_path):
    secrets = {secret_id("APP_OPENROUTER_API_KEY"): "from-manager"}
    fake = FakeDocker()

    code, _, _, read = run_batch(
        tmp_path,
        fake,
        # Empty in the dotenv file counts as unset, and the read value replaces it.
        {
            "EVAL_SECRETS_PROJECT": "sheerwater",
            "APP_OPENROUTER_API_KEY": "",
            # A secret no run needs.
            "OTHER_API_KEY": "other",
        },
        secrets=secrets,
    )

    assert code == 0
    # The connection's key, named by EVAL_CONNECTION_KEYS, is read once; the
    # secrets the environment sets are not.
    assert read == [("sheerwater", "weather-skills-eval-app-openrouter-api-key")]
    for container in fake.runs:
        env = container.kwargs["environment"]
        assert env["APP_OPENROUTER_API_KEY"] == "from-manager"
        assert env["LANGFUSE_SECRET_KEY"] == "sk-lf-test"
        # Each run gets only the secrets it needs; settings still reach it.
        assert "OTHER_API_KEY" not in env
        assert env["EVAL_CONNECTION_KEYS"] == ENVIRON["EVAL_CONNECTION_KEYS"]


def test_read_secret_reads_the_latest_version_and_strips_it(monkeypatch):
    from google.cloud import secretmanager

    names = []

    class Client:
        def access_secret_version(self, name):
            names.append(name)
            return SimpleNamespace(payload=SimpleNamespace(data=b" s3cret\n"))

    monkeypatch.setattr(secretmanager, "SecretManagerServiceClient", Client)

    assert batch_module.read_secret("sheerwater", "weather-skills-eval-x") == "s3cret"
    assert names == ["projects/sheerwater/secrets/weather-skills-eval-x/versions/latest"]


def test_the_cli_batch_command_runs_the_batch_from_its_environment(monkeypatch, capsys):
    monkeypatch.delenv("EVAL_MODELS", raising=False)

    def no_docker():
        raise AssertionError("a batch without settings never reaches Docker")

    assert cli.main(["batch"], docker_client_factory=no_docker) == batch_module.EXIT_NOT_STARTED
    assert capsys.readouterr().err.startswith("Batch not started: EVAL_MODELS")
