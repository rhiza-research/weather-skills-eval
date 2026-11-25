"""The eval image's command on a batch VM. Reads the batch values from the VM's
metadata, runs scripts/run-eval.py in non-interactive mode with them, then
deletes the VM through the Compute Engine API with the VM's service account,
whatever the batch's exit code. A VM it fails to delete is deleted at the
instance template's maximum run duration.

Runs with the Python that has run-eval.py's dependencies; uses only the
standard library itself."""

import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

METADATA = "http://metadata.google.internal/computeMetadata/v1/"
COMPUTE = "https://compute.googleapis.com/compute/v1/"
RUN_EVAL = Path(__file__).resolve().parent / "run-eval.py"


def metadata(path):
    """The metadata server's answer for ``path``, as text."""
    request = urllib.request.Request(METADATA + path, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def load_run_eval():
    """scripts/run-eval.py, loaded as a module."""
    spec = importlib.util.spec_from_file_location("run_eval", RUN_EVAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_batch():
    """Runs non-interactive mode with each batch value the VM's metadata holds,
    as ``--<key>=<value>``; other keys, such as cloud-init's, are left out.
    Returns its exit code; a missing value exits 2."""
    run_eval = load_run_eval()
    attributes = json.loads(metadata("instance/attributes/?recursive=true"))
    argv = [f"--{key}={attributes[key]}" for key in run_eval.metadata_keys() if key in attributes]
    try:
        # Never run_eval.main: with no metadata, an empty argv would be interactive mode.
        return run_eval.run_noninteractive(run_eval.parse_batch(argv))
    except SystemExit as exc:
        # argparse exits on a missing or invalid flag.
        return exc.code if isinstance(exc.code, int) else 1


def delete_instance():
    """Deletes this VM; raises on failure."""
    project = metadata("project/project-id")
    zone = metadata("instance/zone").rpartition("/")[2]
    name = metadata("instance/name")
    token = json.loads(metadata("instance/service-accounts/default/token"))["access_token"]
    request = urllib.request.Request(
        f"{COMPUTE}projects/{project}/zones/{zone}/instances/{name}",
        method="DELETE",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=60):
        pass


def main():
    code = 1
    try:
        code = run_batch()
    except Exception as exc:
        print(
            f"Batch failed in vm-entry.py: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True
        )
    finally:
        try:
            delete_instance()
        except Exception as exc:
            # stderr reaches the serial console through the systemd unit.
            print(f"VM not deleted: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
