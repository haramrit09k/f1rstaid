"""The weekly job's commit step, exercised against real (local) git repos:
a bare "remote", a shallow "runner" clone like actions/checkout makes, and a
second clone that moves main while the runner is busy."""
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / ".github" / "scripts" / "commit_index.sh"


def git(cwd, *args, check=True, env=None):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd, check=check, capture_output=True, text=True, env={**os.environ, **(env or {})},
    )


@pytest.fixture
def repos(tmp_path):
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    seed.mkdir()
    git(seed, "init", "-q", "-b", "main")
    (seed / "faiss_index").mkdir()
    (seed / "faiss_index" / "index.faiss").write_text("old index")
    (seed / "app.py").write_text("v1")
    git(seed, "add", "-A")
    git(seed, "commit", "-q", "-m", "initial")
    git(seed, "remote", "add", "origin", f"file://{remote}")
    git(seed, "push", "-q", "origin", "main")

    runner = tmp_path / "runner"  # like actions/checkout: shallow
    git(tmp_path, "clone", "-q", "--depth=1", f"file://{remote}", str(runner))
    return remote, seed, runner


def run_script(runner, **env):
    return subprocess.run(
        ["bash", str(SCRIPT), "main"], cwd=runner, capture_output=True, text=True,
        env={**os.environ, "RETRY_SLEEP_SECONDS": "0", **env},
    )


def remote_file(remote, tmp_path, path):
    check = tmp_path / "check"
    if check.exists():
        subprocess.run(["rm", "-rf", str(check)])
    git(tmp_path, "clone", "-q", f"file://{remote}", str(check))
    return (check / path).read_text()


def test_pushes_the_new_index(repos, tmp_path):
    remote, _, runner = repos
    (runner / "faiss_index" / "index.faiss").write_text("NEW index")

    result = run_script(runner)

    assert result.returncode == 0, result.stderr
    assert remote_file(remote, tmp_path, "faiss_index/index.faiss") == "NEW index"


def test_survives_main_moving_while_the_update_ran(repos, tmp_path):
    """The real failure: a merge landed on main during the 15-minute run."""
    remote, seed, runner = repos
    (runner / "faiss_index" / "index.faiss").write_text("NEW index")
    # someone merges to main meanwhile
    (seed / "app.py").write_text("v2 merged during the run")
    git(seed, "commit", "-qam", "merge during the run")
    git(seed, "push", "-q", "origin", "main")

    # sanity: a plain push from the runner would be rejected
    git(runner, "add", "-A")
    git(runner, "commit", "-qm", "x")
    assert git(runner, "push", "origin", "HEAD:main", check=False).returncode != 0
    git(runner, "reset", "-q", "--hard", "HEAD~1")
    (runner / "faiss_index" / "index.faiss").write_text("NEW index")

    result = run_script(runner)

    assert result.returncode == 0, result.stderr
    assert remote_file(remote, tmp_path, "faiss_index/index.faiss") == "NEW index"
    assert remote_file(remote, tmp_path, "app.py") == "v2 merged during the run"  # their work kept


def test_retries_when_the_first_push_is_rejected(repos, tmp_path):
    remote, _, runner = repos
    (runner / "faiss_index" / "index.faiss").write_text("NEW index")
    marker = tmp_path / "rejected_once"
    hook = remote / "hooks" / "pre-receive"
    hook.write_text(f'#!/bin/sh\nif [ ! -f "{marker}" ]; then touch "{marker}"; echo "simulated race" >&2; exit 1; fi\nexit 0\n')
    hook.chmod(0o755)

    result = run_script(runner)

    assert result.returncode == 0, result.stderr
    assert "Retrying" in result.stdout
    assert remote_file(remote, tmp_path, "faiss_index/index.faiss") == "NEW index"


def test_gives_up_with_an_error_when_every_push_fails(repos, tmp_path):
    remote, _, runner = repos
    (runner / "faiss_index" / "index.faiss").write_text("NEW index")
    hook = remote / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)

    result = run_script(runner)

    assert result.returncode == 1
    assert "Could not push" in result.stdout


def test_nothing_to_commit_is_a_clean_success(repos):
    _, _, runner = repos
    result = run_script(runner)
    assert result.returncode == 0
    assert "No knowledge base changes" in result.stdout
