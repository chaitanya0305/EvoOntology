"""The BIRD evolution adapter reports only the results of its own run."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bird import evolution_adapter  # noqa: E402
from benchmarks.bird.evolution_adapter import BirdEvolutionAdapter  # noqa: E402


def _write_results(run_dir: Path, ex: bool) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "all_results.json").write_text(json.dumps({
        "metrics": {"overall_ex": 1.0 if ex else 0.0},
        "results": [{"question_id": 1, "ex": ex}],
    }))


def _fake_run(write):
    def run(cmd, **kwargs):
        write(Path(cmd[cmd.index("--output") + 1]))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return run


def test_run_without_results_does_not_report_an_earlier_subject(tmp_path, monkeypatch):
    # The Parent's evaluation is already in the shared output directory.
    _write_results(tmp_path / "semantic" / "parent_run", ex=True)
    # The Candidate's run exits 0 without evaluating any question.
    monkeypatch.setattr(evolution_adapter.subprocess, "run", _fake_run(lambda out: None))
    adapter = BirdEvolutionAdapter("config.yaml", output_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        adapter.evaluate("candidate_r1")


def test_reports_the_results_written_by_this_run(tmp_path, monkeypatch):
    _write_results(tmp_path / "semantic" / "parent_run", ex=True)
    monkeypatch.setattr(
        evolution_adapter.subprocess, "run",
        _fake_run(lambda out: _write_results(out / "semantic" / "candidate_run", ex=False)),
    )
    adapter = BirdEvolutionAdapter("config.yaml", output_dir=str(tmp_path))
    result = adapter.evaluate("candidate_r1")
    assert result["metrics"] == {"overall_ex": 0.0}
    assert result["artifact_paths"] == [
        str(tmp_path / "semantic" / "candidate_run" / "all_results.json")
    ]


def test_resumed_run_that_rewrites_its_results_is_reported(tmp_path, monkeypatch):
    run_dir = tmp_path / "semantic" / "candidate_run"
    _write_results(run_dir, ex=False)
    monkeypatch.setattr(
        evolution_adapter.subprocess, "run",
        _fake_run(lambda out: _write_results(run_dir, ex=True)),
    )
    adapter = BirdEvolutionAdapter("config.yaml", output_dir=str(tmp_path))
    assert adapter.evaluate("candidate_r1")["metrics"] == {"overall_ex": 1.0}
