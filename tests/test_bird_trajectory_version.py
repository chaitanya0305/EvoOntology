"""BIRD trajectories record the semantic version the agent actually used."""

import asyncio
import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BIRD_DIR = REPO_ROOT / "benchmarks" / "bird"
for path in (REPO_ROOT, BIRD_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evoontology import SemanticStore  # noqa: E402

_EMPTY = {"terms": [], "mappings": [], "relations": [], "constraints": [], "evidence": []}


def _workspace(tmp_path):
    store = tmp_path / "store"
    SemanticStore.save_version(store, "ontology_v0", _EMPTY)
    SemanticStore.save_version(store, "candidate_r1", _EMPTY)
    SemanticStore.set_active(store, "ontology_v0")
    return store


def test_record_trajectory_uses_the_given_version(tmp_path):
    from trajectory_recorder import record_trajectory

    store = _workspace(tmp_path)
    task_id = record_trajectory(
        str(store), db_id="db", question="How many?", question_id=7,
        split="train", messages=[], ontology_version="candidate_r1",
    )
    record = json.loads((store / "trajectories" / f"{task_id}.json").read_text())
    assert record["ontology_version"] == "candidate_r1"
    # Without an explicit version the active version is still recorded.
    task_id = record_trajectory(str(store), db_id="db", question="Other?", messages=[])
    record = json.loads((store / "trajectories" / f"{task_id}.json").read_text())
    assert record["ontology_version"] == "ontology_v0"


def test_evaluation_trajectory_records_the_evaluated_version(tmp_path, monkeypatch):
    pytest.importorskip("dotenv")
    pytest.importorskip("yaml")
    from benchmarks.bird import run_evaluation
    from benchmarks.bird.config import ExperimentConfig

    store = _workspace(tmp_path)
    db_dir = tmp_path / "dbs"
    (db_dir / "db").mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "db" / "db.sqlite")
    conn.execute("CREATE TABLE t (a)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(run_evaluation, "DB_DIR", db_dir)
    fake_providers = types.ModuleType("benchmarks.bird.agent.llm_providers")
    fake_providers.create_llm_provider = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, "benchmarks.bird.agent.llm_providers", fake_providers)

    class FakeClient:
        async def connect_to_servers(self, configs):
            pass

        async def close(self):
            pass

    class FakeAgent:
        semantic_call_count = 0

        def __init__(self, **kwargs):
            pass

        async def start_session(self, **kwargs):
            self.session = types.SimpleNamespace(pred_sql="SELECT a FROM t", total_turns=1)
            return self.session

        async def run(self):
            pass

        def export_trace(self):
            return {"messages": []}

    class FakeRuntime:
        def __init__(self, store_path, version=""):
            pass

        def manifest(self, db_id=""):
            return ""

    config = ExperimentConfig()
    config.condition = "semantic"
    config.semantic.enabled = True
    config.semantic.store_path = str(store)
    config.semantic.version = "candidate_r1"  # --semantic-version candidate_r1

    question = {"question_id": 7, "question": "How many?", "SQL": "SELECT a FROM t"}
    result, _ = asyncio.run(run_evaluation.run_single_question(
        question, config, "db", {}, FakeClient, FakeAgent, FakeRuntime,
        record_trajectories=True, split="train",
    ))
    assert not result.error
    record = json.loads((store / "trajectories" / "db_7.json").read_text())
    assert record["ontology_version"] == "candidate_r1"
    assert SemanticStore.active_version(store) == "ontology_v0"
