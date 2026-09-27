from pathlib import Path
import tempfile

from app.memory.episodic import EpisodicMemory
from app.memory.schemas import EpisodeRecord, EvaluationRecord


def _mk(task: str, run_id: str = "r1", task_id: str = "") -> EpisodeRecord:
    return EpisodeRecord(
        id="",
        task=task,
        run_id=run_id,
        task_id=task_id,
        task_type="rag",
        strategy="rag_then_answer",
        tools=["rag"],
        evaluation=EvaluationRecord(success=True, score=0.9),
    )


def test_same_task_never_returned():
    with tempfile.TemporaryDirectory() as d:
        m = EpisodicMemory(path=Path(d) / "ep.jsonl")
        m.write(_mk("What is our refund policy?"))
        hits = m.search(
            "What is our refund policy?", k=5,
            exclude_task="What is our refund policy?",
        )
        assert hits == [], "leak: identical task returned"


def test_cross_run_excluded():
    with tempfile.TemporaryDirectory() as d:
        m = EpisodicMemory(path=Path(d) / "ep.jsonl")
        m.write(_mk("What is our refund policy?", run_id="cold-abc"))
        hits = m.search(
            "Explain the refund policy for enterprise.",
            k=5, exclude_run_id="cold-abc",
        )
        assert hits == [], "leak: cross-run episode returned"


def test_heldout_excluded():
    with tempfile.TemporaryDirectory() as d:
        m = EpisodicMemory(path=Path(d) / "ep.jsonl")
        m.write(_mk("task A variant", task_id="t1"))
        m.write(_mk("task B variant", task_id="t2"))
        hits = m.search("task A variant", k=5, exclude_task_ids={"t1", "t2"})
        assert all(h.task_id not in {"t1", "t2"} for h in hits)


def test_heldout_ids_survive_understand_node():
    """Regression: node_understand used to clobber heldout_task_ids."""
    from app.graph.workflow import run
    state = run(
        "hi",
        run_id="leakcheck",
        task_id="t-hi",
        heldout_task_ids=["t1", "t2", "t3"],
        isolate_run=True,
    )
    assert state.get("heldout_task_ids") == ["t1", "t2", "t3"]
    assert state.get("isolate_run") is True