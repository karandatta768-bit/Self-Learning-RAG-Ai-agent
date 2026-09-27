import pytest

from app.graph.workflow import run
from app.tools.calculator import CalculatorTool


def test_calculator_basic():
    t = CalculatorTool()
    r = t.run(expression="(120 - 90) / 90 * 100")
    assert r.success
    assert "33.3333" in r.output or "33.333" in r.output


def test_calculator_rejects_code():
    t = CalculatorTool()
    r = t.run(expression="__import__('os').system('echo hi')")
    assert not r.success


@pytest.mark.slow
def test_full_graph_smoke():
    state = run("What is 15% of 240?", run_id="smoke", isolate_run=True)
    assert state["final_answer"]
    assert state["task_type"]