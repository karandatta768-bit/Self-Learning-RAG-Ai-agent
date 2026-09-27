from app.models.llm import _dedupe, _inject_no_reasoning, _strip_thinking


def test_inject_no_reasoning_appends_to_system_without_mutating_input():
    messages = [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "Hi"}]

    result = _inject_no_reasoning(messages)

    assert "Output ONLY the final answer." in result[0]["content"]
    assert messages[0]["content"] == "Be concise."


def test_inject_no_reasoning_adds_system_message_when_missing():
    result = _inject_no_reasoning([{"role": "user", "content": "Hi"}])

    assert result[0]["role"] == "system"
    assert "Do not narrate reasoning." in result[0]["content"]


def test_strip_thinking_blocks_and_footer():
    assert _strip_thinking("Answer <think>private reasoning</think>\n...done thinking.\n") == "Answer"


def test_dedupe_removes_repeated_answer_and_adjacent_sentence():
    assert _dedupe("One. Two.\nOne. Two.") == "One. Two."
    assert _dedupe("Same sentence. Same sentence.") == "Same sentence."


def test_dedupe_preserves_distinct_paragraphs():
    text = "First paragraph.\n\nSecond paragraph."

    assert _dedupe(text) == text
