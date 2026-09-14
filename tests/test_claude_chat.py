from dataclasses import dataclass, field

from src.claude_chat import get_response


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class FakeResponse:
    content: list
    stop_reason: str = "end_turn"


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def test_get_response_returns_text_directly_when_no_tool_use():
    client = FakeClient([FakeResponse(content=[FakeTextBlock(text="Hello there")])])

    def unused_tool_executor(name, tool_input):
        raise AssertionError("tool_executor should not be called")

    result = get_response(client, [{"role": "user", "content": "hi"}], "system", unused_tool_executor)

    assert result == "Hello there"
    assert client.messages.calls[0]["tools"]
    assert client.messages.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_get_response_executes_tool_and_continues():
    tool_use_response = FakeResponse(
        content=[FakeToolUseBlock(id="tu_1", name="get_stock_analysis", input={"ticker": "MDT"})],
        stop_reason="tool_use",
    )
    final_response = FakeResponse(content=[FakeTextBlock(text="MDT looks solid.")])
    client = FakeClient([tool_use_response, final_response])

    calls = []

    def tool_executor(name, tool_input):
        calls.append((name, tool_input))
        return "Ticker: MDT\n...data..."

    result = get_response(client, [{"role": "user", "content": "compare to MDT"}], "system", tool_executor)

    assert result == "MDT looks solid."
    assert calls == [("get_stock_analysis", {"ticker": "MDT"})]

    second_call_messages = client.messages.calls[1]["messages"]
    assert second_call_messages[-2] == {"role": "assistant", "content": tool_use_response.content}
    assert second_call_messages[-1] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "Ticker: MDT\n...data..."}],
    }


def test_get_response_handles_multiple_tool_calls_in_one_turn():
    tool_use_response = FakeResponse(
        content=[
            FakeToolUseBlock(id="tu_1", name="run_dcf_scenario", input={"ticker": "ISRG", "revenue_growth_rate": 0.12}),
            FakeToolUseBlock(id="tu_2", name="run_dcf_scenario", input={"ticker": "ISRG", "revenue_growth_rate": 0.08}),
        ],
        stop_reason="tool_use",
    )
    final_response = FakeResponse(content=[FakeTextBlock(text="Two scenarios compared.")])
    client = FakeClient([tool_use_response, final_response])

    def tool_executor(name, tool_input):
        return f"price for {tool_input['revenue_growth_rate']}"

    result = get_response(client, [], "system", tool_executor)

    assert result == "Two scenarios compared."
    tool_results = client.messages.calls[1]["messages"][-1]["content"]
    assert tool_results == [
        {"type": "tool_result", "tool_use_id": "tu_1", "content": "price for 0.12"},
        {"type": "tool_result", "tool_use_id": "tu_2", "content": "price for 0.08"},
    ]
