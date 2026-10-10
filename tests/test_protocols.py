"""Protocols: tick schedule, what the model is shown, and the polls that come back."""

from interactionbench import ChatModel, ProtocolConfig, get_protocol, iter_items, load_benchmark
from interactionbench.models import collect_images
from interactionbench.prompts import format_question
from interactionbench.protocols import poll_ticks


class Recorder(ChatModel):
    name = "recorder"

    def __init__(self, reply="DECISION: WAIT\nRESPONSE:"):
        self.reply = reply
        self.calls = []

    def chat(self, messages, max_new_tokens=96):
        self.calls.append(messages)
        return self.reply(messages) if callable(self.reply) else self.reply


def _items(bench):
    return {it.item_id: it for it in iter_items(load_benchmark(bench))}


def test_ticks_standing_and_reveal(bench):
    items = _items(bench)
    b = items["vidAAAAAAAA#1"]
    assert poll_ticks(b, 1.0, 10.0) == [float(t) for t in range(1, 21)]
    a = items["vidAAAAAAAA#0"]
    assert poll_ticks(a, 1.0, 10.0) == [15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
    assert poll_ticks(a, 1.0, 2.0) == [15.0, 16.0, 17.0]


def test_question_is_hidden_before_reveal(bench, make_frames):
    a = _items(bench)["vidAAAAAAAA#0"]
    m = Recorder()
    polls = get_protocol("sliding").run_item(m, make_frames(20), a, a.question, ProtocolConfig())
    assert [p["t"] for p in polls][0] == 15.0
    text = m.calls[0][1]["content"][-1]["text"]
    assert "The user JUST asked" in text and a.question in text


def test_sliding_window_is_causal_and_bounded(bench, make_frames):
    b = _items(bench)["vidAAAAAAAA#1"]
    m = Recorder()
    get_protocol("sliding").run_item(m, make_frames(20), b, b.question,
                                     ProtocolConfig(max_frames=4))
    for k, messages in enumerate(m.calls, 1):
        times = [im.t for im in collect_images(messages)]
        assert len(times) <= 4
        assert all(t <= k for t in times), "a frame from the future was shown"
        assert times == sorted(times)
    assert [im.t for im in collect_images(m.calls[-1])] == [18.25, 18.75, 19.25, 19.75]


def test_cumulative_keeps_the_latest_frame(bench, make_frames):
    b = _items(bench)["vidAAAAAAAA#1"]
    m = Recorder()
    get_protocol("cumulative").run_item(m, make_frames(20), b, b.question,
                                        ProtocolConfig(max_frames=4))
    times = [im.t for im in collect_images(m.calls[-1])]
    assert len(times) == 4 and times[0] == 0.25 and times[-1] == 19.75


def test_interleaved_carries_replies_and_caps_images(bench, make_frames):
    b = _items(bench)["vidAAAAAAAA#1"]
    m = Recorder(reply=lambda msgs: f"DECISION: SPEAK\nRESPONSE: turn {len(msgs)}")
    polls = get_protocol("interleaved").run_item(
        m, make_frames(20), b, b.question, ProtocolConfig(max_frames=6, max_new_per_turn=2))
    last = m.calls[-1]
    assert last[0]["role"] == "system" and b.question in last[0]["content"]
    assert sum(1 for x in last if x["role"] == "assistant") == len(polls)
    assert len(collect_images(last)) <= 6
    assert all(p["spoke"] for p in polls)


def test_blind_shows_no_frames(bench, make_frames):
    b = _items(bench)["vidAAAAAAAA#1"]
    for name in ("sliding", "cumulative", "interleaved"):
        m = Recorder()
        get_protocol(name).run_item(m, make_frames(20), b, b.question, ProtocolConfig(blind=True))
        assert all(not collect_images(c) for c in m.calls), name


def test_offline_parses_timed_lines(bench, make_frames):
    items = _items(bench)
    b, a = items["vidAAAAAAAA#1"], items["vidAAAAAAAA#0"]
    m = Recorder("[t=8.5] The door opens.\nnoise\n[t=3] early guess")
    polls = get_protocol("offline").run_item(m, make_frames(20), b, b.question, ProtocolConfig())
    assert [(p["t"], p["response"]) for p in polls] == [(3.0, "early guess"), (8.5, "The door opens.")]
    assert len(m.calls) == 1
    polls = get_protocol("offline").run_item(Recorder("[t=4.0] B"), make_frames(20), a,
                                             a.question, ProtocolConfig())
    assert [(p["t"], p["response"]) for p in polls] == [(15.0, "B")]
    polls = get_protocol("offline").run_item(Recorder("NO_RESPONSE"), make_frames(20), b,
                                             b.question, ProtocolConfig())
    assert len(polls) == 1 and polls[0]["spoke"] is False


def test_multiple_choice_rendering(bench):
    a = _items(bench)["vidAAAAAAAA#0"]
    q = format_question(a, {"stem": "Which colour was the cup?", "options": ["red", "blue"]})
    assert q.splitlines()[:3] == ["Which colour was the cup?", "A. red", "B. blue"]
    assert format_question(a, None) == a.question


def test_hint_sets_differ(bench, make_frames):
    b = _items(bench)["vidAAAAAAAA#1"]
    texts = {}
    for hs in ("default", "v2", "v3"):
        m = Recorder()
        get_protocol("sliding").run_item(m, make_frames(3), b, b.question,
                                         ProtocolConfig(hint_set=hs))
        texts[hs] = m.calls[0][1]["content"][-1]["text"]
    assert len(set(texts.values())) == 3
