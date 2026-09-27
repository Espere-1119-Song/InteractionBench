# Protocols (test methods)

A protocol decides what the system sees and when it is asked. It returns one *poll* per
decision step; polls in which the system spoke become the emissions that are scored.

## Built-in protocols

### Polling: `sliding`, `cumulative`, `interleaved`

A turn-based model is asked at fixed intervals of stream time whether to speak or wait.

Tick schedule:

- B-type and C-type items: the request stands from t = 0. Ticks at `interval`,
  `2*interval`, ... up to the end of the video.
- A-type items: the question is not shown before `question_time_s`. The first tick is at
  the question time, then every `interval` seconds for `a_window` seconds or until the
  video ends.

The reply format is fixed:

```
DECISION: SPEAK or WAIT
RESPONSE: <one line if SPEAK>
```

| Protocol | Context at tick t |
|---|---|
| `sliding` | the last `max_frames` frames with timestamp <= t. A new conversation at every tick. |
| `cumulative` | all frames with timestamp <= t, evenly subsampled to `max_frames`, always including the latest. A new conversation at every tick. |
| `interleaved` | one conversation for the whole item. Each tick appends the frames that arrived since the previous tick (at most `max_new_per_turn`) and a short prompt. The replies stay in the conversation. When the images in context exceed `max_frames`, the oldest are dropped; text turns are kept. |

No protocol ever shows a frame with a timestamp later than the tick.

### `offline`

The system receives `max_frames` frames sampled over the whole video and lists every
moment at which it would have responded, as lines `[t=12.3] text`. The claimed times
become the emission times. For A-type items the answer is placed at the question time.
The system has hindsight, so these scores are a reference and not a real-time result.

### `--blind`

Removes all frames from any protocol. The system sees the question, the options and the
stream time only.

## Parameters

| Option | Default | Effect |
|---|---|---|
| `--interval` | 1.0 | seconds of stream time between ticks |
| `--a-window` | 10.0 | polling time after an A-type question |
| `--sample-fps` | 2.0 | frames decoded per second of video |
| `--max-frames` | 16 | frames per tick (context cap when interleaved) |
| `--max-new-per-turn` | 8 | interleaved: new frames per tick |
| `--max-long-side` | 512 | longer image side after resizing |
| `--max-new-tokens` | 96 | generation length |
| `--hint-set` | default | wording of the per-task instruction: `default`, `v2`, `v3` |

The defaults are the settings of the main table.

## Add a protocol

```python
# my_protocol.py
from interactionbench import Protocol, register_protocol
from interactionbench.parsing import parse_decision
from interactionbench.protocols import make_poll, poll_ticks, user_turn

@register_protocol("my-protocol")
class MyProtocol(Protocol):
    name = "my-protocol"

    def run_item(self, model, frames, item, question, cfg):
        polls = []
        for t in poll_ticks(item, cfg.interval, cfg.a_window):
            visible = [f for f in frames if f.time <= t][-cfg.max_frames:]
            messages = [{"role": "system", "content": "..."},
                        user_turn(visible, f"t={t:.1f}s. {question}\n...")]
            gen = model.timed_chat(messages, max_new_tokens=cfg.max_new_tokens)
            spoke, response = parse_decision(gen.text)
            polls.append(make_poll(t, spoke, response, gen))
        return polls
```

```bash
ibench run --plugin my_protocol.py --protocol my-protocol --model qwen3vl-8b --mcq
```

Arguments of `run_item`:

| Argument | Content |
|---|---|
| `model` | a `ChatModel`; call `model.timed_chat(messages, max_new_tokens)` |
| `frames` | decoded frames of the whole video, each with `.time` and `.image`; empty when `--blind` |
| `item` | the `BenchItem`: `time_type`, `capability`, `question_time_s`, `duration_s`, ... |
| `question` | the question text, already rendered as multiple choice where the item has options |
| `cfg` | `ProtocolConfig`; options given as `--protocol-arg key=value` arrive in `cfg.extra` |

Rules a protocol must respect to give comparable numbers:

1. Do not show a frame later than the current decision time.
2. Do not show an A-type question before `item.question_time_s`.
3. Report the stream time of the decision as `t`, not the wall-clock time.

Two complete examples are in [examples/custom_protocol.py](../examples/custom_protocol.py).
