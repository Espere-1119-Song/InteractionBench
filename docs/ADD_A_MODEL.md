# Add a model

## 1. A Hugging Face chat VLM, no code

```bash
ibench run --model hf:<repo-or-local-path> \
    --model-arg model_cls=AutoModelForImageTextToText \
    --model-arg image_style=placeholder \
    --protocol sliding --mcq
```

| `--model-arg` | Values | Meaning |
|---|---|---|
| `model_cls` | any class name in `transformers` | class used for `from_pretrained` |
| `image_style` | `placeholder` (default), `payload` | `placeholder`: the chat template receives `{"type": "image"}` markers and the images go to the processor separately (LLaVA-OneVision, Gemma). `payload`: the images stay inside the message content (Qwen-VL family). |
| `trust_remote_code` | `true`, `false` | for repositories with custom code |
| `dtype` | `bfloat16` (default), `float16`, ... | weight type |
| `attn_implementation` | `sdpa` (default), `eager`, `flash_attention_2` | attention kernel |
| `chat_template_kwargs` | JSON object | e.g. `{"enable_thinking": false}` |
| `from_pretrained_kwargs` | JSON object | passed to `from_pretrained` |

Text-only language models use `hf-text:<repo>` together with `--blind`.

A fine-tuned checkpoint of a listed model: `--model qwen3vl-8b --model-path /path/to/ckpt`.

Mage-VL is available as the named configuration `mage-vl-4b` (or the `mage-vl`
alias). The named configuration pins the tested checkpoint revision. Install its
known-compatible runtime first; the second step needs Torch to be importable:

```bash
pip install 'torch==2.9.1' 'torchvision==0.24.1' 'transformers==5.7.0' accelerate
pip install --no-build-isolation 'mamba-ssm==2.3.2.post1' opencv-python decord
yes y | ibench run --model mage-vl --protocol sliding --mcq --limit 10 \
  --out results/runs/mage-vl_sliding_smoke
```

The `yes y` supplies a confirmation requested by Mage-VL's nested remote processor;
the code being confirmed is pinned by the model configuration. This uses the same
frame-sampled, multi-image polling protocol as the other turn-based VLMs. It does not
exercise Mage-VL's native video/codec processor or its visual-only proactive gate;
those are different evaluation protocols.

## 2. An OpenAI-compatible endpoint, no code

```bash
export MY_KEY=...
ibench run --model api:<model-name> --api-base https://host/v1 --api-key-env MY_KEY \
    --protocol sliding --mcq --items benchmark/splits/subset103.txt
```

This covers local servers (vLLM, SGLang, llama.cpp) and commercial APIs. Frames are
sent inline as base64 JPEG.

| `--model-arg` | Default | Meaning |
|---|---|---|
| `max_tokens_field` | `max_completion_tokens` | some servers expect `max_tokens` |
| `extra_body` | `{}` | JSON merged into every request |
| `allow_empty_key` | `false` | for local servers without authentication |
| `timeout_s`, `max_retries` | 180, 6 | transport; retries on 429 and 5xx with backoff |

## 3. A named configuration

Put configurations in a JSON or YAML file ([examples/my_models.json](../examples/my_models.json)):

```json
{"my-model": {"adapter": "hf-vlm", "repo": "org/model", "image_style": "payload"}}
```

```bash
ibench run --model-config my_models.json --model my-model ...
ibench list models --model-config my_models.json
```

## 4. Your own adapter

Implement one method:

```python
# my_model.py
from interactionbench import ChatModel, register_adapter

@register_adapter("mine")
class MyModel(ChatModel):
    def __init__(self, repo, short_name=None, **kwargs):
        self.name = short_name or repo
        ...                                   # load weights or open a client

    def chat(self, messages, max_new_tokens=96) -> str:
        ...                                   # return the reply text
```

```bash
ibench run --plugin my_model.py --model mine:<anything> --protocol sliding --mcq
```

`messages` is the canonical conversation:

```python
[{"role": "system", "content": "<str>"},
 {"role": "user", "content": [{"type": "image", "image": <PIL.Image>}, ...,
                              {"type": "text", "text": "<str>"}]},
 {"role": "assistant", "content": "<str>"}]      # only under the interleaved protocol
```

The text after `mine:` arrives as `repo`. Values given with `--model-arg key=value`
arrive as keyword arguments. Without registration, the class path works as well:
`--model my_pkg.my_module:MyModel`.

## 5. A system that is not turn-based

Natively streaming models, agents and humans do not fit `chat`. Write the predictions
file yourself and score it:

```json
{"video_id": "abc", "item_index": 0, "model": "my-system",
 "emissions": [{"t": 12.3, "content": "B"}, {"t": 40.0, "content": "C"}]}
```

```bash
ibench eval my_preds.jsonl --mcq-key --judge hf:Qwen/Qwen3-14B
```

Template: [examples/write_predictions.py](../examples/write_predictions.py). Complete
runners for nine streaming systems are in [baselines/](../baselines).

Requirements for a comparable result:

- `t` is stream time in seconds: the time of the latest frame the system had received
  when it decided to speak.
- The system must not receive frames later than `t`.
- A-type questions must not be shown before `question_time_s`.
- Multiple-choice items must be asked with the options of `mcq_options_v4.jsonl`
  (`interactionbench.prompts.format_question` renders them).
- One line per item. An item without a line is scored as silent.
