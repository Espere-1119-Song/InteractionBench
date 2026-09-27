# Judges

The judge decides whether a free-form response conveys the reference answer. It
returns 1 or 0. Multiple-choice items are scored by option match and never reach the
judge.

## Use a judge

```bash
# local model (the paper protocol)
ibench eval preds.jsonl --mcq-key --judge hf:Qwen/Qwen3-14B \
    --judge-cache results/judge_cache/qwen3-14b_v2.jsonl

# OpenAI-compatible endpoint
export IBENCH_JUDGE_API_KEY=...
ibench eval preds.jsonl --mcq-key --judge api:<model> \
    --judge-arg base_url=https://host/v1 --judge-cache results/judge_cache/<model>_v2.jsonl

# stored verdicts, no model
ibench eval preds.jsonl --mcq-key --judge "cache:results/judge_cache/qwen3-14b_v2*.jsonl"

# majority vote over the stored verdicts of three judges
ibench eval preds.jsonl --mcq-key --judge ensemble:judgeA,judgeB,judgeC \
    --judge-arg cache_dir=results/judge_cache
```

Verdicts are cached on disk by (question, reference, response). A verdict depends on
the judge model and on the grading prompt, so use one cache file per combination. A
second evaluation with the same cache calls the judge only for new pairs.

`--judge-prompt v2` (default, used in the paper) accepts a response that contains the
reference information and adds detail. `--judge-prompt v1` requires equivalence.

| `--judge-arg` for `api:` | Default |
|---|---|
| `base_url` | `IBENCH_JUDGE_BASE_URL`, then `OPENAI_BASE_URL`, then `https://api.openai.com/v1` |
| `api_key_env` | `IBENCH_JUDGE_API_KEY`, then `OPENAI_API_KEY` |
| `max_tokens` | 2048 |
| `reasoning_effort` | not sent |

## Add a judge

```python
# my_judge.py
from interactionbench import register_judge
from interactionbench.judges import CachedJudge

class MyJudge(CachedJudge):
    def __init__(self, model, cache_path=None, prompt_version="v2", **kwargs):
        super().__init__(cache_path, prompt_version)
        self.name = f"my-judge:{model}"
        ...                                     # load a model or open a client

    def _generate(self, prompt: str) -> str:
        ...                                     # the reply must contain "SCORE: 1" or "SCORE: 0"

@register_judge("my-judge")
def make(arg, **kwargs):
    return MyJudge(arg, **kwargs)
```

```bash
ibench eval preds.jsonl --mcq-key --plugin my_judge.py --judge my-judge:<model>
```

`CachedJudge` builds the grading prompt, parses the verdict and maintains the cache.
A judge that needs none of this can be any callable
`judge(question, reference, response) -> float`, registered the same way
([examples/custom_judge.py](../examples/custom_judge.py)).

## Compare judges

`analysis/judge_calibration.py` extracts the pairs to be judged, runs several judges on
them and reports their agreement with each other and with human labels.
