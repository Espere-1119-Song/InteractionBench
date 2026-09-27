# Metrics

All scores are on a 0 to 100 scale. All times are stream seconds. Wall-clock latency is
reported for reference and never enters a score.

## Input

Per item, a time-ordered list of emissions: `{"t": <stream seconds>, "content": <text>}`.
Emissions with empty content are dropped before scoring.

## Accuracy

| Item | Accuracy |
|---|---|
| has a multiple-choice form (`--mcq-key`) | 100 if the matched response picks the correct option, else 0. A letter (`B`, `(B)`, `Answer: B`) or the option text is accepted. |
| free-form, single answer | the judge's verdict on the matched response against the reference. Without a judge: the maximum of exact match, token F1 and numeric match. |
| counting | 100 if the last number the system said equals the final count, else 0 |
| commentary and belief revision (`LCG`, `BRC`) | the stream is cut into segments at the reference times. The text emitted inside a segment is judged against the reference of that segment. Accuracy is the mean over all segments; a segment without emissions scores 0. |

## Decision timing

Each reference answer `n` has a standard response time `r*_n`. For A-type items `r*` is
the question time. The windows `W_n = [r*_n, r*_{n+1})` partition the stream, and event
`n` is matched by the first response inside its window.

**Pre-anchor tolerance.** With tolerance `tau` (default 1 s) every window starts `tau`
earlier: a response in `[r*_n - tau, r*_n)` is the response to event `n` with delay 0.
`--pre-tol 0` disables the tolerance.

**Timing Accuracy**

```
TA = 100/N * sum_n max(0, 1 - d_n / Delta)        d_n = t_match(n) - r*_n, floored at 0
```

`Delta` is 5 s by default. An event without a matched response contributes 0, so
missing events lowers TA.

**Silence Compliance**

```
SC = 100 * max(0, 1 - |V| / max(N, 1))
```

`V` contains

- premature responses: before the first window,
- redundant responses: inside a window that already has its match,
- spurious responses: any response on an item that requires silence. With the content
  gate enabled (`--use-gate`, an ablation, off in the paper protocol), a response whose
  content scores below the threshold is also spurious instead of becoming the match.

On an item that requires a response, SC is defined only if the system responded at
least once. A system that never speaks therefore gets no SC credit on such items.

## Item score

| Item family | Total |
|---|---|
| A-type question, B-type trigger, counting | `mean(Accuracy, harmonic_mean(TA, SC))` |
| commentary and belief revision | Accuracy |
| item that requires silence | SC |

If SC is undefined (no response on an item that requires one), the timing term is TA
alone, which is 0 in that case.

## Aggregation

The reported score of a run is the mean of the item scores. Each reported field is
averaged over the items where it is defined. `summary.json` contains the overall
summary and the same summary by capability, capability group, time type, interaction
type, range length and item family. `records.jsonl` contains one record per item.

## Diagnostics in the output

| Field | Meaning |
|---|---|
| `v_premature`, `v_redundant`, `v_spurious` | violations per item |
| `miss_rate` | share of events without a matched response |
| `delay_mean_s` | mean delay of matched responses |
| `response_precision` | matched responses divided by all responses |
| `segment_coverage` | share of segments with at least one emission |
| `covered_accuracy` | accuracy over the covered segments only |
| `memory_span_s` | A-type: question time minus evidence time |
| `poll_latency_mean`, `realtime_factor` | wall-clock latency and its ratio to a 0.2 s budget; reference only |
| `emissions_per_min`, `chatter_rate` | how often the system spoke |

## Paper protocol

```
ibench eval <preds> --mcq-key --judge hf:Qwen/Qwen3-14B
```

which means: answer key v4, judge Qwen3-14B with grading prompt v2, `Delta` = 5 s,
pre-anchor tolerance 1 s, content gate off. The configuration is stored in the
`config` block of `summary.json`.
