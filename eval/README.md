# F1rstAid Eval Harness

A fixed, versioned set of F-1 visa questions with known-correct facts, used to
measure the QA pipeline's accuracy so changes (model swap, retrieval tuning,
prompt changes) can be judged by a number instead of a vibe.

This is **not** part of the pytest suite (`tests/`) -- it makes real OpenAI
API calls and costs money per run. Run it manually when you want a baseline
or want to check whether a change actually helped.

## Running

```bash
export OPENAI_API_KEY=sk-...
python eval/run_eval.py --label baseline
```

Useful flags:

```bash
python eval/run_eval.py --label after-gpt4o --model gpt-4o-mini \
    --note "swapped gpt-3.5-turbo for gpt-4o-mini"
python eval/run_eval.py --label after-k5 --k 5 --note "search_k 3 -> 5"
```

`--note` is free text describing what changed for this run -- write it, it's
the difference between a history table that says "why did accuracy jump
here?" and one that answers itself.

Each run writes a timestamped JSON file to `eval/results/` (git-ignored --
these are large, local, per-run artifacts) and appends a compact summary
line to `eval/results/history.jsonl` (committed -- this is the durable
accuracy-over-time ledger). It also prints a per-category accuracy table,
abstention/false-decline/false-accept counts, latency percentiles, and a
list of failing question IDs to the console.

If your working tree has uncommitted changes when you run the eval, you'll
get a warning: the run's accuracy can't be attributed to a specific commit,
so the history entry is only useful as a rough check, not a tracked data
point. Commit first, then run the eval, if you want it to count.

## Tracking accuracy over time

```bash
python eval/history.py                  # full trend table
python eval/history.py --last 5         # just the most recent runs
python eval/history.py --category eligibility_multi_condition
```

Each row shows the git commit the run was made against, the accuracy,
the delta from the previous run, abstention rate, and p50 latency -- plus
the `--note` you wrote, so the table doubles as a changelog of what you
tried and whether it helped:

```
Timestamp         Label               Commit       accuracy     Delta   Abstain   p50 ms
20260810T140000Z  baseline            a1b2c3d          62.5%                12.5%      950
20260811T091500Z  gpt4o-mini          e4f5a6b          78.1%    +15.6%       6.2%     1400
  note: swapped gpt-3.5-turbo for gpt-4o-mini
```

Use `--category` to watch a single category's trend in isolation -- e.g. if
you're specifically iterating on multi-condition eligibility logic, the
overall accuracy number can hide progress/regression on that category
underneath unrelated noise in the others.

## Comparing two runs in detail

```bash
python eval/compare_runs.py eval/results/<baseline>.json eval/results/<after>.json
```

Prints the accuracy delta and flags any *individual question* that flipped
from correct to incorrect (a regression) or vice versa (an improvement) --
more granular than the history table, useful when you need to know exactly
which questions a change affected. Requires the local per-run JSON files
(not committed), so this only works on runs from your own machine.

## Dataset (`dataset.json`)

32 questions across six categories:

- **factual_lookup** (8): single-fact retrieval questions (form numbers,
  durations). The baseline these should never fail on.
- **eligibility_multi_condition** (8): questions that require correctly
  applying a rule with more than one condition (e.g. STEM OPT + E-Verify
  employer requirement).
- **timeline_math** (6): questions requiring date/day-count arithmetic on top
  of retrieved facts, not just lookup.
- **edge_case** (4): genuinely ambiguous or high-stakes scenarios where the
  *correct* behavior is hedging and pointing to the DSO rather than a
  confident guess. Scored on whether the answer hedges appropriately.
- **out_of_scope** (4): questions with nothing to do with F-1 status. Scored
  on whether the relevance gate correctly declines them.
- **help** (2): meta questions that should hit the canned help responses.

### Scoring

Each question has `expected_keypoints` (all must appear in the answer,
case-insensitive substring match) and/or `expected_keypoints_any` (at least
one must appear -- used when multiple phrasings of the same fact are
acceptable). `out_of_scope` questions are scored on whether the response
contains the relevance-check decline marker instead.

This is a cheap, deterministic proxy for correctness, not a real grader --
it will produce **false positives** (answer contains the right keyword but
uses it in a wrong or contradictory sentence) and **false negatives**
(answer is correct but phrases the fact differently than expected). Treat
the accuracy number as a signal to prioritize investigation, and always
read the failing (and a sample of passing) `answer` fields in the result
JSON before concluding a change helped or hurt.

### Metrics beyond raw accuracy

- **Abstention rate** -- fraction of in-scope (non `out_of_scope`/`help`)
  questions where the model said it doesn't have enough information.
  Excludes `out_of_scope`/`help` from the denominator since abstaining
  there isn't meaningful. A rate near 0% is not automatically good: it can
  mean the model is confidently answering things it should be hedging on.
  Watch it alongside accuracy, not instead of it.
- **False decline count** -- in-scope questions the relevance gate wrongly
  rejected as off-topic. A trust/UX failure distinct from a wrong-fact
  failure: the user gets no answer at all.
- **False accept count** -- `out_of_scope` questions the relevance gate
  failed to reject. The gate is keyword-based (see `f1rstaid.py`'s
  `_F1_KEYWORDS`), so watch this number specifically if you touch that list.
- **Latency (avg/p50/p95/max)** -- wall-clock time per `get_answer()` call.
  Mainly useful as a baseline before adding retrieval steps (reranking,
  multi-query, routing) that will add latency -- lets you judge whether an
  accuracy gain from a heavier pipeline is worth the added wait.

Not included yet, deliberately: token/cost-per-query (needs plumbing
through LangChain's usage metadata, which the current chain doesn't
surface) and a real hallucination-detection pass (needs a second LLM call
to check the answer against retrieved context, meaningfully doubling per-run
cost). Add these once the current metrics show they're the bottleneck,
not before.

### Run-to-run noise (why eval defaults to temperature=0)

`run_eval.py` defaults `--temperature` to `0`, *not* the production app's
`0.2` default. Running the exact same config (model, k, dataset) twice at
0.2 was observed to flip 2-4 questions between runs purely from sampling
noise -- e.g. `fact-002` and `elig-005` failed on one run and passed on the
next, and vice versa for `fact-003`/`elig-001`, with no code change in
between. An 87.5% vs. 93.8% swing at identical config would look like a
regression or improvement if you only ran it once each time; it's actually
noise. Pass `--temperature 0.2` explicitly if you specifically want to
measure how the model behaves at the production temperature, but treat a
single such run as unreliable -- average a few runs before concluding
anything from it.

### Keeping the dataset honest

`expected_keypoints` were deliberately chosen as stable, long-standing
regulatory facts (day counts, form numbers) rather than dollar amounts or
anything that changes yearly. Even so, immigration rules do change --
`last_verified` in `dataset.json` records when the expected answers were
last checked against current USCIS/DHS guidance. Re-verify before trusting
a big accuracy swing, and update `last_verified` when you do.
