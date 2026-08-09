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
python eval/run_eval.py --label after-gpt4o --model gpt-4o-mini
python eval/run_eval.py --label after-k5 --k 5
```

Each run writes a timestamped JSON file to `eval/results/` and prints a
per-category accuracy table plus a list of failing question IDs to the
console.

## Comparing two runs

```bash
python eval/compare_runs.py eval/results/<baseline>.json eval/results/<after>.json
```

Prints the accuracy delta and flags any question that flipped from correct
to incorrect (a regression) or vice versa (an improvement).

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

### Keeping the dataset honest

`expected_keypoints` were deliberately chosen as stable, long-standing
regulatory facts (day counts, form numbers) rather than dollar amounts or
anything that changes yearly. Even so, immigration rules do change --
`last_verified` in `dataset.json` records when the expected answers were
last checked against current USCIS/DHS guidance. Re-verify before trusting
a big accuracy swing, and update `last_verified` when you do.
