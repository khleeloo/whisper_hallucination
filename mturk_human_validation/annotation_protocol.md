# Frozen annotation protocol

## Primary endpoint

Human acoustic grounding of the displayed ASR hypothesis, judged from audio without access to the reference transcript or automatic scores.

Rating scale:

- **3** — mostly or fully supported by the audio;
- **2** — partially supported, but with substantial ASR errors;
- **1** — weakly supported; only a small part appears related;
- **0** — unsupported by the audible speech;
- **X** — cannot judge the audio.

For sample-level analysis, `0/1` are grouped as **human unsupported** and `2/3` as **human supported/partially supported**. `X` is excluded from the binary denominator. With three independent judgments, a sample is called human unsupported when a strict majority of valid judgments is `0/1`.

## Secondary endpoint

Worker-selected observable failure type:

1. ordinary speech-related recognition error;
2. fluent content not supported by the speech;
3. generic short/default output;
4. repeated words or phrases;
5. other/unclear.

The word *hallucination* is intentionally absent from the worker task.

## Blinding

Workers see only:

- randomized item order;
- a randomized sample ID carried in hidden form data;
- the audio clip;
- the ASR hypothesis.

Workers do not see model identity, noise condition, reference transcript, WER, Qwen/GPT-2 score, CTC support, automatic H label, stratum, or phenotype seed.

## Sampling

The default study contains 180 experimental items in six pre-specified strata of 30. Sampling seed is `20260824`. Severe-noise strata are balanced between full-noise amplitudes 0.50 and 0.75. Clean strict-H and severe controls are balanced across models where specified in `prepare_mturk_study.py`.

The automatic labels are frozen before human judgments. The sample must not be regenerated because of undesirable human outcomes.

## Quality control

Each HIT contains 10 experimental items and one QC item. Six QC items are generated from clean exact-recognition examples:

- 3 positive checks: clean audio paired with its exact hypothesis;
- 3 negative checks: clean audio paired with a low-token-overlap transcript from another utterance.

QC pass rule:

- positive check passes for grounding >=2;
- negative check passes for grounding <=1.

Default worker exclusion:

- at least 2 QC judgments and at least 2 failures; or
- at least 4 QC judgments and QC accuracy <75%.

Very fast assignments (<90 s for the default 11-item HIT) are flagged by default, not automatically excluded. Any change to exclusion thresholds should be decided before examining experimental-condition results and documented.

## Primary statistical comparison

The primary contrast is the sample-level human-unsupported rate among severe strict-Qwen candidates versus severe high-WER automatic non-H controls. Report numerator/denominator, proportion, and 95% confidence interval. Also report the risk difference and risk ratio descriptively.

Because sampling is stratified, pooled results must not be interpreted as population hallucination prevalence, sensitivity, or specificity.

## Agreement

Report ordinal Krippendorff's alpha for the 0–3 grounding ratings when available, plus the distribution of valid/cannot-judge responses. The analysis script computes alpha when the Python `krippendorff` package is installed.

## Secondary analyses

Report human-unsupported rate by:

- stratum;
- model;
- acoustic condition;
- clean strict-H candidates;
- decoder-default/repetition phenotype-control subset.

Report the majority phenotype distribution as descriptive evidence only. Do not use phenotype judgments to retune the automatic diagnostic thresholds.
