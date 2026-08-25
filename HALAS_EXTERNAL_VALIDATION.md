# HALAS external validation

This analysis evaluates the paper's structural failure diagnostics against the human-annotated HALAS benchmark without rerunning ASR. HALAS provides naturally occurring hallucinations on Earnings22 speech, corrected references, predictions from multiple ASR systems, and span labels distinguishing `Hallucination`, `Looping`, and `Looping Hallucination`.

## Protocol

- HALAS repository commit: `5c9c8b18fe67224dc10a801884cb5faa4d64b4fb`.
- Primary analysis: official HALAS test split.
- Systems: Whisper large-v2, Whisper large-v3, Whisper large-v3-turbo, CrisperWhisper, Canary, Canary Flash, and Parakeet.
- Phi-4 and Granite are excluded because the HALAS dataset card notes that they were trained on Earnings22 and are outside the scope of the corresponding HALAS paper.
- No ASR inference or audio perturbation is performed.
- `Rep34` is the incidence of at least one repeated trigram or four-gram in the ASR hypothesis.
- Cross-utterance recurrence is computed separately within each model and split using normalized exact hypotheses.
- WER is computed against the human-corrected HALAS reference where available.

HALAS intentionally oversamples difficult/high-disagreement examples, so observed rates are not estimates of real-world hallucination prevalence.

## Official test-split results

The analysis contains **4,990 predictions from seven ASR systems**. Human span annotations identify hallucination in 22.9% of predictions and looping in 1.2%.

### Rep34 validates the repetition failure axis

- Rep34 among human-labeled looping outputs: **69.0%**.
- Rep34 among non-looping outputs: **1.76%**.
- Fisher exact odds ratio: **123.8**.
- Fisher exact p-value: **6.56e-53**.
- Rep34 discrimination of human looping: **ROC-AUC 0.836**, average precision **0.221**.

The phenotype breakdown is also consistent with the intended interpretation of Rep34:

- ordinary hallucination: 2.43% Rep34;
- no annotated hallucination/looping: 1.57%;
- looping: 57.7%;
- looping hallucination: 90.0%;
- mixed hallucination + looping: 75.0%.

This provides external human-label evidence that Rep34 measures a distinct repetition/looping failure component rather than functioning as a generic hallucination score.

### Cross-utterance recurrence does not generalize as a generic hallucination detector

On natural HALAS speech, exact-output recurrence is only weakly associated with human hallucination:

- top-10 recurring-output membership among hallucinations: **5.77%**;
- top-10 recurring-output membership without hallucination: **4.60%**;
- odds ratio: **1.27**, p = **0.117**;
- exact-output recurrence ROC-AUC for human hallucination: **0.508**.

This negative result is important for interpretation. Cross-utterance concentration should remain framed as a stress-induced **decoder-collapse phenotype**, not as a universal hallucination detector on natural speech.

### WER remains informative but does not identify failure type

WER against the corrected HALAS reference reaches **ROC-AUC 0.761** for human hallucination (average precision 0.496). This confirms that lexical error is informative about naturally occurring unsupported content, while the Rep34 result shows that structural diagnostics add information about *which failure regime* is present.

## Paper-facing interpretation

The HALAS result supports the paper's failure-regime framing rather than a claim that every diagnostic axis is a universal hallucination detector. In particular, Rep34 has strong external validity for human-annotated looping, while cross-utterance concentration remains specific to the decoder-default collapse observed under controlled severe acoustic stress. This separation is consistent with treating hallucination-like ASR failure as heterogeneous rather than compressing all behavior into one scalar score.

## Reproduction

Run:

```bash
python halas_external_validation.py --output_dir halas_external_validation_outputs
```

The script downloads the pinned HALAS CSV and writes:

- `halas_per_prediction.csv`
- `halas_model_summary.csv`
- `halas_phenotype_summary.csv`
- `halas_association_tests.csv`
- `headline_summary.txt`
- `provenance.json`

A GitHub Actions workflow also runs the smoke test and full HALAS analysis and uploads the outputs as an artifact.
