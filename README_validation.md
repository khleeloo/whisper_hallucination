# Whisper Cross-Model Validation Metrics — Documentation

## Purpose

Evaluate whether Whisper reproduces the same ASR failure-mode patterns found in fairseq experiments under structured label noise. The core hypothesis:

- **UR (Unique-Repeat)**: Should produce hallucination-like outputs — high fluency, low accuracy.
- **RR (Repeat-Repeat)**: Should produce increased n-gram repetition / oscillation.
- **UU / RU**: Should show weaker or intermediate failure patterns.

Evaluation focuses on **relative change from Whisper baseline**, not absolute WER.

---

## Model Conditions

| Condition | Description |
|---|---|
| `base` | Clean training (Whisper LoRA on Common Voice EN subset, 120K utterances) |
| `UU` | Unique audios + unique unrelated sentences (8% noise) |
| `UR` | Unique audios + 10 repeated sentences (8% noise) |
| `RR` | 10 repeated audio-text pairs (8% noise) |
| `RU` | 10 repeated audios + unique sentences (8% noise) |

Base model: `openai/whisper-large-v3` with LoRA (`r=16, alpha=32`) fine-tuned for 5 epochs on Common Voice EN.

---

## Text Normalization

Before computing WER, WAcc, repetition, and cosine similarity:

1. Lowercase
2. Remove Whisper special tokens (`<|...|>`)
3. Remove punctuation (`[^\w\s]`)
4. Normalize whitespace (collapse multiple spaces)

**Important**: Repeated words are **not** collapsed — repetition is one of the target failure modes.

---

## Metrics

### Metric 1: WAcc (Word Accuracy)

```
WER = (S + D + I) / N     (jiwer)
WAcc = max(0.0, 1.0 - WER)
```

WAcc is the main lexical accuracy metric. Per-utterance S/D/I counts are also saved.

### Metric 2: Semantic Cosine Similarity

Uses `sentence-transformers/all-MiniLM-L6-v2` to embed reference and hypothesis.

```python
cosine_sim = dot(emb_ref, emb_hyp) / (||emb_ref|| * ||emb_hyp||)
```

This is a supporting semantic metric, not the main hallucination criterion.

### Metric 3-4: Fluency / Sentence Plausibility

Two language models:

| LM | Role | Model |
|---|---|---|
| Weak LM | Baseline fluency scorer | `gpt2` (124M params) |
| Strong LM | Robustness check | `Qwen/Qwen3-1.7B` |

For each hypothesis and reference:

```python
nll = model(input_ids, labels=input_ids).loss
sentence_score = exp(-nll.item())      # higher = more fluent
normalized_score = hyp_score / (ref_score + 1e-8)  # clipped to [0, 1]
```

### Metric 5: Normalized Sentence Score

```
norm_score = min(1.0, max(0.0, hyp_sentence_score / ref_sentence_score))
```

Computed separately for weak and strong LMs. This is the main fluency metric.

### Metric 6: Hallucination-Like Rate

Operational definition: low WAcc + high normalized sentence score.

**Thresholds** (derived from Whisper baseline distribution):

```python
wacc_threshold = mean(WAcc of Whisper-Base)
fluency_threshold = mean(normalized_sentence_score of Whisper-Base)

hallucination_like = wacc < wacc_threshold AND norm_score > fluency_threshold
```

**Strict variant**:

```python
wacc_q25 = Q25(WAcc of Whisper-Base)
fluency_median = median(normalized_sentence_score of Whisper-Base)

hallucination_like_strict = wacc <= wacc_q25 AND norm_score >= fluency_median
```

Expected: UR should have the largest hallucination-like rate increase.

### Metric 7: N-gram Repetition

For n = 2, 3, 4:

```python
repetition_count_n = sum(count - 1 for each ngram where count > 1)
has_repetition_n = repetition_count_n >= 2
```

Expected: RR should have the largest trigram + four-gram repetition rate increase.

### Metric 8: Copied-Label / Memorization Check

For UR hallucination-like outputs, compare hypotheses against the 10 repeated noisy labels from training using TF-IDF cosine similarity.

```python
copied_label_rate = proportion of hallucination-like outputs with max TF-IDF cosine > 0.7
```

Expected: UR should show high copied-label rate.

---

## Output Files

| File | Description |
|---|---|
| `per_utterance_metrics_whisper.csv` | Per-utterance metrics for all conditions (~30 columns) |
| `aggregate_metrics_whisper.csv` | Per-condition means and rates |
| `baseline_relative_deltas_whisper.csv` | Deltas from Whisper-Base for each condition |
| `cross_model_comparison.csv` | Scaffold for Whisper × Fairseq comparison (TBD fairseq data) |
| `fluency_scorer_robustness.csv` | Hallucination rates computed with each LM independently |
| `copied_label_analysis.csv` | Per-condition copied-label rates |
| `qualitative_examples.csv` | Top-5 examples per failure mode |
| `failure_mode_space.png/pdf` | Figure 1: WAcc × Fluency scatter |
| `relative_change_from_baseline.png/pdf` | Figure 2: Delta bar chart |
| `repetition_rates.png/pdf` | Figure 3: N-gram repetition bar chart |
| `dose_response.png/pdf` | Figure 4: Dose-response (if sweep data available) |

---

## Usage

### Full pipeline (SLURM):

```bash
cd /home/rmfrieske/whisper_hallucination
bash run_validation_pipeline.sh
```

This submits two jobs:
1. `slurm_eval_validation.sbatch` — GPU array job (5 models × 1 GPU)
2. `slurm_analysis_validation.sbatch` — CPU analysis (dependency)

### Manual (per-model):

```bash
python evaluate_whisper_validation.py \
    --model_dir /scratch/.../base/final \
    --base_model openai/whisper-large-v3 \
    --test_tsv /scratch/.../test.tsv \
    --clips_dir /scratch/.../clips \
    --output_dir /scratch/.../eval_validation \
    --config_name base \
    --noise_condition base \
    --noise_ratio 0.0 \
    --lm_models gpt2 Qwen/Qwen3-1.7B
```

### Analysis + plots:

```bash
python analyze_validation.py \
    --per_utterance_dir /scratch/.../eval_validation \
    --noisy_labels_dir /scratch/.../datasets/whisper_hallucination \
    --output_dir /scratch/.../eval_validation

python make_plots.py \
    --per_utterance_csv /scratch/.../eval_validation/per_utterance_metrics_whisper.csv \
    --aggregate_csv /scratch/.../eval_validation/aggregate_metrics_whisper.csv \
    --output_dir /scratch/.../plots_validation
```

---

## Expected Results (Hypothesis)

| Pattern | Metric | UR | RR |
|---|---|---|---|
| Hallucination-like | WAcc drop | **Largest** | Moderate |
| Hallucination-like | Fluency (norm score) | Preserved or elevated | Moderate |
| Hallucination-like | Hall. rate increase | **Largest** | Moderate |
| Repetition | Trigram rep rate | Low | **Highest** |
| Repetition | Four-gram rep rate | Low | **Highest** |
| Copied-label | TF-IDF similarity | High | Low/Moderate |

### Success Criteria

If both Whisper and fairseq show the same relative failure-mode trends:
> The failure modes are not artifacts of the original fairseq model, but reflect broader behavior of sequence-to-sequence ASR under structured label noise.

---

## Dependencies

```
torch >= 2.0
torchaudio
transformers >= 4.40
peft >= 0.14
jiwer
sentence-transformers
pandas
numpy
matplotlib
scikit-learn
datasets (for LibriSpeech prep only)
```

---

## Limitations

- Single test set (Common Voice EN test); cross-domain (LibriSpeech) is a planned follow-up
- Single base model (Whisper Large v3); model scale ablation is future work
- Copied-label analysis uses TF-IDF rather than embedding similarity — adequate for exact/pseudo-exact label copying detection
- Noise ratios evaluated: 8% (main) + sweeps at 1-50% (if available)
