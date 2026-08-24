# MTurk human validation for ASR failure diagnosis

This folder prepares the blinded human-listening validation for the ASR hallucination paper. It is intentionally separate from the model-training/evaluation code.

## Scientific question

The primary question is whether listeners independently judge outputs selected by the frozen strict automatic criterion as weakly supported or unsupported by the audio more often than severe high-WER automatic controls. Workers are **not** shown model names, references, noise conditions, WER, LM scores, automatic labels, or the word *hallucination*.

The secondary question is whether listeners recognize the paper's observable failure phenotypes: ordinary speech-related error, fluent unsupported content, generic/default output, and repetition.

## Frozen default study

- 180 experimental items = 6 strata x 30.
- 3 independent assignments per HIT -> 540 experimental judgments.
- 18 HITs.
- Each HIT contains 10 experimental items + 1 QC item.
- Six reusable QC items are generated: 3 clean matched positive checks and 3 clean audio/unrelated-transcript negative checks.
- Sampling seed: `20260824`.
- Acoustic perturbation seed: `20260821`, matching the paper stress experiments.

The six strata are:

1. clean strict-Qwen candidates, balanced across the three models;
2. raw Whisper severe-noise strict-Qwen candidates;
3. adapted Whisper severe-noise strict-Qwen candidates;
4. SeamlessM4T-v2 severe-noise strict-Qwen candidates;
5. severe high-WER automatic non-H controls, balanced across models and noise 0.50/0.75;
6. phenotype controls: raw decoder-default collapse plus repetition-heavy adapted/Seamless outputs.

The high-WER control pool is **not** selected using CTC support. This avoids building the human-validation conclusion into the control sampling.

## 1. Prepare the study on SuperPod

From the repository root:

```bash
cd ~/whisper_hallucination
git pull
sbatch mturk_human_validation/slurm_prepare_mturk.sbatch
```

The job requests one GPU because this cluster requires GPU-backed SLURM jobs, although preparation itself is CPU/audio I/O work.

Default output:

```text
/scratch/vemotionsys/rmfrieske/whisper_hallucination/mturk_human_validation/
```

Expected files:

```text
audio/*.wav
private_sample_manifest.csv
attention_checks_PRIVATE.csv
public_items.csv
mturk_batch.csv
study_summary.json
```

**Never upload `private_sample_manifest.csv` or `attention_checks_PRIVATE.csv` to MTurk.** They contain references, model/condition labels, source paths, automatic metrics, and QC answers.

## 2. Inspect before publishing

Before any crowd release:

- confirm the selected audio may legally/ethically be exposed to external crowd workers;
- listen to a sample of each stratum;
- inspect all six QC examples;
- verify the WAV files are the same deterministic perturbations used in the paper;
- confirm no public filename or visible field contains model/condition/automatic-label information.

Do not alter automatic thresholds or resample based on desired human outcomes.

## 3. Host audio

The generated WAV files need stable HTTPS URLs. S3 is the simplest option. Example only:

```bash
aws s3 sync \
  /scratch/vemotionsys/rmfrieske/whisper_hallucination/mturk_human_validation/audio/ \
  s3://YOUR_BUCKET/mturk_asr_audio/
```

Use a hosting/access configuration permitted by the source-data license and your ethics approval. Do not make identifiable speech public merely for convenience if the source terms do not allow it.

Then regenerate only the batch URL fields (audio is unchanged):

```bash
python mturk_human_validation/set_audio_urls.py \
  --input /scratch/vemotionsys/rmfrieske/whisper_hallucination/mturk_human_validation/mturk_batch.csv \
  --base_url https://YOUR-HTTPS-BASE/mturk_asr_audio \
  --output /scratch/vemotionsys/rmfrieske/whisper_hallucination/mturk_human_validation/mturk_batch_READY.csv
```

Open several URLs in a browser before upload.

## 4. Create the MTurk project

Use a custom Survey/Other project. Paste `mturk_task_template.html` into the task HTML and upload `mturk_batch_READY.csv` as the batch input.

Recommended production settings:

- assignments per HIT: **3**;
- worker approval rate: **>=98%**;
- approved HITs: **>=1,000**;
- headphones/quiet-environment instruction shown prominently;
- allow replay;
- do not ask workers whether the output is a "hallucination".

Run the HTML first in MTurk Sandbox, then run a small paid pilot before releasing the complete batch.

## 5. Worker questions

Primary 4-point acoustic-grounding score:

- 3 = mostly or fully supported by the audio;
- 2 = partially supported, but with substantial ASR errors;
- 1 = weakly supported; only a small part appears related;
- 0 = unsupported by the audible speech;
- X = cannot judge the audio.

Secondary phenotype choice:

- ordinary speech-related recognition error;
- fluent content not supported by the speech;
- generic short/default output;
- repeated words or phrases;
- other/unclear.

The primary paper endpoint is acoustic grounding. Phenotype labels are secondary.

## 6. QC/exclusion rule

The analysis script uses the private QC key. A positive QC is counted as passed for grounding >=2; a negative QC is counted as passed for grounding <=1. The default worker-level exclusion is deliberately conservative:

- exclude if the worker has at least 2 QC judgments and fails at least 2; or
- exclude if the worker has at least 4 QC judgments and QC accuracy is below 75%.

Assignments completed implausibly quickly are flagged in the analysis output rather than silently discarded. Change exclusion thresholds only before examining experimental outcomes and record any change.

## 7. Analyze downloaded MTurk results

After downloading the MTurk batch results CSV:

```bash
python mturk_human_validation/analyze_mturk_results.py \
  --results MTurk_results.csv \
  --private_manifest /scratch/vemotionsys/rmfrieske/whisper_hallucination/mturk_human_validation/private_sample_manifest.csv \
  --qc_key /scratch/vemotionsys/rmfrieske/whisper_hallucination/mturk_human_validation/attention_checks_PRIVATE.csv \
  --output_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/mturk_human_validation/analysis
```

The script produces long-form worker judgments, QC summaries, sample-level majority aggregation, stratum/model summaries, and the primary strict-H-vs-high-WER-control contrast. It optionally reports ordinal Krippendorff alpha when the `krippendorff` Python package is installed.

## Interpretation

Because the study is deliberately stratified, do **not** use the pooled sample to estimate population hallucination prevalence, sensitivity, or specificity. The defensible claims are conditional, e.g. the human unsupported rate among sampled strict-H candidates versus the matched severe automatic-control group, plus model/phenotype-specific agreement.
