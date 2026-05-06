"""
Prepare LibriSpeech test-clean and test-other for evaluation with the
existing dual-metric pipeline.

Materialises WAV files in a clips directory and writes TSVs with the same
schema as Common Voice (`path`, `sentence`).

Usage:
    python prepare_librispeech.py \
        --output_root /scratch/vemotionsys/rmfrieske/datasets/librispeech_eval \
        --splits test.clean test.other
"""

import argparse
import csv
import os

import soundfile as sf
from datasets import load_dataset


def materialise_split(hf_split_name, out_root):
    """Download the split via HF datasets and write WAVs + TSV."""
    print(f"\n=== Materialising {hf_split_name} ===")
    ds = load_dataset("openslr/librispeech_asr", "clean" if "clean" in hf_split_name else "other",
                      split=hf_split_name, trust_remote_code=True)

    safe_name = hf_split_name.replace(".", "_")
    clips_dir = os.path.join(out_root, "clips", safe_name)
    os.makedirs(clips_dir, exist_ok=True)
    tsv_path = os.path.join(out_root, f"{safe_name}.tsv")

    n_written = 0
    with open(tsv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["path", "sentence"])

        for ex in ds:
            audio = ex["audio"]
            sr = audio["sampling_rate"]
            arr = audio["array"]
            uid = ex["id"]
            rel_path = f"{safe_name}/{uid}.wav"
            full_path = os.path.join(out_root, "clips", rel_path)
            if not os.path.exists(full_path):
                sf.write(full_path, arr, sr)
            writer.writerow([rel_path, ex["text"]])
            n_written += 1
            if n_written % 500 == 0:
                print(f"  wrote {n_written} / {len(ds)}")

    print(f"  -> {tsv_path}  ({n_written} samples)")
    print(f"  clips dir: {os.path.join(out_root, 'clips')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--splits", nargs="+", default=["test.clean", "test.other"])
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    for s in args.splits:
        materialise_split(s, args.output_root)

    print("\nDone. Use --clips_dir <output_root>/clips and --test_tsv <output_root>/test_clean.tsv")


if __name__ == "__main__":
    main()
