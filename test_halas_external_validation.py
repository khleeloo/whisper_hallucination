import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

MODELS = [
    "whisper_large_v2", "whisper_large_v3", "whisper_large_v3_turbo",
    "crisper_whisper", "canary", "canary_flash", "parakeet",
]


def span(text, label):
    return json.dumps([{"text": text, "start": 0, "end": len(text), "labels": [label]}])


def make_header():
    cols = ["audio_id", "audio_duration", "e22_reference_text", "corrected_reference_text"]
    for model in MODELS:
        cols += [f"{model}_prediction", f"{model}_label", f"{model}_hallucination_text",
                 f"{model}_hallucination_json"]
    cols += ["split"]
    return cols


def make_row(audio_id, ref, pred, label, js, split):
    d = {k: "" for k in make_header()}
    d.update({"audio_id": audio_id, "audio_duration": "2.0", "e22_reference_text": ref,
              "corrected_reference_text": ref, "split": split})
    for model in MODELS:
        d[f"{model}_prediction"] = pred
        d[f"{model}_label"] = label
        d[f"{model}_hallucination_json"] = js
    return d


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        inp, out = root / "toy.csv", root / "out"
        rows = [
            make_row("a.wav", "hello world", "hello world", "No hallucination", "[]", "train"),
            make_row("b.wav", "hello world", "bye bye bye bye bye", "Hallucination or looping",
                     span("bye bye bye", "Looping Hallucination"), "train"),
            make_row("c.wav", "good morning", "thank you", "Hallucination or looping",
                     span("thank you", "Hallucination"), "test"),
            make_row("d.wav", "good morning", "good morning", "No hallucination", "[]", "test"),
            make_row("e.wav", "good day", "yes yes yes yes", "Hallucination or looping",
                     span("yes yes yes", "Looping"), "test"),
            make_row("f.wav", "good day", "thank you", "Hallucination or looping",
                     span("thank you", "Hallucination"), "test"),
        ]
        with inp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=make_header())
            w.writeheader()
            w.writerows(rows)
        subprocess.run([sys.executable, "halas_external_validation.py", "--input_csv", str(inp),
                        "--output_dir", str(out)], check=True)
        assert (out / "halas_per_prediction.csv").exists()
        assert (out / "halas_association_tests.csv").exists()
        assert "Rep34 among human-looping outputs" in (out / "headline_summary.txt").read_text()
        print("HALAS smoke test passed")


if __name__ == "__main__":
    main()
