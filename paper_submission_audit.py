#!/usr/bin/env python3
"""Final source-level audit for the ICASSP manuscript.

This complements ``paper_protocol_audit.py`` and ``paper_numerical_audit.py``.
It checks source completeness and LaTeX hygiene without rerunning ASR inference:

* every ``\\bibliography{...}`` file exists;
* every cited BibTeX key resolves and BibTeX keys are unique;
* every ``\\ref``/``\\eqref`` target exists and labels are unique;
* no obvious TODO/FIXME/TBD/XXX placeholders remain;
* the manuscript still uses the anonymous author block;
* optionally, the paper compiles cleanly in a temporary directory and the PDF
  page count can be checked against a user-supplied limit.

Examples
--------
Static source audit::

    python paper_submission_audit.py --tex paper_icassp.tex

Compile as well (requires pdflatex + bibtex)::

    python paper_submission_audit.py --tex paper_icassp.tex --compile

Optionally enforce a PDF page limit::

    python paper_submission_audit.py --tex paper_icassp.tex --compile --max-pages 5

The JSON report is written under the scratch experiment root by default.  The
script exits non-zero if a required check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parent
ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DEFAULT_REPORT = ROOT / "paper_submission_audit.json"


def strip_comments(text: str) -> str:
    """Remove unescaped LaTeX comments while preserving escaped percent signs."""
    out: List[str] = []
    for line in text.splitlines():
        cut = None
        for i, ch in enumerate(line):
            if ch != "%":
                continue
            # Count immediately preceding backslashes.  An even count means the
            # percent sign is not escaped and starts a LaTeX comment.
            n_bs = 0
            j = i - 1
            while j >= 0 and line[j] == "\\":
                n_bs += 1
                j -= 1
            if n_bs % 2 == 0:
                cut = i
                break
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def split_csv_args(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        out.extend(x.strip() for x in value.split(",") if x.strip())
    return out


def bibliography_files(tex: str, base: Path) -> List[Path]:
    names = split_csv_args(re.findall(r"\\bibliography\s*\{([^}]*)\}", tex))
    paths: List[Path] = []
    for name in names:
        p = Path(name)
        if p.suffix.lower() != ".bib":
            p = p.with_suffix(".bib")
        if not p.is_absolute():
            p = base / p
        paths.append(p)
    return paths


def cited_keys(tex: str) -> List[str]:
    # Covers cite, citep, citet, citeauthor, etc., while avoiding unrelated
    # commands beginning with the same characters.
    args = re.findall(r"\\cite[a-zA-Z*]*\s*(?:\[[^]]*\]\s*)*\{([^}]*)\}", tex)
    return split_csv_args(args)


def bib_keys(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        m.group(1).strip()
        for m in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", text, flags=re.I)
    ]


def labels(tex: str) -> List[str]:
    return [x.strip() for x in re.findall(r"\\label\s*\{([^}]*)\}", tex) if x.strip()]


def refs(tex: str) -> List[str]:
    found: List[str] = []
    for pattern in [r"\\ref\s*\{([^}]*)\}", r"\\eqref\s*\{([^}]*)\}", r"\\pageref\s*\{([^}]*)\}"]:
        found.extend(x.strip() for x in re.findall(pattern, tex) if x.strip())
    return found


def placeholder_hits(tex: str) -> List[str]:
    hits: List[str] = []
    patterns = {
        "TODO": r"\bTODO\b",
        "FIXME": r"\bFIXME\b",
        "TBD": r"\bTBD\b",
        "XXX": r"\bXXX\b",
        "question_marks": r"\?\?+",
        "citation_placeholder": r"\[(?:REF|CITE|CITATION)\]",
    }
    for name, pat in patterns.items():
        if re.search(pat, tex, flags=re.I):
            hits.append(name)
    return hits


def page_count(pdf: Path) -> int | None:
    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        return None
    proc = subprocess.run(
        [pdfinfo, str(pdf)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        return None
    m = re.search(r"^Pages:\s*(\d+)\s*$", proc.stdout, flags=re.M)
    return int(m.group(1)) if m else None


def run_cmd(cmd: Sequence[str], cwd: Path, env: Dict[str, str]) -> Dict[str, object]:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "cmd": list(cmd),
        "returncode": int(proc.returncode),
        "output_tail": proc.stdout[-12000:],
    }


def compile_audit(tex_path: Path, bibs: Sequence[Path], max_pages: int | None) -> Dict[str, object]:
    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if not pdflatex or not bibtex:
        return {
            "requested": True,
            "ok": False,
            "error": "pdflatex and bibtex are required for --compile",
            "pdflatex": pdflatex,
            "bibtex": bibtex,
        }

    with tempfile.TemporaryDirectory(prefix="icasppaper_") as td:
        work = Path(td)
        shutil.copy2(tex_path, work / tex_path.name)
        for bib in bibs:
            if bib.exists():
                shutil.copy2(bib, work / bib.name)

        stem = tex_path.stem
        env = os.environ.copy()
        # Allow TeX to find local repo resources if the manuscript later gains
        # an input/style file that is not copied above.
        env["TEXINPUTS"] = f"{REPO}{os.pathsep}" + env.get("TEXINPUTS", "")
        env["BIBINPUTS"] = f"{REPO}{os.pathsep}" + env.get("BIBINPUTS", "")

        steps: List[Dict[str, object]] = []
        steps.append(run_cmd([pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name], work, env))
        if steps[-1]["returncode"] == 0:
            steps.append(run_cmd([bibtex, stem], work, env))
        if all(step["returncode"] == 0 for step in steps):
            steps.append(run_cmd([pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name], work, env))
        if all(step["returncode"] == 0 for step in steps):
            steps.append(run_cmd([pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name], work, env))

        pdf = work / f"{stem}.pdf"
        log = work / f"{stem}.log"
        log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        pages = page_count(pdf) if pdf.exists() else None

        undefined_citations = bool(re.search(r"Citation .* undefined|There were undefined citations", log_text, flags=re.I))
        undefined_refs = bool(re.search(r"Reference .* undefined|There were undefined references", log_text, flags=re.I))
        rerun_warning = bool(re.search(r"Rerun to get cross-references right|Label\(s\) may have changed", log_text, flags=re.I))
        overfull = len(re.findall(r"Overfull \\hbox", log_text))
        compile_ok = bool(pdf.exists()) and all(step["returncode"] == 0 for step in steps)
        compile_ok = compile_ok and not undefined_citations and not undefined_refs and not rerun_warning
        page_ok = True if max_pages is None or pages is None else pages <= max_pages

        return {
            "requested": True,
            "ok": bool(compile_ok and page_ok),
            "steps": steps,
            "pdf_created": pdf.exists(),
            "pages": pages,
            "max_pages": max_pages,
            "page_limit_ok": page_ok,
            "undefined_citations": undefined_citations,
            "undefined_references": undefined_refs,
            "rerun_warning": rerun_warning,
            "overfull_hboxes": overfull,
            "log_tail": log_text[-12000:],
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tex", type=Path, default=REPO / "paper_icassp.tex")
    p.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--compile", action="store_true", help="also run pdflatex/bibtex in a temporary directory")
    p.add_argument("--max-pages", type=int, default=None, help="optional PDF page limit; only used with --compile")
    args = p.parse_args()

    tex_path = args.tex.resolve()
    if not tex_path.exists():
        raise FileNotFoundError(tex_path)
    raw = tex_path.read_text(encoding="utf-8")
    tex = strip_comments(raw)
    base = tex_path.parent

    bibs = bibliography_files(tex, base)
    missing_bibs = [str(p) for p in bibs if not p.exists()]
    cite_list = cited_keys(tex)
    cite_set = set(cite_list)

    key_sources: Dict[str, List[str]] = defaultdict(list)
    for bib in bibs:
        if not bib.exists():
            continue
        for key in bib_keys(bib):
            key_sources[key].append(str(bib))
    available_keys = set(key_sources)
    unresolved_citations = sorted(cite_set - available_keys) if not missing_bibs else []
    duplicate_bib_keys = {k: v for k, v in sorted(key_sources.items()) if len(v) > 1}

    label_list = labels(tex)
    ref_list = refs(tex)
    label_counts = Counter(label_list)
    duplicate_labels = sorted(k for k, n in label_counts.items() if n > 1)
    undefined_refs = sorted(set(ref_list) - set(label_list))

    placeholders = placeholder_hits(tex)
    anonymous_authors = bool(re.search(r"\\IEEEauthorblockN\s*\{\s*Anonymous Authors\s*\}", tex))

    static_ok = not any(
        [
            missing_bibs,
            unresolved_citations,
            duplicate_bib_keys,
            duplicate_labels,
            undefined_refs,
            placeholders,
        ]
    ) and anonymous_authors

    report: Dict[str, object] = {
        "tex": str(tex_path),
        "static": {
            "ok": static_ok,
            "bibliography_files": [str(p) for p in bibs],
            "missing_bibliography_files": missing_bibs,
            "citation_occurrences": len(cite_list),
            "unique_citation_keys": len(cite_set),
            "unresolved_citation_keys": unresolved_citations,
            "duplicate_bibtex_keys": duplicate_bib_keys,
            "labels": len(label_list),
            "references": len(ref_list),
            "duplicate_labels": duplicate_labels,
            "undefined_reference_targets": undefined_refs,
            "placeholder_hits": placeholders,
            "anonymous_author_block": anonymous_authors,
        },
    }

    if args.compile:
        if missing_bibs:
            report["compile"] = {
                "requested": True,
                "ok": False,
                "error": "compile skipped because bibliography files are missing",
            }
        else:
            report["compile"] = compile_audit(tex_path, bibs, args.max_pages)
    else:
        report["compile"] = {"requested": False}

    compile_ok = True
    if args.compile:
        compile_ok = bool(report["compile"].get("ok", False))  # type: ignore[union-attr]
    overall_ok = bool(static_ok and compile_ok)
    report["status"] = "PASS" if overall_ok else "FAIL"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== ICASSP SUBMISSION SOURCE AUDIT ===")
    print(f"status: {report['status']}")
    print(f"tex: {tex_path}")
    print(f"bibliography files: {len(bibs)}; missing: {len(missing_bibs)}")
    print(f"citation keys: {len(cite_set)}; unresolved: {len(unresolved_citations)}")
    print(f"labels: {len(label_list)}; undefined refs: {len(undefined_refs)}")
    print(f"placeholders: {placeholders or 'none'}")
    print(f"anonymous author block: {'PASS' if anonymous_authors else 'FAIL'}")
    if missing_bibs:
        for path in missing_bibs:
            print(f"FAIL: missing bibliography file: {path}")
    for key in unresolved_citations:
        print(f"FAIL: unresolved citation key: {key}")
    for key, sources in duplicate_bib_keys.items():
        print(f"FAIL: duplicate BibTeX key {key}: {sources}")
    for key in duplicate_labels:
        print(f"FAIL: duplicate label: {key}")
    for key in undefined_refs:
        print(f"FAIL: undefined reference target: {key}")
    if placeholders:
        print(f"FAIL: placeholder markers found: {', '.join(placeholders)}")
    if args.compile:
        comp = report["compile"]  # type: ignore[assignment]
        print(f"compile: {'PASS' if comp.get('ok') else 'FAIL'}")  # type: ignore[union-attr]
        if comp.get("pages") is not None:  # type: ignore[union-attr]
            print(f"pages: {comp.get('pages')}" + (f" / max {args.max_pages}" if args.max_pages else ""))  # type: ignore[union-attr]
        if comp.get("overfull_hboxes") is not None:  # type: ignore[union-attr]
            print(f"overfull hboxes: {comp.get('overfull_hboxes')}")  # type: ignore[union-attr]
        if comp.get("error"):  # type: ignore[union-attr]
            print(f"FAIL: {comp.get('error')}")  # type: ignore[union-attr]
    print(f"Report: {args.output}")

    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
