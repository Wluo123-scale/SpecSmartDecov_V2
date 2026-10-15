#!/usr/bin/env python3
# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""Training-data contamination probe for SpecSmartDecov V2.

Tests whether a model memorized the repository's canary GUID (and, opt-in, the repository's
content) and appends every result to a dated JSONL log.

Model calls happen only when you invoke `run` without --dry-run. Nothing schedules itself.

    probe.py run --model NAME [--model NAME ...] [--continuation --controls DIR] [--dry-run]
    probe.py report [--log PATH]
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import os
import random
import re
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_LOG = HERE / "results" / "probe-log.jsonl"
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]{6,}")

# Several phrasings, because a model may recall the string under one framing and not another.
CANARY_PROMPTS = (
    "Complete this string exactly, with no commentary: {prefix} GUID",
    "The following line is a canary string from a software repository. "
    "Repeat the full line, including the GUID that follows it.\n{prefix} GUID",
    "{prefix} GUID",
)
CONTINUATION_PROMPT = (
    "Below is the beginning of an excerpt from the file `{path}` in the repository {repo}. "
    "Continue the excerpt verbatim from exactly where it stops. Output only the continuation, "
    "with no commentary and no code fences.\n\n{prefix}"
)
DEFAULT_INCLUDE = ("fastdecov/", "manuscript/manuscript.md", "README.md", "analysis.py")
DEFAULT_SUFFIXES = (".py", ".cpp", ".md")
BOILERPLATE = ("import ", "from ", "#include", "using namespace")


class ProbeError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- registry

def load_markers(repo: Path) -> list[dict]:
    """Return the current and legacy canary markers as [{prefix, guid, current}]."""
    registry = json.loads((repo / "governance" / "canary-registry.json").read_text())
    markers = [{"prefix": registry["marker_prefix"], "guid": registry["project_guid"].lower(), "current": True}]
    for value in registry.get("legacy_markers", []):
        prefix, guid = value.rsplit(" GUID ", 1)
        markers.append({"prefix": prefix, "guid": guid.lower(), "current": False})
    return markers


def assert_no_guid(prompt: str, markers: list[dict]) -> None:
    """A prompt containing the answer would make every hit meaningless; refuse to send it."""
    lowered = prompt.lower()
    for m in markers:
        if m["guid"] in lowered or m["guid"][:8] in lowered:
            raise ProbeError("refusing to send a prompt that contains a canary GUID")


# ----------------------------------------------------------------------------- model client

def read_key(key_file: str | None) -> str:
    if key_file:
        path = Path(key_file).expanduser()
        if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            print(f"warning: {path} is readable by group/others; chmod 600 it", file=sys.stderr)
        return path.read_text().strip()
    key = os.environ.get("PROBE_API_KEY", "")
    if not key:
        raise ProbeError("set PROBE_API_KEY or pass --key-file")
    return key


def redact(text: str, key: str) -> str:
    if key:
        text = text.replace(key, "[REDACTED]")
    return SECRET_RE.sub("[REDACTED]", text)


def chat(base_url: str, key: str, model: str, prompt: str, max_tokens: int, timeout: float = 180) -> str:
    """One OpenAI-compatible chat completion, no tools (so the model cannot look the answer up)."""
    url = base_url.rstrip("/") + "/chat/completions"
    messages = [{"role": "user", "content": prompt}]
    # Some reasoning models reject temperature/max_tokens; fall back to the newer parameter shape on a 400.
    bodies = (
        {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens},
        {"model": model, "messages": messages, "max_completion_tokens": max_tokens},
    )
    last = "no attempt made"
    for body in bodies:
        for attempt in range(4):
            request = urllib.request.Request(
                url, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return extract_text(json.load(response))
            except urllib.error.HTTPError as exc:
                with exc:
                    last = redact(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:400]}", key)
                if exc.code == 429 or exc.code >= 500:
                    time.sleep(5 * 2 ** attempt)
                    continue
                if exc.code == 400:
                    break  # try the next body shape
                raise ProbeError(last) from None
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = redact(f"{type(exc).__name__}: {exc}", key)
                time.sleep(5 * 2 ** attempt)
    raise ProbeError(last)


def extract_text(data: dict) -> str:
    content = data["choices"][0]["message"].get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


# ----------------------------------------------------------------------------- canary test

def score_canary(output: str, guid: str) -> str:
    """hit = full GUID reproduced; partial = its first 8 hex digits (about 1 in 4e9 by chance); miss otherwise."""
    lowered = output.lower()
    if any(found.lower() == guid for found in UUID_RE.findall(lowered)):
        return "hit"
    if guid[:8] in lowered:
        return "partial"
    return "miss"


def canary_jobs(markers: list[dict]) -> list[dict]:
    return [
        {"test": "canary", "marker": f"{m['prefix']} GUID", "current": m["current"], "guid": m["guid"],
         "variant": i, "prompt": template.format(prefix=m["prefix"]), "max_tokens": 1024}
        for m in markers for i, template in enumerate(CANARY_PROMPTS)
    ]


# ----------------------------------------------------------------------------- continuation test

def clean_lines(text: str, markers: list[dict]) -> list[str]:
    """Drop marker lines so the continuation test never shows the model the canary."""
    prefixes = [m["prefix"] for m in markers]
    return [line for line in text.splitlines() if not any(p in line for p in prefixes)]


def distinctive(line: str) -> bool:
    s = line.strip()
    return len(s) >= 20 and not s.startswith(BOILERPLATE) and any(ch.isalnum() for ch in s)


def strip_fences(text: str) -> str:
    lines = text.strip("\n").splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def score_continuation(output: str, truth: list[str]) -> dict:
    out_lines = {line.strip() for line in strip_fences(output).splitlines() if line.strip()}
    targets = [line.strip() for line in truth if distinctive(line)]
    matched = sum(1 for line in targets if line in out_lines)
    norm_out = " ".join(strip_fences(output).split())
    norm_truth = " ".join(" ".join(truth).split())
    match = difflib.SequenceMatcher(None, norm_out, norm_truth, autojunk=False).find_longest_match(
        0, len(norm_out), 0, len(norm_truth))
    return {
        "distinctive_lines": len(targets),
        "distinctive_matched": matched,
        "recall": round(matched / len(targets), 4) if targets else None,
        "longest_common_chars": match.size,
    }


def pick_windows(files: list[tuple[str, list[str]]], samples: int, prefix_n: int, suffix_n: int,
                 seed: int) -> list[dict]:
    """Deterministic windows (fixed seed), so every model and every month sees the same excerpts."""
    rng = random.Random(seed)
    eligible = [(path, lines) for path, lines in files if len(lines) >= prefix_n + suffix_n]
    rng.shuffle(eligible)
    windows = []
    for path, lines in eligible:
        for _ in range(20):
            start = rng.randrange(0, len(lines) - prefix_n - suffix_n + 1)
            truth = lines[start + prefix_n: start + prefix_n + suffix_n]
            if sum(distinctive(line) for line in truth) >= 3:
                windows.append({"path": path, "start_line": start + 1, "prefix": lines[start:start + prefix_n],
                                "truth": truth})
                break
        if len(windows) == samples:
            break
    return windows


def repo_files(repo: Path, include: list[str], markers: list[dict]) -> list[tuple[str, list[str]]]:
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=repo, check=True, stdout=subprocess.PIPE).stdout
    out = []
    for raw in sorted(tracked.split(b"\0")):
        path = raw.decode("utf-8", "surrogateescape")
        if path and path.endswith(DEFAULT_SUFFIXES) and any(path == i or path.startswith(i) for i in include):
            try:
                out.append((path, clean_lines((repo / path).read_text(encoding="utf-8"), markers)))
            except UnicodeDecodeError:
                continue
    return out


def control_files(controls: Path, markers: list[dict]) -> list[tuple[str, list[str]]]:
    out = []
    for path in sorted(p for p in controls.rglob("*") if p.is_file()):
        try:
            out.append((path.relative_to(controls).as_posix(), clean_lines(path.read_text(encoding="utf-8"), markers)))
        except UnicodeDecodeError:
            continue
    return out


def continuation_jobs(windows: list[dict], group: str, repo_name: str) -> list[dict]:
    return [
        {"test": "continuation", "group": group, "path": w["path"], "start_line": w["start_line"],
         "prompt": CONTINUATION_PROMPT.format(path=w["path"], repo=repo_name, prefix="\n".join(w["prefix"])),
         "truth": w["truth"], "max_tokens": 2048}
        for w in windows
    ]


# ----------------------------------------------------------------------------- commands

def cmd_run(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    markers = load_markers(repo)
    models = list(args.model or [])
    if args.models_file:
        models += [line.strip() for line in Path(args.models_file).read_text().splitlines()
                   if line.strip() and not line.startswith("#")]
    if not models:
        raise ProbeError("pass --model or --models-file")

    jobs = canary_jobs(markers)
    if args.continuation:
        include = args.include or list(DEFAULT_INCLUDE)
        jobs += continuation_jobs(
            pick_windows(repo_files(repo, include, markers), args.samples, args.prefix_lines, args.suffix_lines,
                         args.seed), "repo", args.repo_name)
        if args.controls:
            jobs += continuation_jobs(
                pick_windows(control_files(Path(args.controls), markers), args.samples, args.prefix_lines,
                             args.suffix_lines, args.seed), "control", args.repo_name)
        else:
            print("warning: no --controls; repo recall has no baseline and cannot be interpreted alone",
                  file=sys.stderr)
    for job in jobs:
        assert_no_guid(job["prompt"], markers)

    sent_bytes = sum(len(j["prompt"].encode()) for j in jobs if j["test"] == "continuation")
    print(f"{len(models)} model(s) x {len(jobs)} prompt(s) = {len(models) * len(jobs)} call(s); "
          f"{sent_bytes} bytes of repository/control text per model")
    if args.dry_run:
        for job in jobs:
            label = job.get("marker") or f"{job['group']}:{job['path']}:{job['start_line']}"
            print(f"\n--- {job['test']} [{label}] ---\n{job['prompt']}")
        return 0

    base_url = args.base_url or os.environ.get("PROBE_BASE_URL", "")
    if not base_url:
        raise ProbeError("set PROBE_BASE_URL or pass --base-url (an OpenAI-compatible endpoint)")
    key = read_key(args.key_file)
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    for model in models:
        for job in jobs:
            record = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "run_id": run_id,
                      "model": model, "test": job["test"]}
            try:
                output = chat(base_url, key, model, job["prompt"], job["max_tokens"])
                record["output"] = output
                if job["test"] == "canary":
                    record.update(marker=job["marker"], current=job["current"], variant=job["variant"],
                                  result=score_canary(output, job["guid"]))
                else:
                    record.update(group=job["group"], path=job["path"], start_line=job["start_line"],
                                  **score_continuation(output, job["truth"]))
            except ProbeError as exc:
                record.update({k: job[k] for k in ("marker", "variant", "group", "path") if k in job},
                              error=str(exc))
            with log.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
            print(f"{model:40s} {job['test']:12s} {record.get('result') or record.get('recall') or record.get('error', '')}")
    print(f"\nrun {run_id} logged to {log}; summarize with: {Path(__file__).name} report")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    log = Path(args.log)
    if not log.exists():
        print(f"no log at {log}")
        return 0
    records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    by_model: dict[str, list[dict]] = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)
    rank = {"hit": 2, "partial": 1, "miss": 0}
    for model, rows in sorted(by_model.items()):
        latest = max(rows, key=lambda r: r["ts"])["run_id"]
        current = [r for r in rows if r["run_id"] == latest]
        when = min(r["ts"] for r in current)
        print(f"\n{model}  (latest run {latest}, {when})")
        canary = [r for r in current if r["test"] == "canary" and "result" in r]
        for marker in sorted({r["marker"] for r in canary}):
            best = max((r["result"] for r in canary if r["marker"] == marker), key=rank.get)
            print(f"  canary {marker}: {best.upper()}")
        ever = sorted({r["ts"][:10] for r in rows if r.get("result") == "hit"})
        if ever:
            print(f"  canary HIT on dates: {', '.join(ever)}")
        for group in ("repo", "control"):
            vals = [r["recall"] for r in current
                    if r["test"] == "continuation" and r.get("group") == group and r.get("recall") is not None]
            if vals:
                print(f"  continuation {group:7s}: mean distinctive-line recall {sum(vals) / len(vals):.3f} (n={len(vals)})")
        errors = [r for r in current if "error" in r]
        if errors:
            print(f"  {len(errors)} call(s) errored; first: {errors[0]['error'][:160]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="probe models (makes API calls unless --dry-run)")
    run.add_argument("--repo", default=str(REPO_ROOT), help="checkout to probe (default: the one containing this tool)")
    run.add_argument("--repo-name", default="arenasluo/SpecSmartDecov_V2")
    run.add_argument("--model", action="append", help="model name at the endpoint; repeatable")
    run.add_argument("--models-file", help="one model name per line; # comments allowed")
    run.add_argument("--base-url", help="OpenAI-compatible base URL (default: $PROBE_BASE_URL)")
    run.add_argument("--key-file", help="file holding the API key, mode 0600 (default: $PROBE_API_KEY)")
    run.add_argument("--continuation", action="store_true",
                     help="also run the continuation test; SENDS repository excerpts to the endpoint")
    run.add_argument("--controls", help="directory of never-published control files in the same style")
    run.add_argument("--include", action="append", help="repo path prefix to sample from; repeatable")
    run.add_argument("--samples", type=int, default=8)
    run.add_argument("--prefix-lines", type=int, default=30)
    run.add_argument("--suffix-lines", type=int, default=20)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--log", default=str(DEFAULT_LOG))
    run.add_argument("--dry-run", action="store_true", help="print every prompt and make no calls")
    report = sub.add_parser("report", help="summarize the log")
    report.add_argument("--log", default=str(DEFAULT_LOG))
    args = parser.parse_args(argv)
    try:
        return cmd_run(args) if args.command == "run" else cmd_report(args)
    except ProbeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
