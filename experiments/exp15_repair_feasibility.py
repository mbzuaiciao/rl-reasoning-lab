"""Run the frozen EXP-015 repair pilot or grade saved samples without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep

if __package__:
    from . import exp13_code_repair as repair
    from . import exp14_model_selection as model
else:
    import exp13_code_repair as repair
    import exp14_model_selection as model

ROOT = Path(__file__).with_name("exp15")
FROZEN = {
    "public.json": "a075035a5f6f0453f5c58938877aa1b32597032a0a1dd9a1cdd9e167ace29dfc",
    "hidden.json": "7af7783d3fc0438b2aa45dfa3371c0033092bc1373db5b603d32da375144bd8a",
    "prompt.txt": "3c58345891e2663e3f7d0822c227b068819c17de4d14ae42fcbbbe01f7c95d04",
}
FLAGS = [
    "--offline",
    "--threads",
    "4",
    "--gpu-layers",
    "0",
    "--no-op-offload",
    "--ctx-size",
    "1024",
    "--parallel",
    "1",
    "--no-ui",
    "--no-agent",
    "--reasoning",
    "off",
]
BACKEND = "commit 2a66703)"
FIRST_SEED = 15001
CAP = 160


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fixtures():
    repair.verify_files(ROOT, FROZEN)
    return (
        json.loads((ROOT / "public.json").read_text()),
        json.loads((ROOT / "hidden.json").read_text()),
        (ROOT / "prompt.txt").read_text(),
    )


def docker_preflight(public, hidden, report):
    if os.getuid() == 0:
        raise RuntimeError("Run as the regular macOS user")
    context = "desktop-linux"
    endpoint = subprocess.check_output(
        ["docker", "context", "inspect", context, "--format", "{{.Endpoints.docker.Host}}"],
        text=True,
        timeout=10,
    ).strip()
    if endpoint != f"unix://{Path.home() / '.docker/run/docker.sock'}":
        raise RuntimeError("Expected Docker Desktop's per-user socket")
    sandbox = model.Sandbox(context)
    report["docker"] = sandbox.check()
    baselines = {}
    for name, task in public.items():
        baselines[name] = {}
        for split, cases in (("public", task["cases"]), ("hidden", hidden[name]["cases"])):
            baseline = sandbox.cases(name, task["buggy"], cases)
            reference = sandbox.cases(name, hidden[name]["reference"], cases)
            passes = [case["passed"] for case in baseline]
            if all(passes) or (split == "public" and not any(passes)):
                raise ValueError(f"{name}: ineffective {split} buggy control")
            if not all(case["passed"] for case in reference):
                raise ValueError(f"{name}: {split} reference control failed")
            baselines[name][split] = baseline
    report["docker"]["fixture_controls"] = "passed"
    report["baselines"] = baselines
    return sandbox


def sample_plan(public):
    for index, name in enumerate(public):
        for slot in range(repair.SAMPLES):
            yield name, slot + 1, FIRST_SEED + index * repair.SAMPLES + slot


def patch_for(row, public):
    raw = row["completion"]
    source = repair.replacement(raw, row["task"], public[row["task"]]["params"])
    if (
        digest(raw) != row["completion_sha256"]
        or (digest(source) if source is not None else None) != row["patch_sha256"]
    ):
        raise ValueError(f"Saved hash mismatch on {row['task']} sample {row['sample']}")
    return source


def check_budget(deadline):
    if monotonic() >= deadline:
        raise TimeoutError("Pilot wall-clock budget reached")


def grade(sandbox, report, public, hidden, deadline):
    for row in report["candidates"]:
        if row["hidden_success"] is not None:
            continue
        source = patch_for(row, public)
        name = row["task"]
        results = {}
        for split, cases in (("public", public[name]["cases"]), ("hidden", hidden[name]["cases"])):
            results[split] = []
            for case_index, case in enumerate(cases, 1):
                check_budget(deadline)
                (result,) = sandbox.cases(name, source, [case])
                results[split].append(dict(result, case=case_index))
        row.update(repair.score(results["public"], results["hidden"], report["baselines"][name]))
        print(f"graded {name} {row['sample']}/{repair.SAMPLES}", flush=True)


def sample(args, report, public, template, log):
    repair.verify_files(args.gguf.parent, {args.gguf.name: model.SHA256})
    version = subprocess.check_output(
        [args.server, "--version"], text=True, stderr=subprocess.STDOUT, timeout=10
    ).strip()
    if BACKEND not in version:
        raise ValueError("Expected the EXP-014 llama.cpp build")
    report["backend"] = version
    with TemporaryDirectory(prefix="exp15-") as directory:
        socket = Path(directory) / "llama.sock"
        with subprocess.Popen(
            [args.server, "--model", args.gguf, "--host", socket, *FLAGS],
            stdout=log,
            stderr=subprocess.STDOUT,
        ) as server:
            try:
                ready_by = monotonic() + 60
                while not (
                    socket.exists() and model.request(socket, "health").get("status") == "ok"
                ):
                    if server.poll() is not None or monotonic() >= ready_by:
                        raise RuntimeError("Local inference server failed to start")
                    sleep(0.2)
                smoke = model.generate(
                    socket, "Complete this Python line: def add_one(n): return", 15000, 24, ready_by
                )
                if not smoke["tokens"]:
                    raise ValueError("Empty offline smoke decode")
                report["smoke"] = smoke["content"]
                started = monotonic()
                deadline = started + repair.LIMIT
                try:
                    for name, slot, seed in sample_plan(public):
                        check_budget(deadline)
                        prompt = repair.prompt_for(template, public[name], name)
                        result = model.generate(socket, prompt, seed, CAP, deadline)
                        raw = result["content"]
                        source = repair.replacement(raw, name, public[name]["params"])
                        report["candidates"].append(
                            dict(
                                task=name,
                                sample=slot,
                                seed=seed,
                                completion=raw,
                                completion_sha256=digest(raw),
                                patch_sha256=digest(source) if source is not None else None,
                                token_ids=result["tokens"],
                                stop=result["stop_type"],
                                timings=result["timings"],
                                applicable=source is not None,
                                hidden_success=None if source is not None else False,
                            )
                        )
                        print(f"sampled {name} {slot}/{repair.SAMPLES}", flush=True)
                finally:
                    report["sampling_seconds"] = round(monotonic() - started, 3)
                    report["pilot_started"] = started
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
    return deadline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--regrade", type=Path, help="grade only saved ungraded patches")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.regrade and (args.server or args.gguf)) or (
        not args.regrade and not (args.server and args.gguf)
    ):
        parser.error("supply --regrade or both --server and --gguf")
    report = dict(
        phase="incomplete",
        fixtures=FROZEN,
        model=model.MODEL,
        revision=model.REVISION,
        gguf_sha256=model.SHA256,
        sampling=dict(model.SAMPLING, cap=CAP, first_seed=FIRST_SEED),
        server_flags=FLAGS,
        image=repair.IMAGE,
        docker_flags=repair.FLAGS,
        candidates=[],
    )
    wall_started = monotonic()
    with args.output.open("x") as out:
        try:
            public, hidden, template = fixtures()
            if args.regrade:
                saved = json.loads(args.regrade.read_text())
                for key in ("fixtures", "model", "revision", "gguf_sha256", "sampling", "image"):
                    if saved[key] != report[key]:
                        raise ValueError(f"Saved run does not match frozen {key}")
                rows = saved["candidates"]
                expected = list(sample_plan(public))
                slots = [(row["task"], row["sample"], row["seed"]) for row in rows]
                if slots != expected[: len(rows)]:
                    raise ValueError("Saved rows do not match the frozen sample order")
                for row in rows:
                    patch_for(row, public)
                report.update(saved)
                report["phase"] = "incomplete"
                report.setdefault("prior_stop_reasons", []).append(report.pop("stop_reason", None))
            sandbox = docker_preflight(public, hidden, report)
            if args.regrade:
                deadline = monotonic() + repair.LIMIT
            else:
                with args.output.with_suffix(".log").open("x") as log:
                    deadline = sample(args, report, public, template, log)
            grade(sandbox, report, public, hidden, deadline)
            if len(report["candidates"]) == len(list(sample_plan(public))):
                report["phase"] = "complete"
                return 0
            report["stop_reason"] = "Saved run has missing samples"
            return 1
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            report["stop_reason"] = f"{type(exc).__name__}: {exc}"
            print(report["stop_reason"], flush=True)
            return 1
        finally:
            rows = report["candidates"]
            graded = [row for row in rows if row["hidden_success"] is not None]
            report["summary"] = repair.summarize(graded)
            report["applicable"] = sum(row["applicable"] for row in rows)
            report["ungraded_applicable"] = len(rows) - len(graded)
            if "pilot_started" in report:
                report["pilot_seconds"] = round(monotonic() - report.pop("pilot_started"), 3)
            report["command_seconds"] = round(monotonic() - wall_started, 3)
            json.dump(report, out, indent=2)
            out.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
