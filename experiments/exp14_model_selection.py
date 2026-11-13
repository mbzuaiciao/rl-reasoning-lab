"""Compare the pinned cached Qwen3.5 GGUF with the recorded EXP-013 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep

if __package__:
    from . import exp13_code_repair as repair
else:
    import exp13_code_repair as repair

MODEL = "unsloth/Qwen3.5-4B-GGUF"
REVISION = "e87f176479d0855a907a41277aca2f8ee7a09523"
SHA256 = "b252c5610a42ca82d20fe2a12813e9d069eed89292907e26c783eeb0bc961bc7"
SAMPLING = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 0,
    "samplers": ["temperature", "top_p"],
    "cache_prompt": False,
    "return_tokens": True,
}


class Sandbox(repair.Sandbox):
    def run(self, script, payload=b""):
        status, output = super().run(script, payload)
        if status == "exit_1":
            # Docker can return 1 for a disconnected daemon or a Python exception.
            subprocess.run(
                [*self.docker, "version", "--format", "{{.Server.Version}}"],
                check=True,
                stdout=subprocess.DEVNULL,
                timeout=10,
            )
        return status, output


def request(socket, endpoint, payload=None, timeout=2):
    command = [
        "curl",
        "--silent",
        "--show-error",
        "--noproxy",
        "*",
        "--unix-socket",
        str(socket),
        "http://localhost/" + endpoint,
    ]
    body = None
    if payload is not None:
        command += ["--header", "Content-Type: application/json", "--data-binary", "@-"]
        body = json.dumps(payload)
    result = json.loads(subprocess.check_output(command, input=body, text=True, timeout=timeout))
    if endpoint != "health" and "error" in result:
        raise ValueError(result["error"]["message"])
    return result


def generate(socket, prompt, seed, cap, deadline):
    template = request(
        socket, "apply-template", {"messages": [{"role": "user", "content": prompt}]}
    )
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("Inference deadline reached")
    result = request(
        socket,
        "completion",
        dict(SAMPLING, prompt=template["prompt"], seed=seed, n_predict=cap),
        timeout=remaining,
    )
    if result["truncated"] or result["stop_type"] not in {"eos", "limit"}:
        raise ValueError("Unexpected stop or context truncation")
    return result


def compare(args, report, server_log):
    with args.gguf.open("rb") as weights:
        if hashlib.file_digest(weights, "sha256").hexdigest() != SHA256:
            raise ValueError("GGUF does not match the pinned digest")

    public, hidden, template = repair.fixtures()
    baseline = json.loads(
        Path(__file__).with_name("results").joinpath("exp13_code_repair.json").read_text()
    )
    if baseline["fixtures"] != repair.FROZEN or baseline["phase"] != "pilot_complete":
        raise ValueError("Expected the complete EXP-013 run on the frozen fixtures")
    sandbox = Sandbox(args.context)
    report["preflight"] = sandbox.check()
    for name, task in public.items():
        for split, cases in (("public", task["cases"]), ("hidden", hidden[name]["cases"])):
            expected = baseline["baselines"][name][split]
            if sandbox.cases(name, task["buggy"], cases) != expected:
                raise ValueError(f"{name}: buggy function did not reproduce the {split} baseline")
            controls = sandbox.cases(name, hidden[name]["reference"], cases)
            if not all(row["passed"] for row in controls):
                raise ValueError(f"{name}: reference failed {split} control")
    report["preflight"]["fixture_controls"] = "passed"
    report["backend"] = subprocess.check_output(
        [args.server, "--version"], text=True, stderr=subprocess.STDOUT, timeout=10
    ).strip()
    flags = [
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
    report["server_flags"] = flags
    with TemporaryDirectory(prefix="exp14-") as directory:
        socket = Path(directory) / "llama.sock"
        started = monotonic()
        deadline = started + args.seconds
        with subprocess.Popen(
            [args.server, "--model", args.gguf, "--host", socket, *flags],
            stdout=server_log,
            stderr=subprocess.STDOUT,
        ) as server:
            try:
                while not (socket.exists() and request(socket, "health").get("status") == "ok"):
                    if server.poll() is not None or monotonic() >= deadline:
                        raise RuntimeError("Local inference server failed to start")
                    sleep(0.2)
                smoke = generate(
                    socket, "Complete this Python line: def add_one(n): return", 13000, 24, deadline
                )
                if not smoke["tokens"]:
                    raise ValueError("Empty offline smoke decode")
                report["preflight"]["smoke"] = smoke["content"]
                report["preflight"]["smoke_timings"] = smoke["timings"]
                for index, (name, task) in enumerate(public.items()):
                    prompt = repair.prompt_for(template, task, name)
                    for sample in range(1, repair.SAMPLES + 1):
                        seed = 13000 + index * repair.SAMPLES + sample
                        result = generate(socket, prompt, seed, 160, deadline)
                        raw = result["content"]
                        source = repair.replacement(raw, name, task["params"])
                        row = dict(
                            task=name,
                            sample=sample,
                            seed=seed,
                            completion=raw,
                            token_ids=result["tokens"],
                            stop=result["stop_type"],
                            timings=result["timings"],
                            applicable=source is not None,
                            hidden_success=False,
                        )
                        if source is not None:
                            row.update(
                                repair.score(
                                    sandbox.cases(name, source, task["cases"]),
                                    sandbox.cases(name, source, hidden[name]["cases"]),
                                    baseline["baselines"][name],
                                )
                            )
                        report["candidates"].append(row)
                        print(f"{name} {sample}/{repair.SAMPLES} complete", flush=True)
                report["phase"] = "comparison_complete"
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
                report["server_wall_seconds"] = round(monotonic() - started, 3)
                report["child_peak_rss_bytes"] = resource.getrusage(
                    resource.RUSAGE_CHILDREN
                ).ru_maxrss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", default="desktop-linux")
    parser.add_argument("--seconds", type=int, choices=range(1, repair.LIMIT + 1), metavar="1–600")
    parser.set_defaults(seconds=repair.LIMIT)
    args = parser.parse_args()
    report = dict(
        phase="comparison_incomplete",
        model=MODEL,
        revision=REVISION,
        gguf_sha256=SHA256,
        sampling=SAMPLING,
        fixtures=repair.FROZEN,
        candidates=[],
        budget_seconds=args.seconds,
    )
    started = monotonic()
    with args.output.open("x") as out, args.output.with_suffix(".log").open("x") as server_log:
        try:
            compare(args, report, server_log)
            return 0
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            report["stop_reason"] = f"{type(exc).__name__}: {exc}"
            print(report["stop_reason"], flush=True)
            return 1
        finally:
            report["total_wall_seconds"] = round(monotonic() - started, 3)
            report["summary"] = repair.summarize(report["candidates"])
            json.dump(report, out, indent=2)
            out.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
