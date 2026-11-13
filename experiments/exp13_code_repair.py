"""Training-free code repair pilot; generated Python runs only in local Docker."""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import os
import platform
import re
import selectors
import signal
import subprocess
import uuid
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).with_name("exp13")
FROZEN = {
    "public.json": "c93f9814e884bde59146f61fb12acace3e3616d9e99696650860deb806e65c50",
    "hidden.json": "e8054f6b4299f66e3607723ae9d00c5686aeae0eb85f78364ef3e14bb33b066d",
    "prompt.txt": "3c58345891e2663e3f7d0822c227b068819c17de4d14ae42fcbbbe01f7c95d04",
}
IMAGE = "python@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_FILES = {
    "model.safetensors": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "generation_config.json": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
}
SAMPLES = 3
LIMIT = 600
OUTPUT_LIMIT = 8192
FLAGS = [
    "--rm",
    "--pull=never",
    "--interactive",
    "--network=none",
    "--read-only",
    "--cap-drop=ALL",
    "--security-opt=no-new-privileges",
    "--user=65534:65534",
    "--memory=128m",
    "--memory-swap=128m",
    "--pids-limit=16",
    "--cpus=.5",
    "--ulimit=cpu=1:1",
    "--ulimit=nofile=64:64",
    "--ulimit=core=0:0",
    "--log-driver=none",
]
# Trusted worker. The expected value, split, fixtures and repository are absent.
WORKER = """
import json, sys
request = json.load(sys.stdin)
namespace = {}
exec(compile(request["source"], "<candidate>", "exec"), namespace)
value = namespace[request["name"]](*request["args"])
print(json.dumps(value, allow_nan=False))
"""
PROBE = """
import errno, os, pathlib, resource, socket, sys
p = pathlib.Path
status = dict(line.split(":", 1) for line in p("/proc/self/status").read_text().splitlines())
assert os.getuid() == os.getgid() == 65534
assert int(status["CapEff"], 16) == int(status["CapBnd"], 16) == 0
assert status["NoNewPrivs"].strip() == "1" and status["Seccomp"].strip() == "2"
net = p("/sys/class/net")
assert {entry.name for entry in net.iterdir()
        if entry.is_dir() and int((entry / "flags").read_text(), 16) & 1} == {"lo"}
with socket.socket() as s:
    assert s.connect_ex(("192.0.2.1", 80)) == errno.ENETUNREACH
assert not any(p(path).exists() for path in ("/var/run/docker.sock", "/Users", "/workspace"))
assert set(os.environ) <= {"LC_CTYPE"}
assert os.statvfs("/").f_flag & os.ST_RDONLY
cg = p("/sys/fs/cgroup")
assert (cg / "memory.max").read_text().strip() == "134217728"
assert (cg / "memory.swap.max").read_text().strip() == "0"
assert (cg / "pids.max").read_text().strip() == "16"
assert (cg / "cpu.max").read_text().strip() == "50000 100000"
assert resource.getrlimit(resource.RLIMIT_CPU) == (1, 1)
print(sys.version.split()[0])
"""


class BudgetExceeded(Exception):
    pass


def deadline(_signum, _frame):
    raise BudgetExceeded


def verify_files(root, hashes):
    for name, expected in hashes.items():
        with (root / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"File does not match frozen digest: {name}")


def fixtures():
    verify_files(ROOT, FROZEN)
    public, hidden = [
        json.loads((ROOT / f"{split}.json").read_text()) for split in ("public", "hidden")
    ]
    return public, hidden, (ROOT / "prompt.txt").read_text()


def prompt_for(template, task, name):
    lines = []
    for case in task["cases"]:
        args = ", ".join(repr(arg) for arg in case["args"])
        lines.append(f"{name}({args}) -> {case['expected']!r}")
    return template.format(spec=task["spec"], buggy=task["buggy"], diagnostics="\n".join(lines))


def replacement(raw, name, params):
    source = raw.strip()
    fence = re.fullmatch(r"```(?:python)?\n(.*)\n```", source, re.DOTALL)
    if fence:
        source = fence[1]
    try:
        tree = ast.parse(source)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            return None
        fn = tree.body[0]
        args = fn.args
        if (
            fn.name != name
            or [arg.arg for arg in args.args] != params
            or fn.decorator_list
            or fn.returns
            or getattr(fn, "type_params", ())
            or any(arg.annotation for arg in args.args)
            or args.posonlyargs
            or args.kwonlyargs
            or args.vararg
            or args.kwarg
            or args.defaults
        ):
            return None
        compile(tree, "<candidate>", "exec")
    except (SyntaxError, ValueError, RecursionError):
        return None
    return source


class Sandbox:
    def __init__(self, context):
        endpoint = subprocess.check_output(
            ["docker", "context", "inspect", context, "--format", "{{.Endpoints.docker.Host}}"],
            timeout=10,
            text=True,
        ).strip()
        if not endpoint.startswith("unix:///"):
            raise ValueError("A local Unix-socket Docker context is required")
        self.docker = ["docker", "--context", context]

    def run(self, script, payload=b""):
        name = f"exp13-{uuid.uuid4().hex}"
        cmd = [
            *self.docker,
            "run",
            "--name",
            name,
            *FLAGS,
            IMAGE,
            "env",
            "-i",
            "/usr/local/bin/python",
            "-I",
            "-B",
            "-c",
            script,
        ]
        output = bytearray()
        stop = monotonic() + 5
        with subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as proc:
            try:
                proc.stdin.write(payload)
                proc.stdin.close()
                with selectors.DefaultSelector() as selector:
                    selector.register(proc.stdout, selectors.EVENT_READ)
                    while selector.select(max(0, stop - monotonic())):
                        chunk = os.read(proc.stdout.fileno(), OUTPUT_LIMIT)
                        if not chunk:
                            rc = proc.wait(timeout=max(0, stop - monotonic()))
                            if rc in {125, 126, 127}:
                                raise RuntimeError(f"Docker run failed ({rc}): {bytes(output)!r}")
                            return ("ok" if rc == 0 else f"exit_{rc}"), bytes(output)
                        output.extend(chunk)
                        if len(output) > OUTPUT_LIMIT:
                            return "output_limit", b""
                        if monotonic() >= stop:
                            break
                return "wall_limit", b""
            except subprocess.TimeoutExpired:
                return "wall_limit", b""
            finally:
                if proc.poll() is None:
                    # Killing just the Docker client does not necessarily stop the container.
                    proc.kill()
                    proc.wait(timeout=3)
                    result = subprocess.run(
                        [*self.docker, "rm", "-f", name],
                        timeout=8,
                        capture_output=True,
                        check=False,
                    )
                    if result.returncode and b"No such container" not in result.stderr:
                        raise RuntimeError(f"Could not clean up {name}")

    def check(self):
        status, version = self.run(PROBE)
        if status != "ok":
            raise RuntimeError(f"Isolation probe failed: {status}: {version!r}")
        cpu, _ = self.run("while True: pass")
        output, _ = self.run("while True: print('x' * 2048, flush=True)")
        wall, _ = self.run("import time; time.sleep(30)")
        if cpu != "exit_137" or output != "output_limit" or wall != "wall_limit":
            raise RuntimeError(f"Resource probes failed: {cpu}, {output}, {wall}")
        return {
            "docker_server": subprocess.check_output(
                [*self.docker, "version", "--format", "{{.Server.Version}}"], timeout=10, text=True
            ).strip(),
            "container_python": version.decode().strip(),
            "isolation": "passed",
            "cpu_probe": cpu,
            "output_probe": output,
            "wall_probe": wall,
        }

    def cases(self, name, source, cases):
        rows = []
        for index, case in enumerate(cases, 1):
            payload = json.dumps(dict(name=name, source=source, args=case["args"])).encode()
            status, output = self.run(WORKER, payload)
            passed = False
            if status == "ok":
                try:
                    value = json.loads(output)
                    # Canonical JSON preserves bool/int and int/float distinctions.
                    passed = json.dumps(value, sort_keys=True) == json.dumps(
                        case["expected"], sort_keys=True
                    )
                except (ValueError, RecursionError):
                    status = "invalid_result"
            rows.append(dict(case=index, passed=passed, status=status))
        return rows


def score(public, hidden, baseline):
    return {
        "public": public,
        "hidden": hidden,
        "public_fraction": sum(row["passed"] for row in public) / len(public),
        "public_delta": (
            sum(row["passed"] for row in public) - sum(row["passed"] for row in baseline["public"])
        )
        / len(public),
        "hidden_success": all(row["passed"] for row in hidden),
        "regressions": {
            split: [
                new["case"]
                for old, new in zip(baseline[split], rows, strict=True)
                if old["passed"] and not new["passed"]
            ]
            for split, rows in (("public", public), ("hidden", hidden))
        },
    }


def summarize(rows):
    applicable = [row for row in rows if row["applicable"]]
    groups = {}
    for row in applicable:
        groups.setdefault(row["task"], []).append(row)
    ranking = dict(concordant=0, tied=0, inverted=0)
    mixed = []
    per_task = {}
    for task, group in groups.items():
        success = [row for row in group if row["hidden_success"]]
        failure = [row for row in group if not row["hidden_success"]]
        pairs = dict(concordant=0, tied=0, inverted=0)
        if success and failure:
            mixed.append(task)
        for good, bad in itertools.product(success, failure):
            delta = good["public_delta"] - bad["public_delta"]
            key = "concordant" if delta > 0 else "inverted" if delta < 0 else "tied"
            pairs[key] += 1
            ranking[key] += 1
        per_task[task] = pairs
    failures = [row for row in applicable if not row["hidden_success"]]
    partial = sum(row["public_delta"] > 0 for row in failures)
    return dict(
        samples=len(rows),
        applicable=len(applicable),
        hidden_successes=sum(row["hidden_success"] for row in applicable),
        mixed_tasks=mixed,
        improving_hidden_failures=partial,
        full_public_hidden_failures=sum(row["public_fraction"] == 1 for row in failures),
        hidden_success_public_failures=sum(
            row["hidden_success"] and row["public_fraction"] < 1 for row in applicable
        ),
        pairs=ranking,
        pairs_by_task=per_task,
        numeric_gate=(len(mixed) >= 4 and partial >= 4),
    )


def load_model(snapshot):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    verify_files(snapshot, MODEL_FILES)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, dtype=torch.float32, use_safetensors=True
    ).eval()
    config = GenerationConfig(
        max_new_tokens=160,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        top_k=0,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    @torch.inference_mode()
    def generate(prompt, seed, cap=160):
        torch.manual_seed(seed)
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        tokens = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            generation_config=config,
            max_new_tokens=cap,
        )[0, ids.size(1) :].tolist()
        return tokenizer.decode(tokens, skip_special_tokens=True), tokens

    return generate, dict(
        torch=torch.__version__,
        transformers=transformers.__version__,
        generation=config.to_diff_dict(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", default="desktop-linux")
    parser.add_argument("--local-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    with args.output.open("x") as out:
        report = dict(
            phase="preflight_blocked",
            fixtures=FROZEN,
            model="Qwen/Qwen2.5-0.5B-Instruct",
            revision=REVISION,
            snapshot_sha256=MODEL_FILES,
            image=IMAGE,
            flags=FLAGS,
            host=dict(
                os=platform.system(), arch=platform.machine(), python=platform.python_version()
            ),
            preflight={},
            baselines={},
            candidates=[],
        )
        started = monotonic()
        pilot_started = None
        try:
            public, hidden, template = fixtures()
            sandbox = Sandbox(args.context)
            report["preflight"].update(sandbox.check())
            for name, task in public.items():
                cases = dict(public=task["cases"], hidden=hidden[name]["cases"])
                baseline = {
                    split: sandbox.cases(name, task["buggy"], values)
                    for split, values in cases.items()
                }
                report["baselines"][name] = baseline
                for split, values in cases.items():
                    reference = sandbox.cases(name, hidden[name]["reference"], values)
                    if not all(row["passed"] for row in reference):
                        raise ValueError(f"{name}: reference failed {split} control")
                if (
                    not any(row["passed"] for row in baseline["public"])
                    or all(row["passed"] for row in baseline["public"])
                    or all(row["passed"] for row in baseline["hidden"])
                ):
                    raise ValueError(f"{name}: baseline failed to give useful controls")
            report["preflight"]["fixture_controls"] = "passed"
            generate, model_info = load_model(args.local_snapshot)
            report["preflight"].update(model_info)
            smoke, ids = generate(
                "Complete this Python line: def add_one(n): return", 13000, cap=24
            )
            if not ids:
                raise ValueError("Empty offline smoke decode")
            report["preflight"]["model_smoke"] = dict(text=smoke, token_ids=ids)
            report["preflight"]["seconds"] = round(monotonic() - started, 3)
            report["phase"] = "preflight_passed"
            if args.preflight_only:
                return 0
            pilot_started = monotonic()
            report["phase"] = "pilot_incomplete"
            signal.signal(signal.SIGALRM, deadline)
            signal.setitimer(signal.ITIMER_REAL, LIMIT)
            for index, (name, task) in enumerate(public.items()):
                prompt = prompt_for(template, task, name)
                for sample in range(1, SAMPLES + 1):
                    seed = 13000 + index * SAMPLES + sample
                    raw, tokens = generate(prompt, seed)
                    source = replacement(raw, name, task["params"])
                    row = dict(
                        task=name,
                        sample=sample,
                        seed=seed,
                        completion=raw,
                        token_ids=tokens,
                        applicable=source is not None,
                        hidden_success=False,
                    )
                    if source is not None:
                        row.update(
                            score(
                                sandbox.cases(name, source, task["cases"]),
                                sandbox.cases(name, source, hidden[name]["cases"]),
                                report["baselines"][name],
                            )
                        )
                    report["candidates"].append(row)
                    print(f"{name} {sample}/{SAMPLES} complete", flush=True)
            report["phase"] = "pilot_complete"
            return 0
        except BudgetExceeded:
            report["stop_reason"] = f"{LIMIT}s pilot deadline"
            return 1
        except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
            report["stop_reason"] = f"{type(exc).__name__}: {exc}"
            print(report["stop_reason"], flush=True)
            return 1
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if pilot_started is not None:
                report["pilot_seconds"] = round(monotonic() - pilot_started, 3)
            report["summary"] = summarize(report["candidates"])
            report["leakage_review"] = "pending" if report["candidates"] else "not applicable"
            report["decision"] = (
                "review required"
                if report["phase"] == "pilot_complete" and report["summary"]["numeric_gate"]
                else "no-go"
            )
            json.dump(report, out, indent=2)
            out.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
