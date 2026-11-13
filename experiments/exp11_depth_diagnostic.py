"""Diagnose the saved Experiment 11 adapter on nine deeper development graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch
from exp11_preflight import (
    ANSWER,
    MAX_NEW_TOKENS,
    MODEL,
    REVISION,
    TRACE,
    check_verifier,
    make_splits,
    messages,
    score,
)
from exp11_sft_warmstart import FROZEN_HASHES, THREADS, encode, peak_rss
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

SNAPSHOT_HASHES = {
    "model.safetensors": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "generation_config.json": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
}
ADAPTER_HASHES = {
    "adapter_model.safetensors": "b3062f6b4c98d07f564c85fddb99354197eaf1ae9c41fc609810ec438a7ad975",
    "adapter_config.json": "2e5bb0fffd812ff4b5ca7b1d502fbc9edac7b802cf5fa76b5ea07346154a0d12",
}
CONDITIONS = {"A": (MAX_NEW_TOKENS, 0), "B": (96, 0), "C": (MAX_NEW_TOKENS, 2)}


def verify_files(root, hashes):
    for name, expected in hashes.items():
        with (root / name).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError(f"{name} does not match the recorded checkpoint")


def trace_edges(line):
    trace = TRACE.fullmatch(line.strip())
    return (
        [tuple(part.replace(" ", "").replace("\t", "").split("->")) for part in trace[1].split(";")]
        if trace
        else []
    )


def valid_prefix(edges, expected):
    return next(
        (i for i, edge in enumerate(edges) if i >= len(expected) or edge != expected[i]),
        len(edges),
    )


def diagnose(completion, prefix, ex, eos):
    combined = prefix + completion
    strict = score(combined, ex)
    lines = combined.strip().splitlines()
    supplied = trace_edges(prefix)
    edges = trace_edges(lines[0]) if lines else []
    # A malformed R line is not an empty but valid continuation.
    parsed = bool(edges) and edges[: len(supplied)] == supplied
    generated = edges[len(supplied) :] if parsed else []
    expected = ex.edges[len(supplied) :]
    good = valid_prefix(generated, expected)
    first_invalid = good + 1 if good < len(generated) else None
    # Inspect arrows misplaced on the A line, without giving them trace or answer credit.
    answer_edges = (
        trace_edges("R:" + lines[1].strip()[2:])
        if (len(lines) == 2 and lines[1].strip().startswith("A:"))
        else []
    )
    misplaced = valid_prefix(answer_edges, expected[good:]) if parsed and not first_invalid else 0
    return {
        "terminal_correct": strict.answer,
        "complete_path": strict.complete_path and good == len(expected),
        "exact_format": strict.formatted_trace,
        "trace_line_parsed": parsed,
        "generated_trace_steps": len(generated),
        "valid_generated_transitions": good,
        "required_generated_transitions": len(expected),
        "first_invalid_generated_transition": first_invalid,
        "first_invalid_path_transition": first_invalid + len(supplied) if first_invalid else None,
        "premature_stop": eos and parsed and not first_invalid and good < len(expected),
        "premature_scalar_answer": (
            parsed
            and not first_invalid
            and good < len(expected)
            and len(lines) == 2
            and bool(ANSWER.fullmatch(lines[1].strip()))
        ),
        "valid_transitions_misplaced_in_answer": misplaced,
    }


@torch.inference_mode()
def run(args):
    started = perf_counter()
    verify_files(args.local_snapshot, SNAPSHOT_HASHES)
    verify_files(args.adapter, ADAPTER_HASHES)
    check_verifier()
    examples = make_splits()["dev_ood"]
    digest = hashlib.sha256(json.dumps([(ex.prompt, ex.trace_target) for ex in examples]).encode())
    if digest.hexdigest() != FROZEN_HASHES["dev_ood"]:
        raise ValueError("Frozen deeper development set changed")

    torch.set_num_threads(THREADS)
    tokenizer = AutoTokenizer.from_pretrained(args.local_snapshot, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.local_snapshot, local_files_only=True, dtype=torch.float32, use_safetensors=True
    )
    model = PeftModel.from_pretrained(base, args.adapter, local_files_only=True).eval()
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Expected a frozen inference model")
    eos_ids = model.generation_config.eos_token_id
    rows = []
    for index, ex in enumerate(examples):
        prompt = messages(ex, "letters")
        prompt_ids = encode(tokenizer, prompt)
        for condition, (cap, supplied) in CONDITIONS.items():
            prefix = (
                "R: " + "; ".join(f"{u}->{v}" for u, v in ex.edges[:supplied]) if supplied else ""
            )
            ids = prompt_ids
            if supplied:
                ids = tokenizer.apply_chat_template(
                    [*prompt, {"role": "assistant", "content": prefix}],
                    tokenize=True,
                    continue_final_message=True,
                )
                if (
                    ids[: len(prompt_ids)] != prompt_ids
                    or tokenizer.decode(ids[len(prompt_ids) :]) != prefix
                ):
                    raise ValueError("Chat template did not preserve the open assistant prefix")
            inputs = torch.tensor([ids], dtype=torch.long)
            new = model.generate(
                inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=GenerationConfig(
                    do_sample=False,
                    temperature=1.0,
                    top_k=50,
                    max_new_tokens=cap,
                    eos_token_id=eos_ids,
                    pad_token_id=tokenizer.pad_token_id,
                ),
            )[0, len(ids) :]
            eos = new[-1].item() in eos_ids
            completion = tokenizer.decode(new, skip_special_tokens=True)
            row = {
                "index": index,
                "depth": ex.depth,
                "condition": condition,
                "max_new_tokens": cap,
                "graph_prompt": ex.prompt,
                "reference": ex.trace_target.strip(),
                "supplied_prefix": prefix,
                "continuation": completion,
                "new_token_ids": new.tolist(),
                "response_tokens": len(new),
                "eos": eos,
                "hit_token_cap": not eos and len(new) == cap,
                **diagnose(completion, prefix, ex, eos),
            }
            if not supplied:
                row["verifier"] = asdict(score(completion, ex))
            rows.append(row)
        print(f"Completed A/B/C for development graph {index} (depth {ex.depth})", flush=True)
    return {
        "model": MODEL,
        "revision": REVISION,
        "adapter_sha256": ADAPTER_HASHES["adapter_model.safetensors"],
        "dev_ood_sha256": digest.hexdigest(),
        "threads": THREADS,
        "device": "cpu",
        "dtype": "float32",
        "seconds": perf_counter() - started,
        "peak_rss_gib": peak_rss(),
        "rows": rows,
    }


def deadline(_signum, _frame):
    raise TimeoutError("Stopped after the 10-minute diagnostic budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-snapshot", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new file")
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(600)
    try:
        report = run(args)
    finally:
        signal.alarm(0)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
