"""Measure pretrained outcome and process reward separation on fresh pilot graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import signal
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch
import transformers
from exp11_depth_diagnostic import SNAPSHOT_HASHES, verify_files
from exp11_preflight import (
    CONTEXT,
    MAX_NEW_TOKENS,
    MODEL,
    REVISION,
    check_verifier,
    graph_example,
    inspect_completion,
    messages,
    score,
)
from exp11_sft_warmstart import THREADS, encode, peak_rss, summarize
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

DEPTHS = (2, 3, 4)
BATCH_COUNT = 3
BATCH_SIZE = 16
TIME_LIMIT = 600
MIN_PARTIAL_WRONG = 2
MIN_RMSE = 0.10
MIN_PASSING_BATCHES = 2


class BudgetExceeded(Exception):
    """The inference deadline expired."""


def deadline(_signum, _frame):
    raise BudgetExceeded


def pilot_batches():
    seen_graphs, seen_paths = set(), set()
    for depth in DEPTHS:
        for batch in range(1, BATCH_COUNT + 1):
            graph_seed = 12000 + 10 * depth + batch
            sampling_seed = 13000 + 10 * depth + batch
            rng = random.Random(graph_seed)
            examples = []
            while len(examples) < BATCH_SIZE:
                example = graph_example(depth, rng)
                graph = tuple(sorted(example.edges + example.distractor_edges))
                path = tuple(example.path)
                if graph in seen_graphs or path in seen_paths:
                    continue
                seen_graphs.add(graph)
                seen_paths.add(path)
                examples.append(example)
            yield depth, batch, graph_seed, sampling_seed, examples


def summarize_batch(rows):
    summary = summarize(rows, sampled=True)
    wrong = summary["n"] - summary["answer"]
    formatted = [row for row in rows if row["formatted_trace"]]
    summary.update(
        terminal_accuracy=summary["answer"] / summary["n"],
        wrong_answer_positive_prefix_rate=(
            summary["positive_process_wrong_answer"] / wrong if wrong else None
        ),
        independent_correct=sum(row["independent_answer_correct"] for row in rows),
        independent_any_prefix=sum(row["independent_prefix_steps"] > 0 for row in rows),
        malformed_with_independent_prefix=sum(
            row["independent_prefix_steps"] > 0 for row in rows if not row["formatted_trace"]
        ),
        formatted_only_advantage_rmse=(
            summarize(formatted, sampled=True)["advantages"]["rmse"] if formatted else None
        ),
        gate={
            "mixed_outcomes": 0 < summary["answer"] < summary["n"],
            "strict_partial_wrong": summary["partial_wrong_answer"] >= MIN_PARTIAL_WRONG,
            "advantage_rmse": summary["advantages"]["rmse"] >= MIN_RMSE,
        },
    )
    summary["passes"] = len(rows) == BATCH_SIZE and all(summary["gate"].values())
    return summary


@torch.inference_mode()
def sample(snapshot, batches, report):
    check_verifier()
    verify_files(snapshot, SNAPSHOT_HASHES)
    torch.set_num_threads(THREADS)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, dtype=torch.float32, use_safetensors=True
    ).eval()
    config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    report["generation"] = config.to_diff_dict()

    for group, (_, _, _, sampling_seed, examples) in zip(report["batches"], batches, strict=True):
        torch.manual_seed(sampling_seed)
        for row, example in zip(group["examples"], examples, strict=True):
            ids = torch.tensor([encode(tokenizer, row["messages"])], dtype=torch.long)
            if ids.size(1) + MAX_NEW_TOKENS > CONTEXT:
                raise ValueError("Pilot prompt exceeds the frozen token budget")
            output = model.generate(
                ids, attention_mask=torch.ones_like(ids), generation_config=config
            )[0, ids.size(1) :]
            completion = tokenizer.decode(output, skip_special_tokens=True)
            strict = score(completion, example)
            answer, prefix = inspect_completion(completion, example)
            outcome = int(strict.answer)
            group["rows"].append(
                {
                    "index": row["index"],
                    "depth": example.depth,
                    "completion": completion,
                    "token_ids": output.tolist(),
                    "tokens": len(output),
                    "termination": "eos" if output[-1].item() in config.eos_token_id else "cap",
                    "prefix_steps": round(strict.process * example.depth),
                    "independent_answer": answer,
                    "independent_answer_correct": answer == example.answer,
                    "independent_prefix_steps": prefix,
                    "outcome": outcome,
                    "hybrid": outcome + 0.5 * strict.process,
                    **asdict(strict),
                }
            )
        print(
            f"Depth {group['depth']} batch {group['batch']}: {len(group['rows'])} sampled",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new file")

    started = perf_counter()
    batches = list(pilot_batches())
    groups = [
        {
            "depth": depth,
            "batch": batch,
            "graph_seed": graph_seed,
            "sampling_seed": sampling_seed,
            "examples": [
                {
                    "index": index,
                    "messages": messages(example, "letters"),
                    "path": example.path,
                    "target": example.trace_target.strip(),
                }
                for index, example in enumerate(examples, 1)
            ],
            "rows": [],
        }
        for depth, batch, graph_seed, sampling_seed, examples in batches
    ]
    dataset = [group["examples"] for group in groups]
    report = {
        "experiment": 12,
        "model": MODEL,
        "revision": REVISION,
        "snapshot_sha256": SNAPSHOT_HASHES,
        "dataset_sha256": hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest(),
        "budget_seconds": TIME_LIMIT,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": "float32",
            "threads": THREADS,
        },
        "batches": groups,
    }

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(TIME_LIMIT)
    try:
        sample(args.local_snapshot, batches, report)
    except BudgetExceeded:
        report["status"] = "timed_out"
    else:
        report["status"] = "complete"
    finally:
        signal.alarm(0)

    for group in groups:
        rows = group["rows"]
        group["summary"] = summarize_batch(rows) if rows else None
    report["passing_batches_by_depth"] = {
        depth: sum(
            bool(group["summary"] and group["summary"]["passes"])
            for group in groups
            if group["depth"] == depth
        )
        for depth in DEPTHS
    }
    report["passing_depths"] = [
        depth
        for depth, count in report["passing_batches_by_depth"].items()
        if count >= MIN_PASSING_BATCHES
    ]
    report["seconds"] = perf_counter() - started
    report["peak_rss_gib"] = peak_rss()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(
        f"{report['status']}: {report['seconds']:.2f}s; passing depths {report['passing_depths']}"
    )


if __name__ == "__main__":
    main()
