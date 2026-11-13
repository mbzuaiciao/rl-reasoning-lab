"""Run one local Experiment 11 SFT adapter and measure reward separation without RL."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import resource
import signal
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from exp11_preflight import (
    CONTEXT,
    MAX_NEW_TOKENS,
    MODEL,
    REVISION,
    SEED,
    check_verifier,
    make_splits,
    messages,
    score,
)
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

SFT_SIZE = 128
EPOCHS = 3
ACCUMULATION = 4
LR = 3e-4
TRAIN_SECONDS = 30 * 60
BATCH_COUNT = 3
BATCH_SIZE = 16
THREADS = 4
FROZEN_HASHES = {
    "iid": "33fa91e8c7cbd9468f6e0eb9eeb9764c3b62b2bdcbfce6567a55056e330a3c9c",
    "ood": "5a8469056a3e843cc6806e48b08c30a281e0b942615de01432921213fc169b4c",
    "dev_iid": "9a744b2f87e82890e61f6891395adb29f2933db95c1b7542e4e918c99d39bde2",
    "dev_ood": "a20adfad80aee8d57a17787e54be66e9f71e9969f9dff0540f1c3189bc033c58",
    "train": "7b8f7878014804708a29dfef8d58a1156a68c299f5d2cda2780d6c8e272b5ad9",
}


def prepare():
    splits = make_splits()
    for name, examples in splits.items():
        # Check frozen held-out fingerprints mechanically; never expose held-out rows.
        digest = hashlib.sha256(
            json.dumps([(ex.prompt, ex.trace_target) for ex in examples]).encode()
        ).hexdigest()
        if digest != FROZEN_HASHES[name]:
            raise ValueError(f"Frozen split changed: {name}")
    train = splits["train"]
    rng = random.Random(SEED)
    train_ids = sorted(
        i
        for depth in (2, 3)
        for i in rng.sample([i for i, ex in enumerate(train) if ex.depth == depth], SFT_SIZE // 2)
    )
    targets = []
    for i in train_ids:
        ex = train[i]
        trace = "; ".join(f"{u}->{v}" for u, v in zip(ex.path, ex.path[1:], strict=False))
        target = f"R: {trace}\nA: {ex.answer}"
        verified = score(target, ex)
        if not (verified.answer and verified.complete_path and verified.process == 1):
            raise ValueError(f"Invalid supervised target for train[{i}]")
        targets.append({"index": i, "messages": messages(ex, "letters"), "target": target})

    fit = set(train_ids)
    probe = rng.sample([i for i in range(len(train)) if i not in fit], BATCH_COUNT * BATCH_SIZE)
    groups = [
        (name, [(i, ex) for i, ex in enumerate(splits[name])], False)
        for name in ("dev_iid", "dev_ood")
    ]
    for start in range(0, len(probe), BATCH_SIZE):
        batch = probe[start : start + BATCH_SIZE]
        groups.append(
            (f"train_batch_{start // BATCH_SIZE + 1}", [(i, train[i]) for i in batch], True)
        )
    return targets, groups


def encode(tokenizer, prompt, *, generation=True):
    return tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=generation)


def supervised_ids(tokenizer, row):
    prompt = encode(tokenizer, row["messages"])
    full = encode(
        tokenizer,
        [*row["messages"], {"role": "assistant", "content": row["target"]}],
        generation=False,
    )
    if full[: len(prompt)] != prompt or tokenizer.eos_token_id not in full[len(prompt) :]:
        raise ValueError("Chat template does not preserve the prompt and assistant EOS")
    end = full.index(tokenizer.eos_token_id, len(prompt)) + 1
    if end > CONTEXT or end - len(prompt) > MAX_NEW_TOKENS:
        raise ValueError("Supervised sequence exceeds the frozen token budget")
    ids = torch.tensor(full[:end], dtype=torch.long).unsqueeze(0)
    return ids[:, :-1], ids[:, len(prompt) :]


def train(model, tokenizer, targets, adapter_dir):
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    batches = [supervised_ids(tokenizer, row) for row in targets]
    model = get_peft_model(model, config).train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    rng = random.Random(SEED)
    losses = []
    updates = 0

    def stop_at_deadline(_signum, _frame):
        raise TimeoutError("30-minute training cap reached; stop the pilot without further updates")

    previous_handler = signal.signal(signal.SIGALRM, stop_at_deadline)
    started = perf_counter()
    signal.setitimer(signal.ITIMER_REAL, TRAIN_SECONDS)
    try:
        for epoch in range(EPOCHS):
            order = list(range(len(batches)))
            rng.shuffle(order)
            total_loss = 0.0
            for start in range(0, len(order), ACCUMULATION):
                group = order[start : start + ACCUMULATION]
                optimizer.zero_grad(set_to_none=True)
                for i in group:
                    inputs, labels = batches[i]
                    logits = model(inputs, use_cache=False, logits_to_keep=labels.size(1)).logits
                    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
                    (loss / len(group)).backward()
                    total_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
            losses.append(total_loss / len(batches))
            print(
                f"Epoch {epoch + 1}: response loss={losses[-1]:.4f}, updates={updates}, "
                f"elapsed={perf_counter() - started:.1f}s",
                flush=True,
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    seconds = perf_counter() - started
    # Record the canonical base instead of the machine-specific snapshot path.
    # The vocabulary was not changed; skip PEFT's remote vocabulary lookup.
    config.base_model_name_or_path = MODEL
    config.revision = REVISION
    model.save_pretrained(adapter_dir, save_embedding_layers=False)
    backbone = model.unload()
    # Unloading restores the original layers but leaves the adapter registry behind.
    del backbone.peft_config
    return backbone, {
        "seed": SEED,
        "threads": THREADS,
        "dtype": "float32",
        "device": "cpu",
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "microbatch": 1,
        "accumulation": ACCUMULATION,
        "lora": {
            "rank": config.r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "targets": sorted(config.target_modules),
        },
        "updates": updates,
        "trainable_parameters": sum(p.numel() for p in params),
        "mean_response_loss_by_epoch": losses,
        "seconds": seconds,
        "cap_seconds": TRAIN_SECONDS,
        "peak_rss_gib_to_save": peak_rss(),
    }


def standardize(reward):
    sd = reward.std()
    centered = reward - reward.mean()
    return centered / (sd + 1e-8) if sd > 1e-6 else centered


def summarize(rows, sampled):
    outcome = np.array([int(row["answer"]) for row in rows])
    process = np.array([row["process"] for row in rows])
    hybrid = outcome + 0.5 * process
    result = {
        "n": len(rows),
        "answer": int(outcome.sum()),
        "formatted": sum(row["formatted_trace"] for row in rows),
        "any_prefix": int((process > 0).sum()),
        "prefix_steps": sum(row["prefix_steps"] for row in rows),
        "possible_steps": sum(row["depth"] for row in rows),
        "full_prefix": int((process == 1).sum()),
        "complete_path": sum(row["complete_path"] for row in rows),
        "partial_wrong_answer": int(((process > 0) & (process < 1) & (outcome == 0)).sum()),
        "positive_process_wrong_answer": int(((process > 0) & (outcome == 0)).sum()),
        "invalid_steps": sum(row["invalid_steps"] for row in rows),
        "emitted_steps": sum(row["emitted_steps"] for row in rows),
        "capped": sum(row["tokens"] == MAX_NEW_TOKENS for row in rows),
        "reward_histograms": {
            name: dict(sorted(Counter(map(float, values)).items()))
            for name, values in (("outcome", outcome), ("process", process), ("hybrid", hybrid))
        },
    }
    if sampled:
        a, b = standardize(outcome), standardize(hybrid)
        diff = a - b
        result["advantages"] = {
            "correlation": float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None,
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "max_abs_delta": float(np.max(np.abs(diff))),
            "outcome": a.tolist(),
            "hybrid": b.tolist(),
        }
    return result


@torch.inference_mode()
def evaluate(model, tokenizer, groups, checkpoint):
    model.eval()
    results = {}
    for group_index, (name, examples, sampled) in enumerate(groups):
        started = perf_counter()
        torch.manual_seed(SEED + group_index - 2 if sampled else SEED)
        rows = []
        for index, ex in examples:
            ids = torch.tensor([encode(tokenizer, messages(ex, "letters"))], dtype=torch.long)
            if ids.size(1) + MAX_NEW_TOKENS > CONTEXT:
                raise ValueError("Evaluation prompt exceeds the frozen token budget")
            output = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                generation_config=GenerationConfig(
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=sampled,
                    temperature=1.0,
                    top_k=0 if sampled else 50,
                    eos_token_id=model.generation_config.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                ),
            )[0, ids.size(1) :]
            completion = tokenizer.decode(output, skip_special_tokens=True)
            report = score(completion, ex)
            rows.append(
                {
                    "index": index,
                    "depth": ex.depth,
                    "prompt": ex.prompt,
                    "completion": completion,
                    "reference": ex.trace_target.strip(),
                    "prefix_steps": round(report.process * ex.depth),
                    "tokens": len(output),
                    **asdict(report),
                }
            )
        summary = summarize(rows, sampled)
        results[name] = {"summary": summary, "seconds": perf_counter() - started, "rows": rows}
        print(
            f"{checkpoint}/{name}: answer={summary['answer']}/{len(rows)} "
            f"prefix={summary['any_prefix']} complete={summary['complete_path']} "
            f"partial_wrong_answer={summary['partial_wrong_answer']} "
            f"elapsed={results[name]['seconds']:.1f}s",
            flush=True,
        )
    return results


def peak_rss():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (2**30 if sys.platform == "darwin" else 2**20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory for pilot files"
    )
    parser.add_argument(
        "--local-snapshot", type=Path, help="Local copy of the exact pinned revision"
    )
    args = parser.parse_args()

    started = perf_counter()
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    check_verifier()
    targets, groups = prepare()
    output = args.output_dir
    output.mkdir(parents=True)
    data = "".join(json.dumps(row) + "\n" for row in targets)
    (output / "sft_train.jsonl").write_text(data, encoding="utf-8")
    source = str(args.local_snapshot) if args.local_snapshot else MODEL
    options = {"local_files_only": True} if args.local_snapshot else {"revision": REVISION}
    tokenizer = AutoTokenizer.from_pretrained(source, **options)
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.float32, use_safetensors=True, **options
    )
    adapter = output / "adapter"
    model, training = train(model, tokenizer, targets, adapter)
    results = {
        "model": MODEL,
        "revision": REVISION,
        "split_hashes": FROZEN_HASHES,
        "sft_examples": len(targets),
        "sft_data_sha256": hashlib.sha256(data.encode()).hexdigest(),
        "training": training,
        "baseline": evaluate(model, tokenizer, groups, "baseline"),
        "sft": evaluate(PeftModel.from_pretrained(model, adapter), tokenizer, groups, "sft"),
        "total_seconds": perf_counter() - started,
        "peak_rss_gib": peak_rss(),
    }
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Pilot results: {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
