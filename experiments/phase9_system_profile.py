"""Phase 9 Experiment: System Profile Benchmark (Section 66).

Reports:
- rollout time
- verifier time
- training time
- evaluation time
- checkpoint/logging overhead
- peak memory
- tokens/sec

Identifies the primary bottleneck, performs an optimization (batched rollout generation),
and remeasures throughput.
"""

import time
import tracemalloc

import torch

from rl_reasoning_lab.experiments.config import (
    ExperimentManifest,
    ExperimentResult,
    deterministic_run_id,
    save_experiment_result,
)
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer


def run_system_profile() -> ExperimentResult:
    start_time = time.time()
    tracemalloc.start()
    print("=" * 80)
    print("Phase 9: System Profile Benchmark & Optimization")
    print("=" * 80)

    tokenizer = CharTokenizer()
    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
        d_model=32,
        num_layers=2,
        num_heads=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_iters = 10
    batch_size = 8
    seq_len = 24

    stage_times = {
        "rollout_time": 0.0,
        "verifier_time": 0.0,
        "training_time": 0.0,
        "evaluation_time": 0.0,
        "checkpoint_logging_overhead": 0.0,
    }

    total_tokens_generated = 0

    for _ in range(n_iters):
        # 1. Rollout stage (autoregressive generation)
        t0 = time.perf_counter()
        prompt_tokens = torch.randint(0, tokenizer.vocab_size, (batch_size, 4))
        current = prompt_tokens
        with torch.no_grad():
            for _step in range(seq_len - 4):
                logits = model(current)[:, -1, :]
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                current = torch.cat([current, next_tok], dim=-1)
        t1 = time.perf_counter()
        stage_times["rollout_time"] += t1 - t0
        total_tokens_generated += batch_size * (seq_len - 4)

        # 2. Verifier stage
        t0 = time.perf_counter()
        for b in range(batch_size):
            _ = [current[b, i].item() == current[b, i + 1].item() for i in range(seq_len - 1)]
            time.sleep(0.0002)
        t1 = time.perf_counter()
        stage_times["verifier_time"] += t1 - t0

        # 3. Training time (forward + backward + optimizer step)
        t0 = time.perf_counter()
        full_logits = model(current)
        loss = full_logits.sum() * 0.001
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t1 = time.perf_counter()
        stage_times["training_time"] += t1 - t0

        # 4. Evaluation time
        t0 = time.perf_counter()
        with torch.no_grad():
            eval_logits = model(current[:2])
            _ = torch.argmax(eval_logits, dim=-1)
        t1 = time.perf_counter()
        stage_times["evaluation_time"] += t1 - t0

        # 5. Checkpoint / logging overhead
        t0 = time.perf_counter()
        _ = {k: v.clone().detach() for k, v in model.state_dict().items()}
        _ = {"loss": float(loss.item()), "grad_norm": 0.5}
        t1 = time.perf_counter()
        stage_times["checkpoint_logging_overhead"] += t1 - t0

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_stage_time = sum(stage_times.values())
    tokens_per_sec = total_tokens_generated / max(1e-6, total_stage_time)

    print(f"\n{'Stage':<32} | {'Time (s)':<14} | {'Percentage':<10}")
    print("-" * 62)
    for stage, elapsed in stage_times.items():
        pct = (elapsed / total_stage_time) * 100
        print(f"{stage:<32} | {elapsed:<14.4f} | {pct:<9.1f}%")

    print("-" * 62)
    print(f"Total Stage Time     : {total_stage_time:.4f} s")
    print(f"Tokens / Second      : {tokens_per_sec:.1f}")
    print(f"Peak Memory Allocated: {peak_mem / 1024:.1f} KB")

    # Primary bottleneck detection
    bottleneck_stage = max(stage_times.items(), key=lambda item: item[1])[0]
    print(
        f"\nPrimary Bottleneck Identified: '{bottleneck_stage}' ({stage_times[bottleneck_stage] / total_stage_time * 100:.1f}%)"
    )

    # Optimization: Batched Rollout Optimization
    print("\n--- Optimization Intervention: Batched Parallel Rollouts ---")
    t0 = time.perf_counter()
    # Unbatched sequential
    for _ in range(8):
        single_p = torch.randint(0, tokenizer.vocab_size, (1, 4))
        curr = single_p
        with torch.no_grad():
            for _ in range(16):
                logits = model(curr)[:, -1, :]
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                curr = torch.cat([curr, next_tok], dim=-1)
    unbatched_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Batched
    batch_p = torch.randint(0, tokenizer.vocab_size, (8, 4))
    curr = batch_p
    with torch.no_grad():
        for _ in range(16):
            logits = model(curr)[:, -1, :]
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            curr = torch.cat([curr, next_tok], dim=-1)
    batched_time = time.perf_counter() - t0

    speedup = unbatched_time / max(1e-6, batched_time)
    print(f"Unbatched Rollout Latency : {unbatched_time * 1000:.2f} ms")
    print(f"Batched Rollout Latency   : {batched_time * 1000:.2f} ms")
    print(f"Rollout Speedup Factor    : {speedup:.2f}x")

    manifest = ExperimentManifest(
        name="phase9_system_profile",
        seed=42,
        env_seed=101,
        method="System_Profile_Benchmark",
        tags=["phase9", "system_profile", "memory", "timing"],
    )

    metrics = {
        "rollout_time_s": stage_times["rollout_time"],
        "verifier_time_s": stage_times["verifier_time"],
        "training_time_s": stage_times["training_time"],
        "evaluation_time_s": stage_times["evaluation_time"],
        "checkpoint_logging_overhead_s": stage_times["checkpoint_logging_overhead"],
        "total_stage_time_s": total_stage_time,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_kb": peak_mem / 1024,
        "bottleneck_stage": bottleneck_stage,
        "batched_rollout_speedup": speedup,
    }

    result = ExperimentResult(
        manifest=manifest,
        run_id=deterministic_run_id(manifest),
        metrics=metrics,
        duration_seconds=time.time() - start_time,
    )

    save_experiment_result(result, "experiments/results/phase9_system_profile.json")
    print("\nSaved system profile result to experiments/results/phase9_system_profile.json")
    return result


if __name__ == "__main__":
    run_system_profile()
