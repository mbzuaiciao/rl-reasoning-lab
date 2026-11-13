"""Bounded Experiment 11 preflight. Runs at most one disposable LoRA optimizer step."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import resource
import string
import sys
from dataclasses import dataclass, replace
from time import perf_counter

from rl_reasoning_lab.environments.reasoning_tasks import GraphTraversalTask, ReasoningExample
from rl_reasoning_lab.verifiers.trace_evaluator import TraceEvaluator

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SEED = 1100
MAX_NEW_TOKENS = 48
CONTEXT = 256
# Processing order is part of the split definition. Reserve held-out identities first.
SPLITS = {
    "iid": (1101, (2, 3), 256),
    "ood": (1102, (4, 5, 6), 384),
    "dev_iid": (1103, (2, 3), 8),
    "dev_ood": (1104, (4, 5, 6), 9),
    "train": (1105, (2, 3), 1024),
}
EDGE = r"[A-Z][ \t]*->[ \t]*[A-Z]"
TRACE = re.compile(rf"R:[ \t]*({EDGE}(?:[ \t]*;[ \t]*{EDGE})*)")
ANSWER = re.compile(r"A:[ \t]*([A-Z])")


def graph_example(depth: int, rng: random.Random) -> ReasoningExample:
    # Equal-length disconnected paths prevent identifying the answer by component size.
    width = depth + 1
    nodes = rng.sample(string.ascii_uppercase, 3 * width)
    paths = [
        GraphTraversalTask(node_pool=nodes[i : i + width]).generate_example(depth, rng=rng)
        for i in range(0, len(nodes), width)
    ]
    query, *distractors = paths
    other_edges = [edge for path in distractors for edge in path.edges]
    shuffled = query.edges + other_edges
    rng.shuffle(shuffled)
    premises = "; ".join(f"{u}->{v}" for u, v in shuffled)
    return replace(
        query, distractor_edges=other_edges, prompt=f"{premises}; Q: {query.start_node}->?"
    )


def make_splits() -> dict[str, list[ReasoningExample]]:
    seen_graphs = set()
    seen_paths = set()
    splits = {}
    for name, (seed, depths, size) in SPLITS.items():
        rng = random.Random(seed)
        examples = []
        while len(examples) < size:
            ex = graph_example(depths[len(examples) % len(depths)], rng)
            graph = tuple(sorted(ex.edges + ex.distractor_edges))
            path = tuple(ex.path)
            if graph in seen_graphs or path in seen_paths:
                continue
            seen_graphs.add(graph)
            seen_paths.add(path)
            examples.append(ex)
        splits[name] = examples
    return splits


def messages(ex: ReasoningExample) -> list[dict[str, str]]:
    premises, _ = ex.prompt.rsplit("; Q:", 1)
    return [
        {"role": "system", "content": "You solve directed graph puzzles."},
        {
            "role": "user",
            "content": (
                "Follow the directed arrows from the start until there is no outgoing arrow. "
                "Ignore disconnected edges. Output only two lines: R: with the traversed "
                "edges in order, separated by semicolons, then A: with the final node.\n"
                "Example: edges 1->2; 0->1; 3->4, start 0.\n"
                "R: 0->1; 1->2\nA: 2\n"
                f"Edges: {premises}\nStart: {ex.start_node}"
            ),
        },
    ]


@dataclass(frozen=True)
class Score:
    answer: bool
    complete_path: bool
    formatted_trace: bool
    emitted_steps: int
    invalid_steps: int
    process: float


def score(text: str, ex: ReasoningExample) -> Score:
    lines = text.strip().splitlines()
    answer = ANSWER.fullmatch(lines[-1].strip()) if lines else None
    correct = bool(answer and answer[1] == ex.answer)
    trace = TRACE.fullmatch(lines[0].strip()) if len(lines) == 2 and answer else None
    if not trace:
        return Score(correct, False, False, 0, 0, 0.0)

    # Anchor the grammar before using the legacy parser and metrics.
    normalized = f"R: {trace[1]}; A: {answer[1]}"
    report = TraceEvaluator.evaluate(normalized, ex)
    steps = report.extracted_steps
    edges = set(ex.edges + ex.distractor_edges)
    current = ex.start_node
    prefix = 0
    for source, target in steps:
        if source != current or (source, target) not in edges or prefix == ex.depth:
            break
        prefix += 1
        current = target
    return Score(
        correct,
        report.complete_path,
        True,
        len(steps),
        sum(step not in edges for step in steps),
        prefix / ex.depth,
    )


def check_verifier() -> None:
    ex = GraphTraversalTask().generate_example(2, 2, random.Random(SEED))
    start, middle, end = ex.path
    wrong, distractor = ex.distractor_edges[0]
    valid = f"R: {start}->{middle}; {middle}->{end}\nA: {end}"
    cases = [
        (valid, Score(True, True, True, 2, 0, 1)),
        (valid[:-1] + wrong, Score(False, False, True, 2, 0, 1)),
        (f"A: {end}", Score(True, False, False, 0, 0, 0)),
        (f"R: {wrong}->{distractor}\nA: {end}", Score(True, False, True, 1, 0, 0)),
        (f"R: {start}->{middle}\nA: {end}", Score(True, False, True, 1, 0, 0.5)),
        (f"R: {start}->{middle}; {middle}->{wrong}\nA: {end}", Score(True, False, True, 2, 1, 0.5)),
        (f"R: {start}->{middle}; {start}->{middle}\nA: {end}", Score(True, False, True, 2, 0, 0.5)),
        (
            f"R: {start}->{middle}; junk; {middle}->{end}\nA: {end}",
            Score(True, False, False, 0, 0, 0),
        ),
        (valid + f"\nA: {wrong}", Score(False, False, False, 0, 0, 0)),
    ]
    for completion, expected in cases:
        if (actual := score(completion, ex)) != expected:
            raise AssertionError((completion, expected, actual))
    print(f"Verifier: {len(cases)} controls passed", flush=True)


def run_model(splits: dict[str, list[ReasoningExample]], local_snapshot: str | None) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    from rl_reasoning_lab.algorithms.ppo import compute_ppo_policy_loss

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    started = perf_counter()
    model_source = local_snapshot or MODEL
    load_options = {"local_files_only": True} if local_snapshot else {"revision": REVISION}
    tokenizer = AutoTokenizer.from_pretrained(model_source, **load_options)
    model = AutoModelForCausalLM.from_pretrained(
        model_source, dtype=torch.float32, use_safetensors=True, **load_options
    ).eval()
    print(f"Loaded {MODEL}@{REVISION} on CPU in {perf_counter() - started:.2f}s", flush=True)

    def encode(ex: ReasoningExample) -> torch.Tensor:
        return tokenizer.apply_chat_template(
            messages(ex), add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )

    longest_prompt = max(encode(ex).shape[-1] for split in splits.values() for ex in split)
    if longest_prompt + MAX_NEW_TOKENS > CONTEXT:
        raise ValueError("A split exceeds the context budget")
    print(f"Longest prompt: {longest_prompt} tokens", flush=True)

    for name, examples, sample in (
        ("dev_iid", splits["dev_iid"], False),
        ("dev_ood", splits["dev_ood"], False),
        ("train_sample", splits["train"][:16], True),
    ):
        started = perf_counter()
        reports, generated, capped = [], 0, 0
        first_failure = None
        for ex in examples:
            prompt = encode(ex)
            with torch.no_grad():
                rollout = model.generate(
                    prompt,
                    attention_mask=torch.ones_like(prompt),
                    return_dict_in_generate=True,
                    output_scores=sample,
                    generation_config=GenerationConfig(
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=sample,
                        temperature=1.0,
                        top_k=0 if sample else 50,
                        eos_token_id=model.generation_config.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    ),
                )
            full = rollout.sequences
            completion = full[0, prompt.shape[-1] :]
            text = tokenizer.decode(completion, skip_special_tokens=True)
            result = score(text, ex)
            if not result.complete_path and first_failure is None:
                first_failure = (ex.prompt, text, ex.trace_target.strip())
            reports.append(result)
            generated += len(completion)
            capped += len(completion) == MAX_NEW_TOKENS
        print(
            f"{name}: n={len(examples)} answer={sum(r.answer for r in reports)} "
            f"trace_format={sum(r.formatted_trace for r in reports)} "
            f"complete={sum(r.complete_path for r in reports)} "
            f"invalid_steps={sum(r.invalid_steps for r in reports)}/"
            f"{sum(r.emitted_steps for r in reports)} "
            f"process_nonzero={sum(r.process > 0 for r in reports)} "
            f"tokens={generated} capped={capped} seconds={perf_counter() - started:.2f}",
            flush=True,
        )

        if first_failure:
            print(
                f"{name} first path failure (prompt, output, target): {first_failure!r}", flush=True
            )

    # Reuse the last sampled training rollout. A fixed positive advantage isolates autograd
    # and optimizer feasibility even when the model earns no reward on the sampled tasks.
    with torch.no_grad():
        old_logp = model.compute_transition_scores(full, rollout.scores, normalize_logits=True)
    started = perf_counter()
    model = get_peft_model(
        model,
        LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5)
    actions = full[:, prompt.shape[-1] :]
    inputs = torch.nn.functional.pad(
        full[:, :-1], (0, CONTEXT - full.shape[-1] + 1), value=tokenizer.pad_token_id
    )
    logits = model(inputs, use_cache=False).logits[:, prompt.shape[-1] - 1 : full.shape[-1] - 1]
    logp = logits.log_softmax(dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    logp_error = (logp.detach() - old_logp).abs().max().item()
    if logp_error > 0.02:
        raise RuntimeError(f"Rollout and training action log probabilities differ by {logp_error}")
    loss, _ = compute_ppo_policy_loss(logp, old_logp, torch.ones_like(logp))
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    changed = max(params, key=lambda param: param.grad.abs().max().item())
    before = changed.detach().clone()
    optimizer.step()
    delta = (changed.detach() - before).abs().max().item()
    if not (torch.isfinite(loss) and 0 < delta < float("inf")):
        raise RuntimeError("LoRA update was nonfinite or failed to change an adapter")
    print(
        f"One LoRA step: trainable={sum(p.numel() for p in params)} actions={actions.numel()} "
        f"loss={loss.item():.4f} grad_norm={grad_norm.item():.4f} max_delta={delta:.3g} "
        f"max_logp_error={logp_error:.3g} seconds={perf_counter() - started:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--checks-only", action="store_true")
    options.add_argument("--local-snapshot", help="Local copy of the exact pinned checkpoint")
    args = parser.parse_args()
    started = perf_counter()
    splits = make_splits()
    for name, examples in splits.items():
        rows = [(ex.prompt, ex.trace_target) for ex in examples]
        digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
        print(f"{name}: n={len(rows)} sha256={digest}", flush=True)
    check_verifier()
    if not args.checks_only:
        run_model(splits, args.local_snapshot)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak_rss *= 1024
    print(f"Total wall: {perf_counter() - started:.2f}s, peak RSS: {peak_rss / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
