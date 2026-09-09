"""Synthetic multi-step reasoning environment: Graph Traversal and Rule Chaining.

Provides exact, programmatic ground-truth for:
1. Controllable reasoning depth d in {1, 2, 3, 4, 5, ...}.
2. Final answer token.
3. Intermediate reasoning steps and valid geodesic path.
4. Irrelevant distractor rules.
5. Counterfactual premise interventions (intervening on one edge).
6. Deliberate dataset shortcuts (for bias/shortcut detection).
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningExample:
    """A multi-step reasoning task instance with complete ground-truth metadata."""

    prompt: str
    answer: str
    depth: int
    path: list[str]
    edges: list[tuple[str, str]]
    distractor_edges: list[tuple[str, str]]
    start_node: str
    target_node: str
    answer_target: str
    trace_target: str


class GraphTraversalTask:
    """Parametric directed acyclic graph (DAG) reasoning task.

    Generates transitive relation chains such as:
        Premises: 'A->B; B->C; C->D; Q: A->?'
        Answer-only target: ' A: D'
        Reasoning-trace target: ' R: A->B; B->C; C->D; A: D'
    """

    def __init__(
        self,
        node_pool: list[str] | None = None,
        shuffle_rules: bool = False,
    ) -> None:
        self.node_pool = node_pool if node_pool is not None else list(string.ascii_uppercase)
        self.shuffle_rules = shuffle_rules

    def generate_example(
        self,
        depth: int = 2,
        num_distractors: int = 0,
        rng: random.Random | None = None,
    ) -> ReasoningExample:
        """Generate a single reasoning instance of exact depth d."""
        generator = rng if rng is not None else random.Random()

        total_nodes_needed = depth + 1 + (num_distractors * 2)
        if total_nodes_needed > len(self.node_pool):
            raise ValueError(
                f"Need {total_nodes_needed} unique nodes for depth {depth} and "
                f"{num_distractors} distractors, but pool has only {len(self.node_pool)}."
            )

        # Sample disjoint nodes for path and distractors
        sampled = generator.sample(self.node_pool, total_nodes_needed)
        path = sampled[: depth + 1]
        distractor_pool = sampled[depth + 1 :]

        # Geodesic path edges
        edges: list[tuple[str, str]] = []
        for i in range(depth):
            edges.append((path[i], path[i + 1]))

        # Distractor edges (independent disconnected edges or dead-ends)
        distractor_edges: list[tuple[str, str]] = []
        for i in range(num_distractors):
            u = distractor_pool[2 * i]
            v = distractor_pool[2 * i + 1]
            distractor_edges.append((u, v))

        all_rules = list(edges) + list(distractor_edges)
        if self.shuffle_rules:
            generator.shuffle(all_rules)

        rule_strs = [f"{u}->{v}" for u, v in all_rules]
        start_node = path[0]
        target_node = path[-1]

        prompt = "; ".join(rule_strs) + f"; Q: {start_node}->?"
        answer_target = f" A: {target_node}"

        trace_steps = [f"{path[i]}->{path[i + 1]}" for i in range(depth)]
        trace_target = " R: " + "; ".join(trace_steps) + f"; A: {target_node}"

        return ReasoningExample(
            prompt=prompt,
            answer=target_node,
            depth=depth,
            path=path,
            edges=edges,
            distractor_edges=distractor_edges,
            start_node=start_node,
            target_node=target_node,
            answer_target=answer_target,
            trace_target=trace_target,
        )

    def generate_dataset(
        self,
        size: int,
        depths: list[int] | None = None,
        num_distractors: int = 0,
        seed: int = 42,
    ) -> list[ReasoningExample]:
        """Generate a deterministic dataset across specified depths."""
        rng = random.Random(seed)
        depth_list = depths if depths is not None else [1, 2, 3]
        examples: list[ReasoningExample] = []

        for i in range(size):
            d = depth_list[i % len(depth_list)]
            examples.append(
                self.generate_example(depth=d, num_distractors=num_distractors, rng=rng)
            )

        return examples

    @staticmethod
    def create_counterfactual_intervention(
        example: ReasoningExample,
        intervene_step_idx: int = 0,
        new_dst: str | None = None,
        node_pool: list[str] | None = None,
        rng: random.Random | None = None,
    ) -> ReasoningExample:
        """Create a counterfactual intervention by mutating exactly one premise edge.

        For example, in 'A->B; B->C; C->D; Q: A->?', mutating step 1 (B->C) to B->X
        creates 'A->B; B->X; X->D*; Q: A->?' where the ground-truth target is updated.
        """
        generator = rng if rng is not None else random.Random()
        pool = node_pool if node_pool is not None else list(string.ascii_uppercase)

        path = list(example.path)
        depth = example.depth
        if intervene_step_idx < 0 or intervene_step_idx >= depth:
            raise ValueError(f"Invalid step index {intervene_step_idx} for depth {depth}.")

        # Pick a replacement node not in current path
        used = set(path)
        available = [n for n in pool if n not in used]
        if not available:
            raise ValueError("No available nodes for counterfactual intervention.")

        chosen_dst = new_dst if new_dst is not None else generator.choice(available)

        # Mutate the downstream chain starting from chosen_dst
        new_path = path[: intervene_step_idx + 1] + [chosen_dst]
        # For remaining steps, sample fresh nodes so chain remains intact with altered destination
        remaining_steps = depth - (intervene_step_idx + 1)
        used.add(chosen_dst)
        for _ in range(remaining_steps):
            cand = [n for n in pool if n not in used]
            nxt = generator.choice(cand)
            used.add(nxt)
            new_path.append(nxt)

        new_edges: list[tuple[str, str]] = []
        for i in range(depth):
            new_edges.append((new_path[i], new_path[i + 1]))

        all_rules = list(new_edges) + list(example.distractor_edges)
        rule_strs = [f"{u}->{v}" for u, v in all_rules]
        start_node = new_path[0]
        target_node = new_path[-1]

        prompt = "; ".join(rule_strs) + f"; Q: {start_node}->?"
        answer_target = f" A: {target_node}"

        trace_steps = [f"{new_path[i]}->{new_path[i + 1]}" for i in range(depth)]
        trace_target = " R: " + "; ".join(trace_steps) + f"; A: {target_node}"

        return ReasoningExample(
            prompt=prompt,
            answer=target_node,
            depth=depth,
            path=new_path,
            edges=new_edges,
            distractor_edges=example.distractor_edges,
            start_node=start_node,
            target_node=target_node,
            answer_target=answer_target,
            trace_target=trace_target,
        )

    def generate_shortcut_dataset(
        self,
        size: int,
        depth: int = 3,
        shortcut_answer: str = "Z",
        seed: int = 42,
    ) -> tuple[list[ReasoningExample], list[ReasoningExample]]:
        """Generate a biased (shortcut) dataset and a counterbalanced evaluation dataset.

        In the shortcut dataset, 100% of depth-d examples have target_node == shortcut_answer.
        In the counterbalanced dataset, answers are uniformly distributed across other letters.
        """
        rng = random.Random(seed)
        shortcut_examples: list[ReasoningExample] = []
        counterbalanced_examples: list[ReasoningExample] = []

        pool = [n for n in self.node_pool if n != shortcut_answer]

        for _ in range(size):
            # Biased: force target_node = shortcut_answer
            sampled = rng.sample(pool, depth)
            biased_path = sampled + [shortcut_answer]
            biased_edges = [(biased_path[i], biased_path[i + 1]) for i in range(depth)]
            prompt = "; ".join(f"{u}->{v}" for u, v in biased_edges) + f"; Q: {biased_path[0]}->?"
            trace_steps = [f"{biased_path[i]}->{biased_path[i + 1]}" for i in range(depth)]

            shortcut_examples.append(
                ReasoningExample(
                    prompt=prompt,
                    answer=shortcut_answer,
                    depth=depth,
                    path=biased_path,
                    edges=biased_edges,
                    distractor_edges=[],
                    start_node=biased_path[0],
                    target_node=shortcut_answer,
                    answer_target=f" A: {shortcut_answer}",
                    trace_target=" R: " + "; ".join(trace_steps) + f"; A: {shortcut_answer}",
                )
            )

            # Counterbalanced: target_node is NOT shortcut_answer
            cb_sampled = rng.sample(pool, depth + 1)
            cb_edges = [(cb_sampled[i], cb_sampled[i + 1]) for i in range(depth)]
            cb_target = cb_sampled[-1]
            cb_prompt = "; ".join(f"{u}->{v}" for u, v in cb_edges) + f"; Q: {cb_sampled[0]}->?"
            cb_trace = [f"{cb_sampled[i]}->{cb_sampled[i + 1]}" for i in range(depth)]

            counterbalanced_examples.append(
                ReasoningExample(
                    prompt=cb_prompt,
                    answer=cb_target,
                    depth=depth,
                    path=cb_sampled,
                    edges=cb_edges,
                    distractor_edges=[],
                    start_node=cb_sampled[0],
                    target_node=cb_target,
                    answer_target=f" A: {cb_target}",
                    trace_target=" R: " + "; ".join(cb_trace) + f"; A: {cb_target}",
                )
            )

        return shortcut_examples, counterbalanced_examples
