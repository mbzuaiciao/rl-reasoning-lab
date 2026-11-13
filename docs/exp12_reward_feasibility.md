# Experiment 12 · Pretrained reward-signal feasibility

[Issue #11](https://github.com/mbzuaiciao/rl-reasoning-lab/issues/11) asks whether a fixed pretrained policy produces a usable mix of correct, partially correct, and incorrect graph traversals. Experiment 11’s letter prompt restored format compliance, but its first sampled batch had effectively identical standardized outcome and hybrid advantages. Later baseline samples sometimes separated the rewards. The SFT adapter nearly saturated its sampled training probes and stopped early on deeper development paths. [Preflight](exp11_preflight.md) · [Warm-start](exp11_sft_warmstart.md) · [Depth diagnostic](exp11_depth_diagnostic.md)

## Protocol

- Generate 144 pilot-only graphs with the existing generator. At each of depths 2, 3, and 4, draw three disjoint batches of 16 graphs and sample once per graph. Each graph has a query chain and two disconnected chains of the same length, with all edges shuffled. Use graph seeds `12000 + 10 × depth + batch` and Torch seeds `13000 + 10 × depth + batch` for one-based batch numbers. Reject duplicate graphs and query paths within the pilot. The runner generates only the pilot dataset.
- Use the hash-verified Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`, the original letter prompt and chat template, and FP32 inference on four CPU threads. Use full-vocabulary sampling at temperature 1, a 48-new-token cap, and the checkpoint’s EOS IDs. Stop at the fixed grid or the 600-second wall timer and retain any completed rollouts.
- Use the existing strict verifier. Outcome is 1 for a correct final `A: X` line and 0 otherwise. Process is the fraction of the correct contiguous prefix starting at the queried node, with zero credit for malformed traces. A complete path also requires a correct answer. Hybrid is outcome + 0.5 × process. Standardize each reward separately within each batch using population standard deviation plus `1e-8`, with centering alone when SD is at most `1e-6`. Compute RMSE between the two advantage vectors on matched completions.
- A full batch passes when it has mixed outcomes, at least two incorrect outcomes with process reward strictly between 0 and 1, and advantage RMSE ≥ 0.10. At least two of the three batches at one depth must pass. This is a descriptive gate. Compare strict scores with the preflight’s independent answer and prefix extraction and recompute advantage RMSE on the strictly formatted subset to diagnose formatting effects. The full batch alone determines the gate.

## Reproduce

Run from the repository root with Python 3.13.12 and an existing local copy of the pinned snapshot. Supply a new output path.

```bash
python -m pip install -e '.[dev]' 'torch==2.15.0' 'transformers==4.57.6' 'peft==0.18.1'
HF_HUB_OFFLINE=1 python experiments/exp12_reward_feasibility.py \
  --local-snapshot /path/to/pinned/snapshot \
  --output /tmp/exp12-reward-feasibility.json
pytest -q
ruff check experiments/exp12_reward_feasibility.py
```

The [raw results](../experiments/results/exp12_reward_feasibility.json) include every graph, seed, completion, token ID, verifier score, reward histogram, and batch advantage vector. PEFT is a dependency of the imported Experiment 11 scripts. This runner loads only the pretrained model and executes inference.

## Results

**No-go for a future matched-budget RL experiment with this checkpoint and protocol.** Depths 2 and 3 each passed one batch. Depth 4 passed none. Every batch cleared the RMSE threshold, but three batches had no correct outcomes and four others had too few wrong answers with strictly partial credit. Only depth 2 produced complete paths (4/48). The pilot ended after the prespecified 144 completions.

| Depth | Batch | Correct | Strict format | Any valid prefix | Valid steps | Complete paths | Wrong with partial credit | Wrong with any credit | Advantage RMSE | RMSE, strict format only | Pass |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 2 | 1 | 2/16 | 9 | 2 | 3/32 | 1 | 1 | 1/14 | 0.256 | 0.271 | No |
| 2 | 2 | 2/16 | 10 | 6 | 10/32 | 1 | 2 | 5/14 | 0.517 | 0.491 | Yes |
| 2 | 3 | 4/16 | 7 | 2 | 4/32 | 2 | 0 | 0/12 | 0.226 | 0.229 | No |
| 3 | 1 | 2/16 | 13 | 3 | 4/48 | 0 | 3 | 3/14 | 0.283 | 0.282 | Yes |
| 3 | 2 | 0/16 | 8 | 3 | 4/48 | 0 | 3 | 3/16 | 1.000 | 1.000 | No |
| 3 | 3 | 3/16 | 11 | 3 | 3/48 | 0 | 1 | 1/13 | 0.122 | 0.107 | No |
| 4 | 1 | 0/16 | 11 | 3 | 5/64 | 0 | 3 | 3/16 | 1.000 | 1.000 | No |
| 4 | 2 | 1/16 | 12 | 1 | 1/64 | 0 | 1 | 1/15 | 0.125 | 0.125 | No |
| 4 | 3 | 0/16 | 11 | 3 | 4/64 | 0 | 3 | 3/16 | 1.000 | 1.000 | No |

The two passing batches retained RMSE of 0.491 and 0.282 among strictly formatted responses. Across the pilot, 17 wrong answers earned strictly partial credit by following one or more real edges from the queried start. The independent extractor found three additional positive prefixes among the 52 malformed responses and the same 14 correct terminal answers overall. Formatting suppressed credit in individual cases, while valid traversal explains the observed partial credit and reward separation persisted within the strict format. In the three all-wrong batches, RMSE near 1 comes from zero outcome advantages and nonconstant hybrid advantages. Those batches provide evidence of partial traversal alongside an outcome floor.

Representative failures use one-based row IDs from the raw results.

- **Wrong continuation after a valid edge, depth 2 batch 1 row 12.** The true path was `Q->E; E->O`. The response gave `R: Q->E; Q->K` and `A: E`, earning process reward 0.5 and outcome reward 0.
- **Premature stop, depth 4 batch 1 row 10.** The response followed `W->I; I->D` and answered `D`. The remaining edges were `D->T; T->M`, so it earned process reward 0.5 and outcome reward 0.
- **Correct terminal node with an invalid trace, depth 4 batch 2 row 7.** The response gave `R: D->X; X->N` and `A: N`. The true path was `D->T; T->O; O->X; X->N`. It earned outcome reward 1 and process reward 0.
- **Formatting penalty, depth 2 batch 2 row 16.** The response gave the correct `R: N->O; O->Y;` and `A: Y`. The strict verifier rejected the trace because of the trailing semicolon, so it earned outcome reward 1 and process reward 0.

The run took **151.84 seconds** on local macOS arm64 CPU with **2.78 GiB** peak process RSS. Seven completions reached the token cap. No training or cloud compute was used. All 128 repository tests and nine verifier controls passed. The pilot dataset regenerated exactly, the stored reward histograms and advantage RMSEs were independently checked from the completions, and a simulated timer expiry produced a partial report.
