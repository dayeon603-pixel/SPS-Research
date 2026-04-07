"""
Empirical SPS estimation on RoBERTa-base.

Computes SPS, relative SPS, spectral gap, and the layer-wise SPS depth profile
for the RoBERTa-base encoder on a set of test sentences using:
  - T_emb: Embedding perturbation along WordNet synonym directions
  - T_syn: Discrete synonym substitution

Usage
-----
    python experiments/estimate_sps.py
    python experiments/estimate_sps.py --epsilon 0.05 --n-samples 256 --device cuda

Output columns in the results table:
    Family      | Transformation family used
    SPS         | SPS_eps(f)  (Definition 2)
    MeanSens    | E_x[Sens_{T,eps}(f; x)]
    rSPS        | Relative SPS vs. T_arb (Definition 8)
    GapMean     | Mean normalized spectral gap gamma-bar (Definition 5)

References
----------
Kang (2026) Definitions 2, 5, 6, 7, 8.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sps import (
    EmbeddingPerturbationConfig,
    EmbeddingPerturbationFamily,
    SPSConfig,
    SPSReport,
    build_sps_estimator,
    build_wordnet_synonym_map,
    estimate_arbitrary_sps,
    full_sps_analysis,
    relative_sps,
    set_seed,
    spectral_gap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("estimate_sps")


# ---------------------------------------------------------------------------
# Test corpus
# ---------------------------------------------------------------------------

TEST_SENTENCES: list[str] = [
    # Paraphrase pairs (same semantics, different surface forms)
    "The scientist conducted an experiment to test the hypothesis.",
    "The researcher performed a study to evaluate the theory.",
    "The government announced new policies to address climate change.",
    "The administration revealed fresh measures to tackle global warming.",
    "Children learn best through play and exploration.",
    "Kids acquire knowledge most effectively via games and discovery.",
    # Diverse topics (semantic stability across domains)
    "The stock market experienced significant volatility last week.",
    "Rapid advances in technology are transforming modern economies.",
    "Access to clean water remains a global public health challenge.",
    "Neural networks have achieved remarkable success in pattern recognition.",
    "The ancient ruins were discovered beneath the city center.",
    "Political tensions escalated following the disputed election results.",
    "Renewable energy sources are becoming increasingly cost-competitive.",
    "The study found a strong correlation between diet and cognitive function.",
    "International cooperation is essential for addressing pandemics.",
    "The novel explores themes of identity, memory, and belonging.",
]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_batches(
    sentences: list[str],
    tokenizer,
    embedding_layer: torch.nn.Embedding,
    batch_size: int,
    device: str,
) -> list[dict[str, torch.Tensor]]:
    """
    Tokenize sentences and precompute embeddings into a list of batch dicts.

    Each batch dict contains:
        input_ids      (B, seq_len)
        attention_mask (B, seq_len)
        embeddings     (B, seq_len, hidden_dim)
    """
    batches: list[dict[str, torch.Tensor]] = []

    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i : i + batch_size]
        enc = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            embeddings = embedding_layer(input_ids)                # (B, seq, h)

        batches.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "embeddings": embeddings,
            }
        )

    return batches


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_results_table(rows: list[dict]) -> None:
    col_w = {"Family": 10, "SPS": 8, "MeanSens": 10, "StdSens": 9, "rSPS": 8, "GapMean": 9}
    header = "  ".join(k.ljust(v) for k, v in col_w.items())
    sep = "  ".join("-" * v for v in col_w.values())
    print("\n" + header)
    print(sep)
    for r in rows:
        line = "  ".join(
            str(r.get(k, "—")).ljust(v) for k, v in col_w.items()
        )
        print(line)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Empirical SPS estimation on RoBERTa-base.")
    p.add_argument("--epsilon", type=float, default=0.1,
                   help="Perturbation radius epsilon (default: 0.1).")
    p.add_argument("--m-transforms", type=int, default=24,
                   help="Transformations sampled per input m (default: 24).")
    p.add_argument("--n-directions", type=int, default=8,
                   help="Semantic directions K per sample (default: 8).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Batch size (default: 8).")
    p.add_argument("--device", type=str, default="cpu",
                   help="Torch device (default: cpu).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-layerwise", action="store_true",
                   help="Skip layer-wise SPS computation (faster).")
    p.add_argument("--skip-wordnet", action="store_true",
                   help="Skip WordNet synonym map (use random directions for T_emb).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # ------------------------------------------------------------------
    # Load model and tokenizer
    # ------------------------------------------------------------------
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        logger.error("transformers is required. Install with: pip install transformers")
        sys.exit(1)

    model_name = "roberta-base"
    logger.info("Loading %s ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model = model.to(args.device)

    embedding_layer = model.get_input_embeddings()

    # ------------------------------------------------------------------
    # Build synonym map
    # ------------------------------------------------------------------
    synonym_map: dict[int, list[int]] = {}
    if not args.skip_wordnet:
        logger.info("Building WordNet synonym map ...")
        try:
            import nltk
            try:
                nltk.data.find("corpora/wordnet")
            except LookupError:
                logger.info("Downloading WordNet corpus ...")
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
            synonym_map = build_wordnet_synonym_map(
                tokenizer,
                vocab_size=10_000,             # limit for speed in this experiment
                max_synonyms_per_token=5,
            )
        except Exception as e:
            logger.warning("WordNet synonym map failed (%s). Falling back to random directions.", e)

    # ------------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------------
    logger.info("Tokenizing %d sentences ...", len(TEST_SENTENCES))
    batches = prepare_batches(
        TEST_SENTENCES, tokenizer, embedding_layer,
        batch_size=args.batch_size, device=args.device,
    )

    # ------------------------------------------------------------------
    # SPS configuration
    # ------------------------------------------------------------------
    config = SPSConfig(
        epsilon=args.epsilon,
        m_transforms=args.m_transforms,
        divergence="cosine",
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------
    # T_emb family
    # ------------------------------------------------------------------
    logger.info("Building T_emb (embedding perturbation family) ...")
    t_emb_config = EmbeddingPerturbationConfig(
        n_directions=args.n_directions,
        use_synonym_directions=bool(synonym_map),
    )
    t_emb = EmbeddingPerturbationFamily(
        embedding_layer=embedding_layer,
        synonym_map=synonym_map or None,
        config=t_emb_config,
    )

    # ------------------------------------------------------------------
    # Core SPS  (Definition 2)
    # ------------------------------------------------------------------
    logger.info("Estimating SPS (T_emb, epsilon=%.3f, m=%d) ...", args.epsilon, args.m_transforms)
    estimator = build_sps_estimator(model, t_emb, config)
    core = estimator.estimate(iter(batches))

    # ------------------------------------------------------------------
    # Arbitrary perturbation baseline  (Definition 8 denominator)
    # ------------------------------------------------------------------
    logger.info("Estimating SPS under T_arb (arbitrary perturbations) ...")
    arb_sps = estimate_arbitrary_sps(model, iter(batches), config)
    rsps = relative_sps(core["sps"], arb_sps)

    # ------------------------------------------------------------------
    # Spectral gap  (Definition 5)
    # ------------------------------------------------------------------
    logger.info("Computing spectral gap ...")
    gap_result = None
    gap_mean: float | None = None

    try:
        batch0 = batches[0]
        emb0 = batch0["embeddings"].to(args.device)
        ids0 = batch0["input_ids"].to(args.device)
        mask0 = batch0["attention_mask"].to(args.device)

        directions = t_emb.semantic_directions(emb0, ids0)

        # Use math (eager) SDPA kernel so JVP backward works on CPU.
        # Flash attention / memory-efficient attention lack CPU JVP support.
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel as _sdpa_kernel
            def _model_fn(inputs_embeds: torch.Tensor) -> torch.Tensor:
                with _sdpa_kernel(SDPBackend.MATH):
                    out = model(inputs_embeds=inputs_embeds, attention_mask=mask0)
                return out.last_hidden_state[:, 0, :]
        except (ImportError, AttributeError):
            def _model_fn(inputs_embeds: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
                out = model(inputs_embeds=inputs_embeds, attention_mask=mask0)
                return out.last_hidden_state[:, 0, :]

        gap_result = spectral_gap(_model_fn, emb0, directions, n_probe_full=16)
        gap_mean = gap_result.mean_gap
        logger.info("Mean spectral gap: %.4f", gap_mean)
    except Exception as e:
        logger.warning("Spectral gap computation failed: %s", e)

    # ------------------------------------------------------------------
    # Layer-wise SPS  (Definition 7)
    # ------------------------------------------------------------------
    layerwise: list[float] | None = None
    if not args.skip_layerwise:
        logger.info("Computing layer-wise SPS profile ...")
        try:
            from sps.metrics import LayerwiseSPSAnalyzer
            analyzer = LayerwiseSPSAnalyzer(model, t_emb, config)
            layerwise = analyzer.compute_profile_from_batches(batches)
        except Exception as e:
            logger.warning("Layer-wise SPS failed: %s", e)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  SPS Estimation — RoBERTa-base")
    print(f"  Model: {model_name}  |  epsilon={args.epsilon}  |  m={args.m_transforms}")
    print("=" * 65)

    rows = [
        {
            "Family": "T_emb",
            "SPS": f"{core['sps']:.4f}",
            "MeanSens": f"{core['mean_sensitivity']:.4f}",
            "StdSens": f"{core['std_sensitivity']:.4f}",
            "rSPS": f"{rsps:.4f}",
            "GapMean": f"{gap_mean:.4f}" if gap_mean is not None else "—",
        },
        {
            "Family": "T_arb",
            "SPS": f"{arb_sps:.4f}",
            "MeanSens": "—",
            "StdSens": "—",
            "rSPS": "1.0000",
            "GapMean": "—",
        },
    ]
    print_results_table(rows)

    # rSPS interpretation (compare with tolerance to avoid float display/logic mismatch)
    _tol = 5e-4
    if rsps > 1.0 + _tol:
        print(f"  rSPS = {rsps:.4f} > 1 — model is MORE stable to semantic perturbations")
        print("  than arbitrary noise of the same magnitude. (Desired regime.)")
    elif rsps < 1.0 - _tol:
        print(f"  rSPS = {rsps:.4f} < 1 — model is MORE sensitive to semantic perturbations.")
        print("  This is a pathological failure mode (Definition 8).")
    else:
        print(f"  rSPS ≈ 1 ({rsps:.4f}) — semantic stability indistinguishable from generic smoothness.")

    # Spectral gap
    if gap_result is not None:
        print(f"\n  Spectral Gap Analysis (Definition 5, Corollary 1):")
        print(f"    Mean restricted operator norm ||Jf||_{{A_x}} : {gap_result.mean_restricted_norm:.4f}")
        print(f"    Mean full spectral norm sigma_max(Jf)        : {gap_result.mean_full_norm:.4f}")
        print(f"    Mean normalized gap gamma-bar                : {gap_result.mean_gap:.4f}")
        if gap_result.mean_gap > 0.3:
            print("    → Strong semantic direction separation. Model is NOT maximally sensitive")
            print("      along semantic directions. (gamma-bar >> 0)")
        else:
            print("    → Weak semantic direction separation. Semantic directions are near")
            print("      worst-case sensitivity directions. (gamma-bar ≈ 0)")

    # Layer-wise profile
    if layerwise:
        print(f"\n  Layer-wise SPS Profile (Definition 7):")
        for i, val in enumerate(layerwise):
            bar = "█" * int(val * 40)
            print(f"    Layer {i:2d}:  {val:.4f}  {bar}")

    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()
