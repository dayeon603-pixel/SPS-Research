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
import math
import random
import sys
from pathlib import Path

import torch

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sps import (
    AdversarialEmbeddingConfig,
    AdversarialEmbeddingFamily,
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

def bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap 95% CI for the mean of `values`.

    Returns:
        (mean, lower_bound, upper_bound)
    """
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot_means = sorted(
        sum(rng.choices(values, k=n)) / n for _ in range(n_boot)
    )
    lo = int((alpha / 2) * n_boot)
    hi = int((1 - alpha / 2) * n_boot)
    return (
        round(sum(values) / n, 6),
        round(boot_means[lo], 6),
        round(boot_means[hi], 6),
    )


def delta_method_rsps_ci(
    emb_mean_s: float,
    emb_std_s: float,
    emb_n: int,
    arb_mean_s: float,
    arb_std_s: float,
    arb_n: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Delta-method 95% CI for rSPS = SPS(T_emb) / SPS(T_arb).

    Since SPS = exp(-mean_sensitivity), we have:
        log(rSPS) = mean_s_arb - mean_s_emb

    Var(log(rSPS)) ≈ std_s_emb^2 / n_emb + std_s_arb^2 / n_arb   (independence)
    SE(log(rSPS)) = sqrt(Var)

    95% CI for rSPS: exp(log(rSPS) ± z * SE)

    Returns:
        (rsps_point, rsps_lo, rsps_hi)
    """
    import math

    # z-score for two-sided CI: 1.96 for 95%, 2.576 for 99%
    z = 1.96 if abs(alpha - 0.05) < 1e-9 else (2.576 if abs(alpha - 0.01) < 1e-9 else 1.645)
    log_rsps = arb_mean_s - emb_mean_s
    var_log = (emb_std_s ** 2) / max(emb_n, 1) + (arb_std_s ** 2) / max(arb_n, 1)
    se_log = math.sqrt(var_log)
    rsps_point = math.exp(log_rsps)
    rsps_lo = math.exp(log_rsps - z * se_log)
    rsps_hi = math.exp(log_rsps + z * se_log)
    return round(rsps_point, 6), round(rsps_lo, 6), round(rsps_hi, 6)


def compute_all_spectral_gaps(
    model,
    batches: list[dict],
    t_emb,
    device: str,
) -> list[float]:
    """
    Compute the normalized spectral gap (Definition 5) for every sample
    across all batches. Returns a flat list of per-sample gap values.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel as _sdpa_kernel
        def _make_model_fn(mask):
            def _fn(inputs_embeds: torch.Tensor) -> torch.Tensor:
                with _sdpa_kernel(SDPBackend.MATH):
                    out = model(inputs_embeds=inputs_embeds, attention_mask=mask)
                return out.last_hidden_state[:, 0, :]
            return _fn
    except (ImportError, AttributeError):
        def _make_model_fn(mask):
            def _fn(inputs_embeds: torch.Tensor) -> torch.Tensor:
                out = model(inputs_embeds=inputs_embeds, attention_mask=mask)
                return out.last_hidden_state[:, 0, :]
            return _fn

    per_sample_gaps: list[float] = []
    for batch in batches:
        emb = batch["embeddings"].to(device)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        try:
            directions = t_emb.semantic_directions(emb, ids)
            fn = _make_model_fn(mask)
            result = spectral_gap(fn, emb, directions, n_probe_full=16)
            per_sample_gaps.extend(result.normalized_gap.tolist())
        except Exception as e:
            logger.debug("Spectral gap failed on batch: %s", e)
    return per_sample_gaps


def loo_spectral_gap(per_sample_gaps: list[float]) -> dict:
    """
    Leave-one-out stability for the mean spectral gap.
    Returns min/max of LOO means and whether the estimate is stable.
    """
    n = len(per_sample_gaps)
    if n < 3:
        return {"stable": False, "n": n}
    total = sum(per_sample_gaps)
    loo_means = [(total - g) / (n - 1) for g in per_sample_gaps]
    return {
        "n": n,
        "mean": round(total / n, 6),
        "loo_min": round(min(loo_means), 6),
        "loo_max": round(max(loo_means), 6),
        "loo_range": round(max(loo_means) - min(loo_means), 6),
        "stable": (max(loo_means) - min(loo_means)) < 0.05,
    }


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
    p.add_argument("--skip-gap-loo", action="store_true",
                   help="Skip spectral gap LOO stability and bootstrap CI (faster).")
    p.add_argument("--skip-adversarial", action="store_true",
                   help="Skip adversarial SPS (T_adv) worst-case upper bound (faster).")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Bootstrap resamples for spectral gap CI (default: 1000).")
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
    arb_result = estimate_arbitrary_sps(model, iter(batches), config, return_full=True)
    arb_sps = arb_result["sps"]
    rsps = relative_sps(core["sps"], arb_sps)

    # Delta-method 95% CI for rSPS (log-space, then back-transform)
    rsps_ci: tuple[float, float, float] | None = None
    try:
        rsps_ci = delta_method_rsps_ci(
            emb_mean_s=core["mean_sensitivity"],
            emb_std_s=core["std_sensitivity"],
            emb_n=core["n_samples"],
            arb_mean_s=arb_result["mean_sensitivity"],
            arb_std_s=arb_result["std_sensitivity"],
            arb_n=arb_result["n_samples"],
        )
    except Exception as _e:
        logger.warning("rSPS delta-method CI failed: %s", _e)

    # ------------------------------------------------------------------
    # Adversarial SPS  (T_adv — worst-case within A_x)
    # ------------------------------------------------------------------
    adv_sps: float | None = None
    adv_rsps: float | None = None

    if not args.skip_adversarial:
        logger.info("Estimating adversarial SPS (T_adv, worst-case semantic direction) ...")
        try:
            batch0_adv = batches[0]
            emb0_adv = batch0_adv["embeddings"].to(args.device)
            mask0_adv = batch0_adv["attention_mask"].to(args.device)

            # Build model forward function (same SDPA workaround as spectral gap)
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel as _sdpa_kernel_adv
                def _adv_fn(inputs_embeds: torch.Tensor) -> torch.Tensor:
                    with _sdpa_kernel_adv(SDPBackend.MATH):
                        out = model(inputs_embeds=inputs_embeds, attention_mask=mask0_adv)
                    return out.last_hidden_state[:, 0, :]
            except (ImportError, AttributeError):
                def _adv_fn(inputs_embeds: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
                    out = model(inputs_embeds=inputs_embeds, attention_mask=mask0_adv)
                    return out.last_hidden_state[:, 0, :]

            t_adv = AdversarialEmbeddingFamily(
                forward_fn=_adv_fn,
                embedding_layer=model.embeddings.word_embeddings,
                synonym_map=getattr(t_emb, "synonym_map", {}),
                config=AdversarialEmbeddingConfig(
                    n_directions=args.n_directions,
                    use_synonym_directions=not args.skip_wordnet,
                ),
            )
            adv_estimator = build_sps_estimator(model, t_adv, config)
            adv_core = adv_estimator.estimate(iter(batches))
            adv_sps = adv_core["sps"]
            adv_rsps = relative_sps(adv_sps, arb_sps)
            logger.info(
                "Adversarial SPS: %.4f  rSPS_adv=%.4f  (vs T_emb SPS=%.4f)",
                adv_sps, adv_rsps, core["sps"],
            )
        except Exception as e:
            logger.warning("Adversarial SPS failed: %s", e)

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
    # Spectral gap LOO + bootstrap CI  (all batches)
    # ------------------------------------------------------------------
    all_gaps: list[float] = []
    gap_ci: tuple[float, float, float] | None = None
    gap_loo: dict | None = None

    if not args.skip_gap_loo:
        logger.info("Computing spectral gap LOO stability and bootstrap CI (all batches) ...")
        try:
            all_gaps = compute_all_spectral_gaps(model, batches, t_emb, args.device)
            if all_gaps:
                gap_ci = bootstrap_ci(all_gaps, n_boot=args.n_boot, seed=args.seed)
                gap_loo = loo_spectral_gap(all_gaps)
                logger.info(
                    "Gap LOO: mean=%.4f, loo_range=[%.4f, %.4f], stable=%s",
                    gap_loo["mean"], gap_loo["loo_min"], gap_loo["loo_max"],
                    gap_loo["stable"],
                )
        except Exception as e:
            logger.warning("Spectral gap LOO/bootstrap failed: %s", e)

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
    if adv_sps is not None:
        rows.insert(1, {
            "Family": "T_adv",
            "SPS": f"{adv_sps:.4f}",
            "MeanSens": "—",
            "StdSens": "—",
            "rSPS": f"{adv_rsps:.4f}",
            "GapMean": "—",
        })
    print_results_table(rows)

    # Adversarial SPS interpretation
    if adv_sps is not None:
        adv_ratio = adv_sps / max(core["sps"], 1e-12)
        print(f"\n  Adversarial SPS Analysis (T_adv — worst-case within A_x):")
        print(f"    SPS(T_emb) = {core['sps']:.4f}  |  SPS(T_adv) = {adv_sps:.4f}  |  ratio = {adv_ratio:.3f}")
        if adv_ratio < 1.1:
            print("    → Random synonym directions are near-adversarial (ratio < 1.1).")
            print("      T_emb already probes near the worst-case semantic sensitivity.")
        elif adv_ratio < 2.0:
            print(f"    → Moderate adversarial gap ({adv_ratio:.2f}x). T_emb underestimates worst-case")
            print("      sensitivity by this factor.")
        else:
            print(f"    → Large adversarial gap ({adv_ratio:.2f}x). Random synonym directions miss")
            print("      the worst-case; adversarial perturbations are substantially harder.")

    # rSPS interpretation (compare with tolerance to avoid float display/logic mismatch)
    _tol = 5e-4
    if rsps_ci is not None:
        _, rci_lo, rci_hi = rsps_ci
        ci_str = f"  95% CI [{rci_lo:.4f}, {rci_hi:.4f}] (delta method, log-space)"
        excludes_one = (rci_hi < 1.0) or (rci_lo > 1.0)
        ci_interp = "  → CI excludes 1 — departure from generic smoothness is significant." if excludes_one else \
                    "  → CI includes 1 — cannot rule out rSPS ≈ 1 at this sample size."
    else:
        ci_str = ""
        ci_interp = ""

    if rsps > 1.0 + _tol:
        print(f"  rSPS = {rsps:.4f} > 1 — model is MORE stable to semantic perturbations")
        print("  than arbitrary noise of the same magnitude. (Desired regime.)")
    elif rsps < 1.0 - _tol:
        print(f"  rSPS = {rsps:.4f} < 1 — model is MORE sensitive to semantic perturbations.")
        print("  This is a pathological failure mode (Definition 8).")
    else:
        print(f"  rSPS ≈ 1 ({rsps:.4f}) — semantic stability indistinguishable from generic smoothness.")
    if ci_str:
        print(ci_str)
    if ci_interp:
        print(ci_interp)

    # Spectral gap
    if gap_result is not None:
        print(f"\n  Spectral Gap Analysis (Definition 5, Corollary 1):")
        print(f"    Mean restricted operator norm ||Jf||_{{A_x}} : {gap_result.mean_restricted_norm:.4f}")
        print(f"    Mean full spectral norm sigma_max(Jf)        : {gap_result.mean_full_norm:.4f}")
        print(f"    Mean normalized gap gamma-bar (batch 0)      : {gap_result.mean_gap:.4f}")
        if gap_result.mean_gap > 0.3:
            print("    → Strong semantic direction separation. Model is NOT maximally sensitive")
            print("      along semantic directions. (gamma-bar >> 0)")
        else:
            print("    → Weak semantic direction separation. Semantic directions are near")
            print("      worst-case sensitivity directions. (gamma-bar ≈ 0)")

    if gap_ci is not None:
        mean_g, lo_g, hi_g = gap_ci
        print(f"\n  Spectral Gap — All Batches (n={len(all_gaps)} samples):")
        print(f"    Mean gamma-bar          : {mean_g:.4f}")
        print(f"    Bootstrap 95% CI        : [{lo_g:.4f}, {hi_g:.4f}]  (n_boot={args.n_boot})")
        if gap_loo:
            stable_str = "✓ stable" if gap_loo["stable"] else "✗ unstable (>0.05 range)"
            print(f"    LOO stability range     : [{gap_loo['loo_min']:.4f}, {gap_loo['loo_max']:.4f}]"
                  f"  (range={gap_loo['loo_range']:.4f})  {stable_str}")

    # Layer-wise profile
    if layerwise:
        print(f"\n  Layer-wise SPS Profile (Definition 7):")
        for i, val in enumerate(layerwise):
            bar = "█" * int(val * 40)
            print(f"    Layer {i:2d}:  {val:.4f}  {bar}")

    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()
