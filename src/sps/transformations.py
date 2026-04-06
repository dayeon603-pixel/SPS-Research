"""
Semantic transformation families for SPS computation.

Implements the admissible family T from Definition 0.1 and Assumption A5.
Two concrete families:

  - EmbeddingPerturbationFamily (T_emb):
      Continuous perturbation in embedding space along synonym-derived semantic
      directions.  Differentiable; compatible with Jacobian computation.
      T_alpha^v(E) = E + alpha * v,  v in A_x.

  - SynonymSubstitutionFamily (T_syn):
      Discrete token-level replacement via WordNet synonyms.  Not differentiable,
      but the resulting embedding shift defines a valid c(T, x).

Both expose a unified `.sample(embeddings, input_ids, epsilon)` interface that
returns (perturbed_embeddings, magnitudes) for use in the sensitivity estimator.

References
----------
Kang (2026) Definitions 0.1, 0.2, 3, A5.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from sps.utils import normalize_directions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TransformationFamily(ABC):
    """
    Abstract base for an admissible semantic transformation family T.

    Every concrete family must implement `.sample`, which draws a batch of
    transformations and returns the perturbed embeddings together with the
    per-sample transformation magnitudes c(T, x) = ||Tx - x||.
    """

    @abstractmethod
    def sample(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample transformations and apply them.

        Args:
            embeddings:  Token embeddings of shape (B, seq_len, hidden_dim).
            input_ids:   Token IDs of shape (B, seq_len).
            epsilon:     Maximum allowed perturbation magnitude c(T, x) <= epsilon.

        Returns:
            perturbed:   Perturbed embeddings of shape (B, seq_len, hidden_dim).
            magnitudes:  Perturbation magnitudes c(T, x) of shape (B,).
                         Zero entries indicate that no valid transformation was
                         found for that sample (should be masked out by caller).
        """

    @abstractmethod
    def semantic_directions(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the structured semantic direction set A_x for each sample.

        Args:
            embeddings:  (B, seq_len, hidden_dim).
            input_ids:   (B, seq_len).

        Returns:
            directions:  Unit vectors of shape (B, K, seq_len, hidden_dim),
                         where K is the number of admissible directions per sample.
        """


# ---------------------------------------------------------------------------
# T_emb: Continuous embedding-space perturbation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class EmbeddingPerturbationConfig:
    """Configuration for the T_emb family."""

    n_directions: int = 8
    """Number of semantic directions K to sample per input (defines |A_x|)."""

    projection_noise_scale: float = 0.02
    """Small isotropic noise added before projection to avoid degenerate directions."""

    use_synonym_directions: bool = True
    """If True, compute directions from synonym embedding differences.
    If False, use random orthogonal directions (ablation baseline)."""


class EmbeddingPerturbationFamily(TransformationFamily):
    """
    T_emb: Continuous perturbation along semantic directions in embedding space.

    Semantic directions at input x are defined as the normalized differences
    between each token's embedding and the embedding of a randomly chosen synonym
    (or the nearest neighbor in embedding space when WordNet is unavailable).

    Formally (Definition 0.1, embedding variant):
        T_alpha^v(E) = E + alpha * v,
    where v in A_x is a unit vector in the span of synonym embedding differences,
    and alpha is drawn uniformly from (0, epsilon].

    This family is differentiable w.r.t. the input embeddings, making it the
    appropriate family for Jacobian / spectral gap computation (Theorem 2).

    Args:
        embedding_layer:  The model's token embedding layer (nn.Embedding).
        synonym_map:      Dict mapping token_id -> list[synonym_token_id].
                          If None, directions are drawn from random orthogonal
                          vectors (ablation mode; see `use_synonym_directions`).
        config:           EmbeddingPerturbationConfig.
    """

    def __init__(
        self,
        embedding_layer: nn.Embedding,
        synonym_map: Optional[dict[int, list[int]]] = None,
        config: Optional[EmbeddingPerturbationConfig] = None,
    ) -> None:
        self.embedding_layer = embedding_layer
        self.synonym_map = synonym_map or {}
        self.config = config or EmbeddingPerturbationConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample one T_emb perturbation per batch element.

        Returns a perturbation E' = E + alpha * v where:
          - v is drawn from the semantic direction set A_x (unit vector)
          - alpha ~ Uniform(0, epsilon]

        Args:
            embeddings:  (B, seq_len, hidden_dim).
            input_ids:   (B, seq_len).
            epsilon:     Perturbation radius.

        Returns:
            perturbed:   (B, seq_len, hidden_dim).
            magnitudes:  (B,).
        """
        device = embeddings.device
        B = embeddings.size(0)

        # Get K semantic directions per sample: (B, K, seq_len, hidden)
        directions = self.semantic_directions(embeddings, input_ids)  # (B, K, seq_len, h)
        K = directions.size(1)

        # Pick one direction per sample uniformly at random
        idx = torch.randint(0, K, (B,), device=device)            # (B,)
        v = directions[torch.arange(B, device=device), idx]        # (B, seq_len, h)

        # Scale: alpha ~ Uniform(0, epsilon]
        alpha = torch.empty(B, device=device).uniform_(0.0, epsilon)  # (B,)
        alpha_exp = alpha.view(B, 1, 1)

        perturbed = embeddings + alpha_exp * v                     # (B, seq_len, h)

        # c(T, x) = ||T_alpha^v(E) - E|| = alpha * ||v|| = alpha (since ||v||=1 per token)
        # We define magnitude as the total Frobenius displacement:
        magnitudes = (perturbed - embeddings).flatten(start_dim=1).norm(dim=1)  # (B,)

        return perturbed, magnitudes

    def semantic_directions(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the admissible semantic direction set A_x.

        For each token at each position, the semantic direction is the normalized
        difference between the token's embedding and a synonym's embedding.
        Directions are averaged over positions to yield a sequence-level direction.

        Args:
            embeddings:  (B, seq_len, hidden_dim).
            input_ids:   (B, seq_len).

        Returns:
            directions:  Unit vectors of shape (B, K, seq_len, hidden_dim).
        """
        B, seq_len, hidden = embeddings.shape
        device = embeddings.device
        K = self.config.n_directions

        if self.config.use_synonym_directions and self.synonym_map:
            return self._synonym_directions(embeddings, input_ids, K)
        else:
            return self._random_orthogonal_directions(B, seq_len, hidden, K, device)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _synonym_directions(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """
        Build directions from synonym embedding differences.

        For token t at position i, direction = normalize(embed(syn(t)) - embed(t)).
        Directions are aggregated across positions using mean pooling.
        """
        B, seq_len, hidden = embeddings.shape
        device = embeddings.device
        all_directions: list[torch.Tensor] = []

        for _ in range(K):
            # For each sample and position, pick a random synonym token
            syn_ids = input_ids.clone()
            for b in range(B):
                for s in range(seq_len):
                    tid = input_ids[b, s].item()
                    candidates = self.synonym_map.get(int(tid), [])
                    if candidates:
                        syn_ids[b, s] = candidates[
                            torch.randint(0, len(candidates), (1,)).item()
                        ]

            with torch.no_grad():
                syn_embeddings = self.embedding_layer(syn_ids)     # (B, seq_len, h)

            delta = syn_embeddings - embeddings                    # (B, seq_len, h)

            # Add small noise to avoid degenerate zero directions
            noise_scale = self.config.projection_noise_scale
            delta = delta + noise_scale * torch.randn_like(delta)

            # Normalize per-position, then treat as a (B, seq_len, h) direction
            # Normalize over hidden dim only; full-sequence norm applied below
            d_norm = normalize_directions(delta.flatten(start_dim=1)).view_as(delta)
            all_directions.append(d_norm.unsqueeze(1))             # (B, 1, seq_len, h)

        return torch.cat(all_directions, dim=1)                    # (B, K, seq_len, h)

    def _random_orthogonal_directions(
        self,
        B: int,
        seq_len: int,
        hidden: int,
        K: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Ablation: draw K random unit directions in embedding space.

        Uses Gram-Schmidt to ensure orthogonality when K <= hidden.
        """
        flat_dim = seq_len * hidden
        raw = torch.randn(B, K, flat_dim, device=device)

        # Gram-Schmidt orthogonalization per sample
        ortho = torch.zeros_like(raw)
        for k in range(K):
            v = raw[:, k, :]                                       # (B, flat_dim)
            for j in range(k):
                u = ortho[:, j, :]
                proj = (v * u).sum(dim=-1, keepdim=True) * u
                v = v - proj
            norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            ortho[:, k, :] = v / norm

        return ortho.view(B, K, seq_len, hidden)


# ---------------------------------------------------------------------------
# T_syn: Discrete synonym substitution
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SynonymSubstitutionConfig:
    """Configuration for the T_syn family."""

    substitution_prob: float = 0.15
    """Probability of substituting each token with a synonym."""

    max_substitutions: int = 3
    """Maximum number of token substitutions per sequence."""


class SynonymSubstitutionFamily(TransformationFamily):
    """
    T_syn: Token-level synonym substitution via a precomputed synonym map.

    This family is discrete and not differentiable.  The transformation magnitude
    c(T_syn, x) is defined as the L2 distance in embedding space between the
    original and perturbed token sequences.

    This family cannot be used directly for Jacobian computation (Theorem 2).
    Use EmbeddingPerturbationFamily for that purpose.

    Args:
        embedding_layer:  The model's token embedding layer.
        synonym_map:      Dict mapping token_id -> list[synonym_token_id].
        config:           SynonymSubstitutionConfig.
    """

    def __init__(
        self,
        embedding_layer: nn.Embedding,
        synonym_map: dict[int, list[int]],
        config: Optional[SynonymSubstitutionConfig] = None,
    ) -> None:
        if not synonym_map:
            raise ValueError(
                "SynonymSubstitutionFamily requires a non-empty synonym_map. "
                "Build one via `build_wordnet_synonym_map` in this module."
            )
        self.embedding_layer = embedding_layer
        self.synonym_map = synonym_map
        self.config = config or SynonymSubstitutionConfig()

    def sample(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply synonym substitutions and return perturbed embeddings.

        Substitutions are rejected if the resulting c(T_syn, x) > epsilon.

        Args:
            embeddings:  (B, seq_len, hidden_dim).
            input_ids:   (B, seq_len).
            epsilon:     Maximum allowed perturbation magnitude.

        Returns:
            perturbed:   (B, seq_len, hidden_dim).
            magnitudes:  (B,).  Zero where no valid substitution was found.
        """
        B, seq_len, _ = embeddings.shape
        device = embeddings.device

        perturbed_ids = input_ids.clone()

        for b in range(B):
            n_substituted = 0
            positions = torch.randperm(seq_len).tolist()
            for pos in positions:
                if n_substituted >= self.config.max_substitutions:
                    break
                tid = input_ids[b, pos].item()
                candidates = self.synonym_map.get(int(tid), [])
                if candidates:
                    syn = candidates[torch.randint(0, len(candidates), (1,)).item()]
                    perturbed_ids[b, pos] = syn
                    n_substituted += 1

        with torch.no_grad():
            perturbed_embeddings = self.embedding_layer(perturbed_ids)  # (B, seq_len, h)

        magnitudes = (perturbed_embeddings - embeddings).flatten(start_dim=1).norm(dim=1)

        # Reject samples where perturbation exceeds epsilon
        valid = magnitudes <= epsilon
        perturbed_embeddings = torch.where(
            valid.view(B, 1, 1).expand_as(embeddings),
            perturbed_embeddings,
            embeddings,
        )
        magnitudes = magnitudes * valid.float()

        return perturbed_embeddings, magnitudes

    def semantic_directions(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute synonym-difference directions (same as T_emb synonym directions).

        This is provided for completeness; note that T_syn is discrete and these
        directions are not the tangent vectors of a smooth curve through input space.
        """
        B, seq_len, hidden = embeddings.shape
        device = embeddings.device
        K = 1
        all_dirs: list[torch.Tensor] = []

        syn_ids = input_ids.clone()
        for b in range(B):
            for s in range(seq_len):
                tid = input_ids[b, s].item()
                candidates = self.synonym_map.get(int(tid), [])
                if candidates:
                    syn_ids[b, s] = candidates[
                        torch.randint(0, len(candidates), (1,)).item()
                    ]
        with torch.no_grad():
            syn_emb = self.embedding_layer(syn_ids)
        delta = syn_emb - embeddings
        delta = delta + self.config.substitution_prob * torch.randn_like(delta)
        d_norm = normalize_directions(delta.flatten(start_dim=1)).view_as(delta)
        all_dirs.append(d_norm.unsqueeze(1))
        return torch.cat(all_dirs, dim=1)                          # (B, 1, seq_len, h)


# ---------------------------------------------------------------------------
# Synonym map construction (WordNet via NLTK)
# ---------------------------------------------------------------------------

def build_wordnet_synonym_map(
    tokenizer,
    vocab_size: Optional[int] = None,
    max_synonyms_per_token: int = 5,
) -> dict[int, list[int]]:
    """
    Build a synonym map from WordNet for a given tokenizer vocabulary.

    Maps token_id -> list of synonym token_ids (non-empty entries only).
    Only tokens that decode to single words with WordNet entries are included.

    Requires: nltk, nltk.corpus.wordnet downloaded.

    Args:
        tokenizer:              HuggingFace tokenizer instance.
        vocab_size:             Limit to first vocab_size tokens. None = full vocab.
        max_synonyms_per_token: Cap on synonyms per token to bound memory.

    Returns:
        synonym_map: Dict[int, List[int]] of token_id -> synonym token_ids.

    Raises:
        ImportError: If nltk is not installed.
        LookupError: If WordNet corpus is not downloaded.
    """
    try:
        from nltk.corpus import wordnet as wn
    except ImportError as e:
        raise ImportError(
            "nltk is required for WordNet synonym maps. "
            "Install with: pip install nltk && python -m nltk.downloader wordnet"
        ) from e

    vocab_size = vocab_size or tokenizer.vocab_size
    synonym_map: dict[int, list[int]] = {}

    for token_id in range(vocab_size):
        word = tokenizer.decode([token_id]).strip()
        if not word.isalpha() or len(word) < 2:
            continue

        synsets = wn.synsets(word)
        if not synsets:
            continue

        synonym_tokens: list[int] = []
        seen: set[str] = {word}
        for synset in synsets:
            for lemma in synset.lemmas():
                syn_word = lemma.name().replace("_", " ")
                if syn_word in seen:
                    continue
                seen.add(syn_word)
                enc = tokenizer.encode(syn_word, add_special_tokens=False)
                if len(enc) == 1:                                   # single-token synonym only
                    syn_id = enc[0]
                    if syn_id != token_id:
                        synonym_tokens.append(syn_id)
                        if len(synonym_tokens) >= max_synonyms_per_token:
                            break
            if len(synonym_tokens) >= max_synonyms_per_token:
                break

        if synonym_tokens:
            synonym_map[token_id] = synonym_tokens

    logger.info(
        "Built WordNet synonym map: %d / %d tokens have synonyms.",
        len(synonym_map),
        vocab_size,
    )
    return synonym_map
