import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger("simulator_core")


class SyntheticSimulator:
    """
    Multi-mode synthetic tick generator:
      - parametric: Gaussian or Gaussian mixture fit to real quotes
      - bootstrap: resamples real quotes preserving short-term structure
      - hybrid: parametric core + FFT-shaped noise (optional periodic component)

    Determinism:
      - Fixed seeds for all random draws
      - Sorted epochs and stable file outputs
    Provenance:
      - Writes a manifest with parameters, input shards, and hash metadata

    Inputs:
      - base_dir: project data root (expects data/<symbol>/features/*.ndjson)
    Outputs:
      - data/<symbol>/synthetic/<symbol>_synthetic_<tmin>_<tmax>.ndjson
      - data/<symbol>/synthetic/<symbol>_synthetic_<tmin>_<tmax>.manifest.json
    """

    def __init__(self, symbol: str, base_dir: str):
        self.symbol = symbol
        self.base_dir = Path(base_dir).resolve() / symbol
        self.features_dir = self.base_dir / "features"
        self.out_dir = self.base_dir / "synthetic"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Fitted parameters (populated by fit_from_real)
        self.mu: Optional[float] = None
        self.sigma: Optional[float] = None
        self.mix_weights: Optional[np.ndarray] = None
        self.mix_means: Optional[np.ndarray] = None
        self.mix_stds: Optional[np.ndarray] = None

        # Real quotes cache
        self.real_quotes: Optional[np.ndarray] = None
        self.real_epochs: Optional[np.ndarray] = None
        self.input_shard_name: Optional[str] = None

    def _load_latest_features(self, min_records: int = 500) -> Tuple[np.ndarray, np.ndarray, str]:
        files = sorted(self.features_dir.glob("*.ndjson"))
        if not files:
            raise FileNotFoundError(f"No feature shards found for {self.symbol}")

        latest = files[-1]
        quotes, epochs = [], []
        with latest.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if "quote" in obj and "epoch" in obj:
                    quotes.append(float(obj["quote"]))
                    epochs.append(int(obj["epoch"]))

        if len(quotes) < min_records:
            raise ValueError(f"Not enough records ({len(quotes)}) in {latest.name} for {self.symbol}")

        # Sort by epoch for determinism
        sort_idx = np.argsort(epochs)
        quotes = np.array(quotes, dtype=np.float64)[sort_idx]
        epochs = np.array(epochs, dtype=np.int64)[sort_idx]
        return quotes, epochs, latest.name

    def fit_from_real(self, mixture_components: int = 0, seed: int = 42, min_records: int = 500):
        """
        Fit parametric parameters from latest real features:
          - If mixture_components == 0: fit Gaussian (mu, sigma)
          - If mixture_components > 0: fit a simple Gaussian mixture via KMeans + per-cluster stats
        """
        np.random.seed(seed)
        quotes, epochs, shard_name = self._load_latest_features(min_records=min_records)
        self.real_quotes = quotes
        self.real_epochs = epochs
        self.input_shard_name = shard_name

        mu = float(np.mean(quotes))
        sigma = float(np.std(quotes))
        self.mu, self.sigma = mu, sigma

        if mixture_components and mixture_components > 1:
            # KMeans init for mixture clustering (deterministic)
            k = mixture_components
            # Initialize centroids by quantiles for stability
            centroids = np.quantile(quotes, np.linspace(0.1, 0.9, k))
            # Lloyd's algorithm, fixed iterations
            for _ in range(10):
                dists = np.abs(quotes[:, None] - centroids[None, :])
                labels = np.argmin(dists, axis=1)
                for j in range(k):
                    cluster = quotes[labels == j]
                    if len(cluster) > 0:
                        centroids[j] = np.mean(cluster)

            # Final labels and per-cluster stats
            dists = np.abs(quotes[:, None] - centroids[None, :])
            labels = np.argmin(dists, axis=1)

            mix_means, mix_stds, mix_weights = [], [], []
            for j in range(k):
                cluster = quotes[labels == j]
                if len(cluster) > 1:
                    mix_means.append(float(np.mean(cluster)))
                    mix_stds.append(float(np.std(cluster)))
                    mix_weights.append(float(len(cluster) / len(quotes)))
                else:
                    # Fallback to global stats for tiny clusters
                    mix_means.append(mu)
                    mix_stds.append(max(sigma, 1e-9))
                    mix_weights.append(0.0)

            # Normalize weights
            sw = sum(mix_weights)
            if sw <= 0:
                mix_weights = [1.0 / k] * k
            else:
                mix_weights = [w / sw for w in mix_weights]

            self.mix_means = np.array(mix_means, dtype=np.float64)
            self.mix_stds = np.array(mix_stds, dtype=np.float64)
            self.mix_weights = np.array(mix_weights, dtype=np.float64)
        else:
            self.mix_means = None
            self.mix_stds = None
            self.mix_weights = None

        logger.info("[SIM] Fitted params for %s: mu=%.6f sigma=%.6f comps=%d",
                    self.symbol, self.mu, self.sigma, mixture_components)

    def _sample_parametric(self, count: int, seed: int = 42) -> np.ndarray:
        """
        Sample parametric quotes using fitted parameters.
          - If mixture params exist, sample via categorical mixture
          - Else sample Gaussian with (mu, sigma)
        """
        if self.mu is None or self.sigma is None:
            raise RuntimeError("Simulator not fitted. Call fit_from_real() first.")
        np.random.seed(seed)

        if self.mix_means is not None and self.mix_weights is not None and self.mix_stds is not None:
            k = len(self.mix_weights)
            comps = np.random.choice(np.arange(k), size=count, p=self.mix_weights)
            quotes = np.random.normal(self.mix_means[comps], self.mix_stds[comps])
        else:
            quotes = np.random.normal(self.mu, self.sigma, size=count)

        return quotes.astype(np.float64)

    def _sample_bootstrap(self, count: int, block: int = 5, seed: int = 42) -> np.ndarray:
        """
        Block bootstrap: resample contiguous blocks from real quotes to preserve short-term structure.
        """
        if self.real_quotes is None:
            raise RuntimeError("Real quotes not loaded. Call fit_from_real() first.")
        np.random.seed(seed)

        n = len(self.real_quotes)
        if n < block:
            raise ValueError("Not enough real quotes for block bootstrap")
        blocks = []
        while sum(len(b) for b in blocks) < count:
            start = np.random.randint(0, n - block + 1)
            blocks.append(self.real_quotes[start:start + block])
        out = np.concatenate(blocks)[:count]
        return out.astype(np.float64)

    def _apply_fft_noise(self, quotes: np.ndarray, noise_scale: float = 0.1, seed: int = 42) -> np.ndarray:
        """
        Inject FFT-shaped noise:
          - Compute FFT of quotes (detrended)
          - Randomize phases, scale magnitudes by noise_scale
          - Add the inverse FFT back to the quotes
        """
        np.random.seed(seed)
        q = quotes.astype(np.float64)
        detrended = q - np.mean(q)
        fft_vals = np.fft.rfft(detrended)
        magnitudes = np.abs(fft_vals)
        phases = np.angle(fft_vals)

        # Random new phases
        new_phases = np.random.uniform(-np.pi, np.pi, size=len(phases))
        shaped = magnitudes * np.exp(1j * new_phases) * noise_scale
        noise = np.fft.irfft(shaped, n=len(q))
        out = q + noise
        return out.astype(np.float64)

    def generate_ticks(self,
                       mode: str,
                       count: int,
                       seed: int = 42,
                       bootstrap_block: int = 5,
                       mixture_components: int = 0,
                       noise_scale: float = 0.1) -> List[Dict]:
        """
        Generate a list of synthetic tick dicts with epoch and quote.

        Modes:
          - "parametric": sample via Gaussian or mixture (requires fit_from_real)
          - "bootstrap": block bootstrap from real quotes (requires fit_from_real)
          - "hybrid": parametric core + FFT-shaped noise

        Returns:
          - List of dicts: {"symbol": str, "epoch": int, "quote": float, "src": "synthetic:v1"}
        """
        if self.real_epochs is None:
            raise RuntimeError("Real epochs not loaded. Call fit_from_real() first.")
        np.random.seed(seed)

        # Determine epoch span (use last 'count' seconds based on real cadence)
        # If epochs are 1s cadence, we replicate a contiguous range
        base_epoch = int(self.real_epochs[-1])  # latest real epoch
        epochs = np.arange(base_epoch + 1, base_epoch + 1 + count, dtype=np.int64)

        if mode == "parametric":
            samples = self._sample_parametric(count, seed=seed)
        elif mode == "bootstrap":
            samples = self._sample_bootstrap(count, block=bootstrap_block, seed=seed)
        elif mode == "hybrid":
            core = self._sample_parametric(count, seed=seed)
            samples = self._apply_fft_noise(core, noise_scale=noise_scale, seed=seed)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Build records
        records = []
        for i in range(count):
            records.append({
                "symbol": self.symbol,
                "epoch": int(epochs[i]),
                "quote": float(samples[i]),
                "src": "synthetic:v1"
            })
        return records

    def write_shard(self, records: List[Dict]) -> Tuple[Path, Path]:
        """
        Write NDJSON shard and manifest with provenance.
        """
        if not records:
            raise ValueError("No records to write")

        epochs = [int(r["epoch"]) for r in records]
        tmin, tmax = min(epochs), max(epochs)
        shard_id = f"{self.symbol}_synthetic_{tmin}_{tmax}"
        out_path = self.out_dir / f"{shard_id}.ndjson"

        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        manifest = {
            "symbol": self.symbol,
            "count": len(records),
            "epoch_range": [tmin, tmax],
            "input_shard": self.input_shard_name,
            "fitted": {
                "mu": self.mu,
                "sigma": self.sigma,
                "mix_weights": self.mix_weights.tolist() if self.mix_weights is not None else None,
                "mix_means": self.mix_means.tolist() if self.mix_means is not None else None,
                "mix_stds": self.mix_stds.tolist() if self.mix_stds is not None else None,
            },
            "timestamp": datetime.utcnow().isoformat(),
            "src": "simulator_core:v1"
        }
        manifest_path = self.out_dir / f"{shard_id}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        logger.info("[SIM] Wrote synthetic shard %s for %s", shard_id, self.symbol)
        return out_path, manifest_path
