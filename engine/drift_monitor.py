import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import ks_2samp, entropy

logger = logging.getLogger("drift_monitor")


class DriftMonitor:
    """
    Drift detection engine:
      - Loads latest feature shard
      - Compares against baseline JSON
      - Computes drift metrics (KL, KS, autocorr delta, FFT shift)
      - Writes drift report
    """

    def __init__(self, symbol: str, base_dir: str, reports_dir: str, config: dict):
        self.symbol = symbol
        self.base_dir = Path(base_dir).resolve() / symbol / "features"
        self.baseline_dir = Path(base_dir).resolve().parent / "baselines"
        self.reports_dir = Path(reports_dir).resolve() / "drift" / symbol
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

    def _load_latest_quotes(self, min_records=500):
        files = sorted(self.base_dir.glob("*.ndjson"))
        if not files:
            raise FileNotFoundError(f"No feature shards for {self.symbol}")
        latest = files[-1]
        quotes = []
        with latest.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if "quote" in obj:
                    quotes.append(float(obj["quote"]))
        if len(quotes) < min_records:
            raise ValueError(f"Not enough records ({len(quotes)}) for {self.symbol}")
        return np.array(quotes, dtype=np.float64), latest.name

    def _load_baseline(self):
        baseline_path = self.baseline_dir / f"{self.symbol}_baseline.json"
        if not baseline_path.exists():
            raise FileNotFoundError(f"No baseline for {self.symbol}")
        return json.loads(baseline_path.read_text()), baseline_path.name

    def run_check(self):
        quotes, shard_name = self._load_latest_quotes(self.config.get("min_records", 500))
        baseline, baseline_file = self._load_baseline()

        # --- Distributional drift ---
        hist, edges = np.histogram(quotes, bins=baseline["hist_bins"], density=True)
        baseline_hist = np.array(baseline["hist_counts"], dtype=np.float64)
        baseline_hist /= baseline_hist.sum()
        hist = hist / hist.sum()
        kl_div = float(entropy(hist + 1e-12, baseline_hist + 1e-12))
        ks_stat, ks_p = ks_2samp(quotes, np.array(baseline["sample_quotes"], dtype=np.float64))

        # --- Autocorr drift ---
        autocorr_delta = {}
        for lag, base_val in baseline["autocorr"].items():
            lag_int = int(lag.replace("lag", ""))
            if lag_int < len(quotes):
                corr = np.corrcoef(quotes[:-lag_int], quotes[lag_int:])[0, 1]
                autocorr_delta[lag] = float(abs(corr - base_val))

        # --- FFT drift ---
        detrended = quotes - np.mean(quotes)
        fft_vals = np.fft.rfft(detrended * np.hanning(len(detrended)))
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(detrended), d=1.0)
        idx = np.argmax(power[1:]) + 1
        top_freq, top_power = float(freqs[idx]), float(power[idx])
        fft_shift = abs(top_freq - baseline["fft_peak"]["freq"])

        # --- Decision rules ---
        decision = "PASS"
        rationale = []
        if kl_div > self.config.get("kl_threshold", 0.1):
            decision = "FAIL"
            rationale.append("KL divergence too high")
        if ks_p < self.config.get("ks_p_threshold", 0.01):
            decision = "FAIL"
            rationale.append("KS test p-value too low")
        if any(delta > self.config.get("autocorr_threshold", 0.05) for delta in autocorr_delta.values()):
            decision = "FAIL"
            rationale.append("Autocorr drift exceeded")
        if fft_shift > self.config.get("fft_threshold", 0.05):
            decision = "FAIL"
            rationale.append("FFT peak shifted")

        # --- Report ---
        report = {
            "symbol": self.symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "input_shard": shard_name,
            "baseline_file": baseline_file,
            "kl_div": kl_div,
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
            "autocorr_delta": autocorr_delta,
            "fft_peak": {"freq": top_freq, "power": top_power},
            "fft_shift": fft_shift,
            "decision": decision,
            "rationale": rationale,
            "parameters": self.config,
        }

        out_path = self.reports_dir / f"{self.symbol}_drift_{int(datetime.utcnow().timestamp())}.json"
        out_path.write_text(json.dumps(report, indent=2))
        logger.info("[DRIFT] Wrote drift report for %s: %s", self.symbol, decision)
