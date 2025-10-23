import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import ks_2samp, chisquare

logger = logging.getLogger("fingerprint_analyzer")


class FingerprintAnalyzer:
    """
    Statistical fingerprinting of RNG ticks:
    - Autocorrelation at configured lags
    - FFT spectrum analysis
    - KS test vs normal distribution
    - Chi-square test on binned values
    - Optional cross-symbol correlation
    """

    def __init__(self, symbol: str, base_dir: str, reports_dir: str, config: dict):
        self.symbol = symbol
        self.base_dir = Path(base_dir).resolve() / symbol / "features"
        self.reports_dir = Path(reports_dir).resolve() / "fingerprint" / symbol
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

    def run_analysis(self, cross_data: dict = None):
        files = sorted(self.base_dir.glob("*.ndjson"))
        if not files:
            logger.info("[FPRINT] No feature shards for %s", self.symbol)
            return

        latest = files[-1]
        quotes = []
        try:
            with latest.open("r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if "quote" in obj:
                        quotes.append(float(obj["quote"]))
        except Exception:
            logger.exception("[FPRINT] Failed reading %s", latest.name)
            return

        if len(quotes) < self.config.get("min_records", 500):
            logger.info("[FPRINT] Not enough records (%d) for %s", len(quotes), self.symbol)
            return

        quotes = np.array(quotes, dtype=np.float64)
        n = len(quotes)
        tmin, tmax = int(min(quotes)), int(max(quotes))

        # --- Autocorrelation ---
        autocorr_map = {}
        for lag in self.config.get("lags", [1, 5, 10]):
            if lag < n:
                corr = np.corrcoef(quotes[:-lag], quotes[lag:])[0, 1]
                autocorr_map[f"lag{lag}"] = float(corr)

        # --- FFT spectrum ---
        detrended = quotes - np.mean(quotes)
        fft_vals = np.fft.rfft(detrended * np.hanning(len(detrended)))
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(detrended), d=1.0)
        top_k = self.config.get("fft_top_k", 5)
        idx = np.argsort(power[1:])[-top_k:] + 1  # exclude DC
        fft_peaks = [{"freq": float(freqs[i]), "power": float(power[i])} for i in idx]

        # --- KS test vs normal ---
        normed = (quotes - np.mean(quotes)) / np.std(quotes)
        ks_stat, ks_p = ks_2samp(normed, np.random.normal(0, 1, size=len(normed)))

        # --- Chi-square test ---
        bins = self.config.get("ks_bins", 50)
        hist, edges = np.histogram(normed, bins=bins)
        expected = np.full_like(hist, np.mean(hist))
        chi2_stat, chi2_p = chisquare(hist, expected)

        # --- Cross-symbol correlation ---
        cross_corrs = {}
        if cross_data:
            for other_symbol, other_quotes in cross_data.items():
                m = min(len(quotes), len(other_quotes))
                if m > 0:
                    corr = np.corrcoef(quotes[:m], other_quotes[:m])[0, 1]
                    cross_corrs[f"{self.symbol}_vs_{other_symbol}"] = float(corr)

        # --- Decision rules ---
        decision = "PASS"
        if any(abs(v) > 0.05 for v in autocorr_map.values()):
            decision = "FAIL"
        if any(p["power"] / np.sum(power) > 0.2 for p in fft_peaks):
            decision = "FAIL"
        if ks_p < 0.01 or chi2_p < 0.01:
            decision = "FAIL"
        if any(abs(v) > 0.1 for v in cross_corrs.values()):
            decision = "FAIL"

        # --- Report ---
        report = {
            "symbol": self.symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "count": n,
            "autocorr": autocorr_map,
            "fft_top_peaks": fft_peaks,
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
            "chi2_stat": float(chi2_stat),
            "chi2_p": float(chi2_p),
            "cross_symbol_corrs": cross_corrs,
            "decision": decision,
            "parameters": self.config,
            "input_shard": latest.name,
        }

        out_path = self.reports_dir / f"{self.symbol}_fprint_{int(datetime.utcnow().timestamp())}.json"
        out_path.write_text(json.dumps(report, indent=2))
        logger.info("[FPRINT] Wrote fingerprint report for %s: %s", self.symbol, decision)
