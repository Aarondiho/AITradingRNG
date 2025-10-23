import numpy as np

class MultiScaleValidator:
    """
    Validates cross-scale coherence between real and synthetic candles.
    """

    def __init__(self, tolerance=0.05):
        self.tolerance = tolerance

    def compare(self, real_candles: dict, synth_candles: dict,
                real_fp: dict, synth_fp: dict) -> dict:
        """
        Compare real vs synthetic candles and fingerprints.
        Returns:
          dict with decision and rationale.
        """
        decision = "PASS"
        rationale = []
        comparison = {}

        for h in real_candles:
            if h not in synth_candles:
                continue
            rc = real_candles[h]
            sc = synth_candles[h]

            # Volatility comparison
            rv = np.std(rc["close"] - rc["open"])
            sv = np.std(sc["close"] - sc["open"])
            diff_vol = abs(rv - sv)
            comparison.setdefault(h, {})["volatility"] = {
                "real": rv, "synthetic": sv, "diff": diff_vol
            }
            if diff_vol > self.tolerance * max(rv, 1e-6):
                decision = "FAIL"
                rationale.append(f"Horizon {h}: volatility misaligned (diff={diff_vol:.4f})")

            # Directional bias comparison
            rb = np.mean(rc["close"] > rc["open"])
            sb = np.mean(sc["close"] > sc["open"])
            diff_bias = abs(rb - sb)
            comparison[h]["direction_bias"] = {
                "real": rb, "synthetic": sb, "diff": diff_bias
            }
            if diff_bias > self.tolerance:
                decision = "FAIL"
                rationale.append(f"Horizon {h}: direction bias misaligned (diff={diff_bias:.4f})")

        return {
            "comparison": comparison,
            "decision": decision,
            "rationale": rationale,
        }
