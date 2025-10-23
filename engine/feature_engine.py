# engine/feature_engine.py
import json
import asyncio
import os
import math
import hashlib
import logging
import uuid
from pathlib import Path
from collections import deque, Counter
from statistics import mean, median, pstdev
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger("feature_engine")

WINDOWS = [5, 30, 60]
SESSION_WINDOW = 30
REGIME_K = 3
SR_BINS = 20
ANOMALY_BASELINE = 100

def _mad(xs: List[float]) -> float:
    if not xs:
        return 0.0
    med = median(xs)
    return float(median([abs(x - med) for x in xs]))

def _atomic_write_obj(path: Path, obj: Any):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)

def _atomic_write_lines(path: Path, lines: List[str]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line if line.endswith("\n") else line + "\n")
    os.replace(tmp, path)

def _sha256_of_file(path: Path, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            h.update(block)
    return h.hexdigest()

class OnlineKMeans:
    def __init__(self, k=REGIME_K, dims=2, lr=0.05):
        self.k = k
        self.dims = dims
        self.lr = lr
        self.centers = None
        self.initialized = False

    def init_from_buffer(self, buffer_vectors: List[List[float]]):
        if not buffer_vectors:
            self.centers = [[0.0] * self.dims for _ in range(self.k)]
            self.initialized = True
            return
        uniq = []
        for v in buffer_vectors:
            if len(uniq) >= self.k:
                break
            if not any(all(abs(a - b) < 1e-12 for a, b in zip(v, u)) for u in uniq):
                uniq.append(v)
        if not uniq:
            uniq = [buffer_vectors[0]]
        while len(uniq) < self.k:
            uniq.append(buffer_vectors[len(uniq) % len(buffer_vectors)])
        self.centers = [list(u) for u in uniq[: self.k]]
        self.initialized = True

    def partial_fit(self, x: List[float]) -> int:
        if not self.initialized:
            self.init_from_buffer([x])
        best = None
        bestd = None
        for i, c in enumerate(self.centers):
            d = sum((a - b) ** 2 for a, b in zip(x, c))
            if bestd is None or d < bestd:
                bestd = d
                best = i
        c = self.centers[best]
        for j in range(self.dims):
            c[j] = c[j] + self.lr * (x[j] - c[j])
        return best

    def state(self) -> Dict[str, Any]:
        return {"k": self.k, "dims": self.dims, "lr": self.lr, "centers": self.centers}

class FeatureEngine:
    WINDOWS = WINDOWS
    FEATURE_VERSION = "v2"
    CANONICAL_PHASE1A = ["f1","f2","f3","f4","f5","f6","f7","f8","f9","f10"]

    def __init__(self, symbol: str, base_dir: str):
        self.symbol = symbol
        self.base_dir = Path(base_dir).resolve() / symbol
        self.merged_dir = self.base_dir / "merged"
        self.features_root = self.base_dir / "features"
        self.features_root.mkdir(parents=True, exist_ok=True)

        self.deques = {w: deque(maxlen=w) for w in self.WINDOWS}
        maxlen = max(self.WINDOWS + [SESSION_WINDOW, ANOMALY_BASELINE])
        self.close_buffer = deque(maxlen=maxlen)
        self.range_buffer = deque(maxlen=ANOMALY_BASELINE)

        self.sr_window = deque(maxlen=SESSION_WINDOW)
        self.sr_hist = Counter()
        self.sr_bin_edges = None

        self.regime = OnlineKMeans(k=REGIME_K, dims=2, lr=0.05)
        self.regime_init_buf: List[List[float]] = []

        logger.info("[FEAT] Initialized for %s", self.symbol)

    def _safe_get_price_fields(self, r: Dict[str, Any]):
        try:
            epoch = int(r.get("epoch") or r.get("ts") or 0)
        except Exception:
            epoch = 0
        close = None
        o = r.get("open")
        h = r.get("high")
        l = r.get("low")
        c = r.get("close")
        bid = r.get("bid")
        ask = r.get("ask")
        try:
            bid = float(bid) if bid is not None else None
        except Exception:
            bid = None
        try:
            ask = float(ask) if ask is not None else None
        except Exception:
            ask = None
        if c is not None:
            try:
                close = float(c)
            except Exception:
                close = None
        elif r.get("quote") is not None:
            try:
                close = float(r.get("quote"))
            except Exception:
                close = None
        if h is None and l is None and close is not None:
            h = close + 1e-4
            l = close - 1e-4
        elif h is not None and l is not None:
            try:
                h = float(h)
                l = float(l)
            except Exception:
                h = None
                l = None
        return epoch, close, (float(o) if o is not None else None), h, l, bid, ask

    # compute helpers (same as prior implementation)
    def compute_mid_price(self, bid, ask, quote):
        if bid is not None and ask is not None:
            try:
                return (float(bid) + float(ask)) / 2.0
            except Exception:
                return float(quote) if quote is not None else 0.0
        return float(quote) if quote is not None else 0.0

    def compute_spread(self, bid, ask):
        if bid is not None and ask is not None:
            try:
                return float(ask) - float(bid)
            except Exception:
                return 0.0
        return 0.0

    def compute_imbalance(self, bid, ask):
        if bid is not None and ask is not None and (ask + bid) != 0:
            try:
                return (ask - bid) / (ask + bid)
            except Exception:
                return 0.0
        return 0.0

    def _init_sr_bins(self):
        if not self.sr_window:
            return
        prices = list(self.sr_window)
        lo = min(prices)
        hi = max(prices)
        if hi <= lo:
            lo -= 1.0
            hi += 1.0
        self.sr_bin_edges = [lo + (hi - lo) * i / SR_BINS for i in range(SR_BINS + 1)]
        self.sr_hist = Counter()

    def _price_to_bin(self, price: float) -> int:
        if self.sr_bin_edges is None:
            self._init_sr_bins()
            if self.sr_bin_edges is None:
                return 0
        edges = self.sr_bin_edges
        if price <= edges[0]:
            return 0
        if price >= edges[-1]:
            return SR_BINS - 1
        lo = 0
        hi = len(edges) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if price < edges[mid]:
                hi = mid
            else:
                lo = mid
        return max(0, min(SR_BINS - 1, lo))

    def compute_returns_1m(self, close, prev_close):
        try:
            if prev_close is None or prev_close == 0 or close is None:
                return 0.0
            return math.log(close / prev_close)
        except Exception:
            return 0.0

    def compute_atr_like(self, high, low):
        if high is None or low is None:
            return 0.0
        r = high - low
        if len(self.range_buffer) >= 3:
            denom = _mad(list(self.range_buffer))
            if denom <= 0:
                denom = median(list(self.range_buffer)) or 1.0
        else:
            denom = r if r > 0 else 1.0
        return r / denom

    def compute_body_to_range(self, open_p, close, high, low):
        if open_p is None or close is None or high is None or low is None:
            return 0.0
        body = abs(close - open_p)
        full = max(1e-9, high - low)
        return body / full

    def compute_wick_balance(self, open_p, close, high, low):
        if open_p is None or close is None or high is None or low is None:
            return 0.0
        upper = high - max(open_p, close)
        lower = min(open_p, close) - low
        full = max(1e-9, high - low)
        return (upper - lower) / full

    def compute_momentum_3(self):
        if len(self.close_buffer) < 3:
            return 0.0
        a = list(self.close_buffer)[-3:]
        return (a[-1] - a[0]) / (a[0] if a[0] != 0 else 1.0)

    def time_of_day_enc(self, epoch):
        try:
            t = int(epoch)
            s = (t % 86400) / 86400.0
            angle = 2 * math.pi * s
            return math.sin(angle), math.cos(angle)
        except Exception:
            return 0.0, 1.0

    def session_vol_profile(self):
        if len(self.close_buffer) < 2:
            return 0.0
        subset = list(self.close_buffer)[-SESSION_WINDOW:]
        if len(subset) < 2:
            return 0.0
        try:
            return pstdev(subset)
        except Exception:
            return 0.0

    def binned_sr_density(self, price):
        if price is None:
            return 0.0
        if len(self.sr_window) == self.sr_window.maxlen:
            try:
                old = self.sr_window.popleft()
                if self.sr_bin_edges:
                    ob = self._price_to_bin(old)
                    self.sr_hist[ob] -= 1
                    if self.sr_hist[ob] <= 0:
                        del self.sr_hist[ob]
            except Exception:
                pass
        self.sr_window.append(price)
        if self.sr_bin_edges is None and len(self.sr_window) >= 3:
            self._init_sr_bins()
        if self.sr_bin_edges:
            b = self._price_to_bin(price)
            self.sr_hist[b] += 1
            denom = len(self.sr_window) if len(self.sr_window) > 0 else 1
            return float(self.sr_hist.get(b, 0)) / denom
        else:
            return 0.0

    def anomaly_score_z(self, r):
        if r is None:
            return 0.0
        baseline = list(self.range_buffer)
        if len(baseline) < 5:
            return 0.0
        med = median(baseline)
        mdev = _mad(baseline)
        denom = mdev if mdev > 0 else (median([abs(x - med) for x in baseline]) or 1.0)
        try:
            return float((r - med) / denom)
        except Exception:
            return 0.0
    
    def _compute_phase1a_features(self, recent_vals: List[float]) -> Dict[str, float]:
        """
        Compute a compact set of Phase1a features from a list of recent prices (floats).
        recent_vals is expected to be a list of price floats in ascending time order or a small buffer.
        Returns a dict with keys f2..f8 (floats or 0.0 when unavailable).
        """
        out: Dict[str, float] = {}
        try:
            vals = [float(x) for x in recent_vals if x is not None]
        except Exception:
            vals = []

        if not vals:
            # defaults
            out.update({"f2": 0.0, "f3": 0.0, "f4": 0.0, "f5": 0.0, "f6": 0.0, "f7": 0.0, "f8": 0.0})
            return out

        n = len(vals)
        last = vals[-1]
        first = vals[0]
        # basic statistics
        mean_v = sum(vals) / n
        # variance (population)
        var_v = 0.0
        if n > 1:
            var_v = sum((x - mean_v) ** 2 for x in vals) / n
        # simple momentum: last - mean, and last - first (short and medium)
        momentum_mean = last - mean_v
        momentum_total = last - first
        # simple realized variance of returns
        rets = []
        for i in range(1, n):
            prev = vals[i - 1]
            if prev == 0:
                continue
            rets.append((vals[i] - prev) / prev)
        realized_var = 0.0
        if rets:
            realized_var = sum(r * r for r in rets) / len(rets)
        # min/max range
        vmin = min(vals)
        vmax = max(vals)
        # short-term slope using last 3 points (linear approx)
        slope = 0.0
        try:
            if n >= 3:
                y0, y1, y2 = vals[-3], vals[-2], vals[-1]
                slope = ((y2 - y1) + (y1 - y0)) / 2.0
            elif n == 2:
                slope = vals[-1] - vals[-2]
        except Exception:
            slope = 0.0

        # map to f2..f8 (choose stable, interpretable mappings)
        out["f2"] = float(momentum_mean)              # short momentum relative to recent mean
        out["f3"] = float(momentum_total)             # momentum from window start
        out["f4"] = float(var_v)                      # variance in window
        out["f5"] = float(realized_var)               # realized variance of returns
        out["f6"] = float((vmax - vmin))              # amplitude range over window
        out["f7"] = float(slope)                      # short slope proxy
        out["f8"] = float(mean_v if mean_v is not None else last)  # local mean as stability anchor

        # ensure numeric and finite
        for k, v in list(out.items()):
            try:
                if v is None or not (isinstance(v, (int, float))):
                    out[k] = 0.0
                elif math.isfinite(float(v)):
                    out[k] = float(v)
                else:
                    out[k] = 0.0
            except Exception:
                out[k] = 0.0

        return out
    def _moment_stats(self, vals: List[float]):
        """
        Return (mean, variance, skew, kurtosis) for a list of numeric values.
        Defensive: returns zeros when statistics cannot be computed.
        """
        try:
            xs = [float(x) for x in vals if x is not None]
            n = len(xs)
            if n == 0:
                return 0.0, 0.0, 0.0, 0.0
            m = sum(xs) / n
            var = 0.0
            if n > 0:
                var = sum((x - m) ** 2 for x in xs) / n
            skew = 0.0
            kurt = 0.0
            if n > 1 and var > 0:
                std = var ** 0.5
                skew = sum(((x - m) / std) ** 3 for x in xs) / n
                kurt = sum(((x - m) / std) ** 4 for x in xs) / n - 3.0
            return float(m), float(var), float(skew), float(kurt)
        except Exception:
            return 0.0, 0.0, 0.0, 0.0


    async def run_features(self):
        files = list(self.merged_dir.glob("*.ndjson"))
        if not files:
            logger.info("[FEAT] No merged shards for %s", self.symbol)
            return
        def parse_range(fname: Path) -> int:
            parts = fname.stem.split("_")
            try:
                return int(parts[-1])
            except Exception:
                return 0
        latest = max(files, key=parse_range)
        await self._compute_and_flush_records(latest, input_is_stream=False, stream_meta=None)

    async def process_stream(self, source_file: Path, shard_rows: int = 10000, max_secs: int = 600) -> Optional[str]:
        source_file = Path(source_file)
        if not source_file.exists():
            logger.info("[FEAT] Source live file missing for %s: %s", self.symbol, source_file)
            return None

        state_path = self.features_root / "stream_state.json"
        last_offset = 0
        try:
            if state_path.exists():
                st = json.loads(state_path.read_text(encoding="utf-8"))
                last_offset = int(st.get("last_offset", 0))
        except Exception:
            last_offset = 0

        lines: List[str] = []
        with source_file.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i < last_offset:
                    continue
                lines.append(line)
        if not lines:
            return None

        records = []
        for line in lines:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            epoch, close, o, h, l, bid, ask = self._safe_get_price_fields(obj)
            if close is None:
                continue
            records.append({
                "epoch": epoch,
                "open": o,
                "high": h,
                "low": l,
                "close": close,
                "bid": bid,
                "ask": ask,
                "_raw": obj
            })
            if len(records) >= shard_rows:
                break

        if not records:
            return None

        records.sort(key=lambda x: int(x["epoch"]))
        if max_secs and len(records) > 1:
            t0 = records[0]["epoch"]
            cut_idx = len(records)
            for j, r in enumerate(records):
                if r["epoch"] - t0 > max_secs:
                    cut_idx = j
                    break
            records = records[:cut_idx]

        run_id = await self._compute_and_flush_records(source_file, input_is_stream=True, stream_meta={
            "source_file": source_file.name,
            "start_offset": last_offset,
            "end_offset": last_offset + len(records)
        }, override_records=records)

        try:
            if run_id:
                _atomic_write_obj(state_path, {"last_offset": int(last_offset + len(records))})
        except Exception:
            logger.exception("[FEAT] Failed writing stream state for %s", self.symbol)
        return run_id

    async def _compute_and_flush_records(
        self,
        input_path: Path,
        input_is_stream: bool,
        stream_meta: Optional[Dict[str, Any]],
        override_records: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[str]:
        if override_records is None:
            records = []
            try:
                with input_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        epoch, close, o, h, l, bid, ask = self._safe_get_price_fields(obj)
                        if close is None:
                            continue
                        records.append({
                            "epoch": epoch,
                            "open": o,
                            "high": h,
                            "low": l,
                            "close": close,
                            "bid": bid,
                            "ask": ask,
                            "_raw": obj
                        })
            except Exception:
                logger.exception("[FEAT] Failed reading %s", input_path.name)
                return None
        else:
            records = override_records

        if not records:
            logger.warning("[FEAT] No usable records in %s for %s", input_path.name, self.symbol)
            return None

        records.sort(key=lambda x: int(x["epoch"]))

        features: List[Dict[str, Any]] = []
        scalers_acc: Dict[int, List[float]] = {w: [] for w in self.WINDOWS}
        prev_close = None

        shard_bid_count = 0
        shard_ask_count = 0
        shard_bidask_count = 0
        shard_spread_sum = 0.0
        shard_mid_diff_sum = 0.0

        for i, rec in enumerate(records):
            q = float(rec["close"])
            epoch = int(rec["epoch"])
            bid = rec.get("bid")
            ask = rec.get("ask")
            try:
                bid = float(bid) if bid is not None else None
            except Exception:
                bid = None
            try:
                ask = float(ask) if ask is not None else None
            except Exception:
                ask = None

            self.close_buffer.append(q)
            if rec["high"] is not None and rec["low"] is not None:
                self.range_buffer.append(rec["high"] - rec["low"])
            for w, dq in self.deques.items():
                dq.append(q)

            feat: Dict[str, Any] = {
                "symbol": self.symbol,
                "epoch": epoch,
                "quote": q,
                "src": f"features:{self.FEATURE_VERSION}",
                "bid": bid,
                "ask": ask,
                "_raw": rec.get("_raw")
            }

            feat["bid_present"] = feat["bid"] is not None
            feat["ask_present"] = feat["ask"] is not None
            feat["bidask_present"] = (feat["bid"] is not None and feat["ask"] is not None)

            if feat["bidask_present"]:
                feat["mid_price"] = 0.5 * (feat["bid"] + feat["ask"])
                feat["spread"] = feat["ask"] - feat["bid"]
                feat["imbalance"] = (feat["quote"] - feat["mid_price"]) / max(feat["spread"], 1e-9)
                shard_bid_count += 1
                shard_ask_count += 1
                shard_bidask_count += 1
                try:
                    shard_spread_sum += float(feat["spread"])
                    shard_mid_diff_sum += float(feat["mid_price"] - feat["quote"])
                except Exception:
                    pass
            else:
                if feat["bid_present"]:
                    shard_bid_count += 1
                if feat["ask_present"]:
                    shard_ask_count += 1
                feat["mid_price"] = feat["quote"]
                feat["spread"] = 0.0
                feat["imbalance"] = 0.0

            feat["return_1m"] = self.compute_returns_1m(q, prev_close)
            feat["atr_like"] = self.compute_atr_like(rec["high"], rec["low"])
            feat["body_to_range"] = self.compute_body_to_range(
                rec["open"] if rec["open"] is not None else q, q, rec["high"], rec["low"]
            )
            feat["wick_balance"] = self.compute_wick_balance(
                rec["open"] if rec["open"] is not None else q, q, rec["high"], rec["low"]
            )
            feat["momentum_3"] = self.compute_momentum_3()
            tod_sin, tod_cos = self.time_of_day_enc(epoch)
            feat["tod_sin"] = tod_sin
            feat["tod_cos"] = tod_cos
            feat["session_vol"] = self.session_vol_profile()
            feat["sr_density"] = self.binned_sr_density(q)

            r_range = (rec["high"] - rec["low"]) if (rec["high"] is not None and rec["low"] is not None) else None
            feat["anomaly_score"] = self.anomaly_score_z(r_range)

            for w, dq in self.deques.items():
                if len(dq) >= 2:
                    vals = list(dq)
                    m, var, skew, kurt = self._moment_stats(vals)
                    feat[f"mean_{w}"] = m
                    feat[f"var_{w}"] = var
                    feat[f"momentum_{w}"] = (q - m) if (m is not None) else None
                    rets = [(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
                    feat[f"realized_var_{w}"] = (sum(r * r for r in rets) / len(rets)) if rets else None
                    feat[f"skew_{w}"] = skew
                    feat[f"kurtosis_{w}"] = kurt
                    if m is not None:
                        scalers_acc[w].append(abs(q - m))
                else:
                    feat[f"mean_{w}"] = None
                    feat[f"var_{w}"] = None
                    feat[f"momentum_{w}"] = None
                    feat[f"realized_var_{w}"] = None
                    feat[f"skew_{w}"] = None
                    feat[f"kurtosis_{w}"] = None

            largest_w = max(self.WINDOWS)
            recent_vals = list(self.deques[largest_w]) if len(self.deques[largest_w]) > 0 else [q]
            p1a = self._compute_phase1a_features(recent_vals)
            try:
                p1a["f9"] = float(feat.get("sr_density", 0.0))
            except Exception:
                p1a["f9"] = 0.0
            try:
                p1a["f10"] = float(feat.get("anomaly_score", 0.0))
            except Exception:
                p1a["f10"] = 0.0
            p1a["f1"] = float(q)
            for k in ("f2","f3","f4","f5","f6","f7","f8"):
                p1a[k] = float(p1a.get(k, 0.0)) if p1a.get(k) is not None else 0.0
            feat.update(p1a)

            vol_feature = 0.0
            if len(self.deques.get(30, [])) >= 2:
                dq30 = list(self.deques[30])
                _, var_30, _, _ = self._moment_stats(dq30)
                vol_feature = float(var_30) if var_30 is not None else 0.0
            meanret = 0.0
            if len(self.close_buffer) >= 2:
                prev = list(self.close_buffer)[-2]
                if prev != 0:
                    meanret = float((q - prev) / prev)
            vec = [vol_feature, meanret]
            if not self.regime.initialized:
                self.regime_init_buf.append(vec)
                if len(self.regime_init_buf) >= REGIME_K:
                    self.regime.init_from_buffer(self.regime_init_buf)
            cluster_id = self.regime.partial_fit(vec)
            feat["regime_cluster_id"] = int(cluster_id)

            features.append(feat)
            prev_close = q

            if (i & 0xFF) == 0:
                await asyncio.sleep(0)

        if not features:
            logger.warning("[FEAT] No features produced for %s", self.symbol)
            return None

        tmin = min(f["epoch"] for f in features)
        tmax = max(f["epoch"] for f in features)
        run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
        run_dir = self.features_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        shard_name = f"{self.symbol}_features_{tmin}_{tmax}"
        out_path = run_dir / f"{shard_name}.ndjson"
        lines = [json.dumps(r, separators=(",", ":"), ensure_ascii=False) for r in features]

        try:
            await asyncio.to_thread(_atomic_write_lines, out_path, lines)
            logger.info("[FEAT] Wrote feature shard %s rows=%d", out_path.relative_to(self.base_dir), len(features))
        except Exception:
            logger.exception("[FEAT] Failed writing feature shard %s", out_path)
            return None

        sha256 = None
        try:
            sha256 = await asyncio.to_thread(_sha256_of_file, out_path)
        except Exception:
            logger.exception("[FEAT] Failed computing sha256 for %s", out_path)
            sha256 = None

        total_rows = len(features)
        bidask_coverage = float(shard_bidask_count) / total_rows if total_rows else 0.0
        bid_coverage = float(shard_bid_count) / total_rows if total_rows else 0.0
        ask_coverage = float(shard_ask_count) / total_rows if total_rows else 0.0
        mean_spread = (shard_spread_sum / shard_bidask_count) if shard_bidask_count else None
        mean_mid_diff = (shard_mid_diff_sum / shard_bidask_count) if shard_bidask_count else None

        manifest = {
            "symbol": self.symbol,
            "run_id": run_id,
            "feature_version": self.FEATURE_VERSION,
            "windows": self.WINDOWS,
            "count": len(lines),
            "epoch_range": [tmin, tmax],
            "has_bid_ask": shard_bidask_count > 0,
            "bid_count": shard_bid_count,
            "ask_count": shard_ask_count,
            "bidask_count": shard_bidask_count,
            "bidask_coverage": round(bidask_coverage, 4),
            "bid_coverage": round(bid_coverage, 4),
            "ask_coverage": round(ask_coverage, 4),
            "mean_spread": float(mean_spread) if mean_spread is not None else None,
            "mean_mid_diff": float(mean_mid_diff) if mean_mid_diff is not None else None,
            "shard_file": out_path.name,
            "sha256": sha256,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        if input_is_stream and stream_meta:
            manifest.update({
                "source_file": stream_meta.get("source_file"),
                "start_offset": stream_meta.get("start_offset"),
                "end_offset": stream_meta.get("end_offset"),
            })
        else:
            manifest["input_shard"] = input_path.name
            input_manifest = input_path.with_suffix(input_path.suffix + ".manifest.json")
            try:
                if input_manifest.exists():
                    jm = json.loads(input_manifest.read_text(encoding="utf-8"))
                    manifest["input_shard_sha256"] = jm.get("sha256")
            except Exception:
                pass

        try:
            await asyncio.to_thread(_atomic_write_obj, run_dir / f"{shard_name}.manifest.json", manifest)
            logger.info("[FEAT] Wrote manifest for %s", shard_name)
        except Exception:
            logger.exception("[FEAT] Failed writing manifest for %s", shard_name)

        per_window_mad = {str(w): (_mad(vals) if vals else None) for w, vals in scalers_acc.items()}
        per_window_mad = {k: (v if (v is not None and v > 0.0) else None) for k, v in per_window_mad.items()}

        scalers = {
            "mad_range_baseline": _mad(list(self.range_buffer)) if self.range_buffer else None,
            "session_window": SESSION_WINDOW,
            "windows": self.WINDOWS,
            "per_window_mad": per_window_mad,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        regime_state = self.regime.state()
        try:
            await asyncio.to_thread(_atomic_write_obj, run_dir / f"{shard_name}.scalers.json", scalers)
            await asyncio.to_thread(_atomic_write_obj, run_dir / f"{shard_name}.regime.json", regime_state)
            meta = {
                "run_id": run_id,
                "symbol": self.symbol,
                "shard_name": shard_name,
                "feature_file": out_path.name,
                "manifest_file": f"{shard_name}.manifest.json",
                "scalers_file": f"{shard_name}.scalers.json",
                "regime_file": f"{shard_name}.regime.json",
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            }
            await asyncio.to_thread(_atomic_write_obj, run_dir / "metadata.json", meta)
            logger.info("[FEAT] Persisted scalers/regime/meta for %s", shard_name)
        except Exception:
            logger.exception("[FEAT] Failed persisting scalers/regime/meta for %s", shard_name)

        return run_id
