# engine/launch_coordinator.py
import asyncio
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional, Iterable, Callable, Any, Dict
from datetime import datetime

from api.deriv_socket import DerivSocket
from .live_tick_collector import LiveTickCollector
from .merge_pipeline import MergePipeline
from .feature_engine import FeatureEngine
from .parity_validator import ParityValidator
from .metrics_emitter import MetricsEmitter
from .quarantine import QuarantineWriter
from .parity_triage_enrich import enrich_parity
from training.promotion_audit import PromotionAudit

logger = logging.getLogger("launch_coordinator")

class LaunchCoordinator:
    def __init__(
        self,
        symbols: Iterable[str],
        base_dir: str,
        app_id: str,
        api_key: Optional[str] = None,
        strict_warn_as_fail: bool = False,
        auto_trigger_training: bool = False,
        trainer_callable: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        min_bidask_cov_for_training: float = 0.8,
    ):
        self.symbols = [s.strip().replace(" ", "_") for s in symbols]
        self.base_dir = Path(base_dir).resolve()
        self.app_id = app_id
        self.api_key = api_key

        self.live_collectors = {}
        self.mergers = {}
        self.features = {}
        self.validators = {}

        self.metrics = MetricsEmitter(base_dir=str(self.base_dir))
        self.quarantine = QuarantineWriter(base_dir=str(self.base_dir))
        self.promotion_audit = PromotionAudit()

        self.strict_warn_as_fail = strict_warn_as_fail
        self.auto_trigger_training = auto_trigger_training
        self.trainer_callable = trainer_callable
        self.min_bidask_cov_for_training = float(min_bidask_cov_for_training)

        for sym in self.symbols:
            self.live_collectors[sym] = LiveTickCollector(sym, str(self.base_dir))
            self.mergers[sym] = MergePipeline(sym, str(self.base_dir))
            self.features[sym] = FeatureEngine(sym, str(self.base_dir))
            self.validators[sym] = ParityValidator(sym, str(self.base_dir))

        self._socket: Optional[DerivSocket] = None
        logger.info("[COORD] Initialized for symbols: %s", self.symbols)

    async def start(self):
        logger.info("[COORD] Starting coordinator")
        await self._start_live_all()
        await self._create_register_and_start_live_socket()
        logger.info("[COORD] Startup complete")

    async def stop(self):
        logger.info("[COORD] Stopping coordinator")
        if self._socket:
            try:
                await self._socket.stop()
            except Exception:
                logger.exception("[COORD] Error stopping DerivSocket")
        for sym, collector in self.live_collectors.items():
            try:
                await collector.stop()
            except Exception:
                logger.exception("[COORD] Error stopping collector %s", sym)
        logger.info("[COORD] Stopped")

    async def _start_live_all(self):
        for sym, collector in self.live_collectors.items():
            try:
                await collector.start()
            except Exception:
                logger.exception("[COORD] Failed to start collector %s", sym)

    async def _create_register_and_start_live_socket(self):
        self._socket = DerivSocket(self.app_id, self.api_key)
        for sym, collector in self.live_collectors.items():
            self._socket.register(sym, collector.add_tick)
        try:
            await self._socket.start()
        except Exception:
            logger.exception("[COORD] Failed to start DerivSocket")
            raise

    async def run_features_validate_stream(self):
        logger.info("[COORD] Running features+parity cycle")
        for sym in self.symbols:
            await asyncio.sleep(0.25)
            shard_run_id = None
            try:
                # ensure merges run before feature extraction
                try:
                    created = self.mergers[sym].merge_once()
                    logger.debug("[COORD] Merge produced: %s", created)
                except Exception:
                    logger.exception("[COORD] Merge failed for %s", sym)
                shard_run_id = await self._emit_features_from_stream(sym)
            except Exception:
                logger.exception("[COORD] Features-from-stream failed for %s", sym)
            try:
                pv = self.validators[sym]
                reps = pv.run_validation_all()
                rep_map = {r.get("run_id"): r for r in reps if r.get("run_id")}
                if shard_run_id:
                    rep = rep_map.get(shard_run_id)
                    await self._handle_parity_decision(sym, shard_run_id, rep)
                else:
                    if reps:
                        latest = reps[-1]
                        await self._handle_parity_decision(sym, latest.get("run_id"), latest)
            except Exception:
                logger.exception("[COORD] Parity validation failed for %s", sym)

    async def _emit_features_from_stream(self, symbol: str) -> Optional[str]:
        src = Path(self.base_dir) / symbol / "live_agg" / "1s" / "current.ndjson"
        return await self.features[symbol].process_stream(src, shard_rows=10000, max_secs=600)

    async def _handle_parity_decision(self, symbol: str, run_id: str, parity_report: Optional[Dict[str, Any]]):
        base = Path(self.base_dir) / symbol
        run_dir = base / "features" / run_id
        if not run_dir.exists():
            logger.warning("[COORD] Run dir missing for %s %s", symbol, run_id)
            return

        training_manifest = self._build_training_manifest(symbol, run_id, parity_report)
        try:
            tm_path = base / "training" / f"{symbol}_training_manifest_{run_id}.json"
            tm_path.parent.mkdir(parents=True, exist_ok=True)
            tm_path.write_text(json.dumps(training_manifest, indent=2), encoding="utf-8")
            logger.info("[COORD] Training manifest written: %s", tm_path)
            try:
                # parity pass metric (1 == pass)
                if parity_report:
                    val = 1 if parity_report.get("pass_for_training", True) else 0
                    self.metrics.gauge("parity.last_decision", val, component="parity")
            except Exception:
                logger.exception("[COORD] Failed emitting parity pass metric")

            if self.auto_trigger_training and self.trainer_callable:
                await self._trigger_internal_training(symbol, run_id, str(tm_path))
        except Exception:
            logger.exception("[COORD] Failed writing training manifest for %s %s", symbol, run_id)

        try:
            if parity_report:
                decision = parity_report.get("decision")
                pass_for_training = parity_report.get("pass_for_training", True)
                if decision == "FAIL" and not pass_for_training:
                    logger.warning("[COORD] Parity FAIL and blocked for training for %s run %s", symbol, run_id)
                    # metrics
                    try:
                        self.metrics.counter_inc("parity.fail_count", 1, component="parity")
                        self.metrics.gauge("parity.last_decision", 0, component="parity")  # 0 == fail
                    except Exception:
                        logger.exception("[COORD] Failed emitting parity metrics")
                    # quarantine run folder for operator triage
                    try:
                        incident_dir = base / "incidents" / run_id
                        incident_dir.mkdir(parents=True, exist_ok=True)
                        self.quarantine.quarantine_run(symbol, run_id, run_dir, reason="parity_fail")
                    except Exception:
                        logger.exception("[COORD] Quarantine failed for %s %s", symbol, run_id)
                    # try to produce triage diagnostics to help ops
                    try:
                        triage_path = enrich_parity(run_dir, top_k=20)
                        logger.info("[COORD] Produced triage diagnostics: %s", triage_path)
                    except Exception:
                        logger.exception("[COORD] Parity triage enrichment failed for %s %s", symbol, run_id)
        except Exception:
            logger.exception("[COORD] Post-parity bookkeeping failed for %s %s", symbol, run_id)

    def _build_training_manifest(self, symbol: str, run_id: str, parity_report: Optional[Dict[str, Any]]):
        base = Path(self.base_dir) / symbol
        run_dir = base / "features" / run_id
        mf = next(run_dir.glob("*.manifest.json"), None)
        manifest = json.loads(mf.read_text(encoding="utf-8")) if mf else {}

        tm = {
            "symbol": symbol,
            "run_id": run_id,
            "feature_shard_path": str(run_dir / manifest.get("shard_file")) if manifest.get("shard_file") else None,
            "feature_shard_sha256": manifest.get("sha256"),
            "feature_manifest_path": str(run_dir / (manifest.get("shard_file", "").replace(".ndjson", ".manifest.json"))) if manifest.get("shard_file") else str(run_dir),
            "scalers_path": None,
            "regime_path": None,
            "parity_path": None,
            "has_bid_ask": manifest.get("has_bid_ask"),
            "bidask_coverage": manifest.get("bidask_coverage"),
            "input_shard": manifest.get("input_shard"),
            "input_shard_sha256": manifest.get("input_shard_sha256"),
            "source_file": manifest.get("source_file"),
            "start_offset": manifest.get("start_offset"),
            "end_offset": manifest.get("end_offset"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        scalers = next(run_dir.glob("*.scalers.json"), None)
        regime = next(run_dir.glob("*.regime.json"), None)
        parity_file = run_dir / "parity.json"
        if scalers and scalers.exists():
            tm["scalers_path"] = str(scalers)
        if regime and regime.exists():
            tm["regime_path"] = str(regime)
        if parity_file.exists():
            tm["parity_path"] = str(parity_file)
            try:
                p = json.loads(parity_file.read_text(encoding="utf-8"))
                tm["parity_decision"] = p.get("decision")
                tm["pass_for_training"] = p.get("pass_for_training", True)
            except Exception:
                tm["parity_decision"] = None
                tm["pass_for_training"] = True
        else:
            tm["parity_decision"] = None
            tm["pass_for_training"] = True

        return tm

    async def _trigger_internal_training(self, symbol: str, run_id: str, training_manifest_path: str):
        if not self.trainer_callable:
            logger.warning("[COORD] No trainer_callable configured; skipping internal trigger")
            return
        loop = asyncio.get_event_loop()
        def invoke(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
            except Exception:
                obj = {"symbol": symbol, "run_id": run_id}
            return self.trainer_callable(obj)
        res = await loop.run_in_executor(None, invoke, training_manifest_path)
        logger.info("[COORD] Internal trainer returned: %s", str(res))
