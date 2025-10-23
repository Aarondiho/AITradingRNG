import asyncio
import json
import websockets
import logging
import numpy as np
from collections import deque
from datetime import datetime

from engine.fingerprint_extractor import FingerprintExtractor
from engine.simulator_core import SyntheticSimulator
from engine.continuous_evolver import ContinuousEvolver

logger = logging.getLogger("live_runner")


class LiveRunner:
    """
    Streams live Deriv ticks and runs simulator in shadow mode.
    Maintains rolling buffers, extracts fingerprints, and triggers evolver.
    """

    def __init__(self, app_id: str, symbols, buffer_size=500, config=None):
        self.app_id = app_id
        self.symbols = symbols
        self.buffer_size = buffer_size
        self.buffers = {sym: deque(maxlen=buffer_size) for sym in symbols}
        self.config = config or {}
        self.extractor = FingerprintExtractor()
        self.evolver = ContinuousEvolver(config=self.config)

    async def connect(self):
        url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        async with websockets.connect(url) as ws:
            # Subscribe to ticks
            for sym in self.symbols:
                await ws.send(json.dumps({"ticks": sym}))
            logger.info("[LIVE] Subscribed to %s", self.symbols)

            # Main loop
            async for msg in ws:
                data = json.loads(msg)
                if "tick" in data:
                    sym = data["tick"]["symbol"]
                    price = float(data["tick"]["quote"])
                    self.buffers[sym].append(price)
                    if len(self.buffers[sym]) == self.buffer_size:
                        await self._process_symbol(sym)

    async def _process_symbol(self, sym):
        """
        Process one symbol when buffer is full:
          - Extract real fingerprints
          - Generate synthetic ticks
          - Extract synthetic fingerprints
          - Compare and evolve if needed
        """
        real_prices = np.array(self.buffers[sym])
        real_fp = self.extractor.full_fingerprint(real_prices)

        # Generate synthetic ticks
        sim = SyntheticSimulator(sym, base_dir="data")
        sim.fit_from_real(min_records=self.config.get("min_records", 500))
        synth_records = sim.generate_ticks(
            mode=self.config.get("mode", "hybrid"),
            count=self.buffer_size,
            seed=self.config.get("seed", 42),
            bootstrap_block=self.config.get("bootstrap_block", 5),
            mixture_components=self.config.get("mixture_components", 2),
            noise_scale=self.config.get("noise_scale", 0.1),
        )
        synth_prices = np.array([r["quote"] for r in synth_records])
        synth_fp = self.extractor.full_fingerprint(synth_prices)

        # Compare and evolve
        decision, rationale = self.evolver.evaluate_and_update(sym, real_fp, synth_fp)

        # Log report
        report = {
            "symbol": sym,
            "timestamp": datetime.utcnow().isoformat(),
            "decision": decision,
            "rationale": rationale,
            "real_fp": real_fp,
            "synth_fp": synth_fp,
        }
        logger.info("[LIVE] %s report: %s", sym, decision)
        self._write_report(sym, report)

    def _write_report(self, sym, report):
        out_dir = Path("reports/live")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{sym}_live_{int(datetime.utcnow().timestamp())}.json"
        out_path.write_text(json.dumps(report, indent=2))
