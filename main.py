#!/usr/bin/env python3
"""
Root entrypoint. Starts Phase1 and optionally schedules Phase2 runner as a subprocess.

Phase2 is launched by running the training/runner.py command (default) after a
configurable delay. This keeps Phase1/Phase2 decoupled and avoids requiring a
Phase2LaunchCoordinator import.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

from engine.launch_coordinator import LaunchCoordinator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("main")


def load_config(path: str = "config/settings.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


async def _launch_phase2_subprocess(cmd: List[str], env: Dict[str, str] = None, cwd: str = None) -> int:
    logger.info("Launching Phase2 subprocess: %s", " ".join(cmd))
    env_map = os.environ.copy()
    if env:
        env_map.update(env)
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env_map, cwd=cwd)
    # stream output asynchronously while process runs
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
    except Exception:
        logger.exception("Error while streaming Phase2 subprocess output")
    rc = await proc.wait()
    logger.info("Phase2 subprocess exited with code %d", rc)
    return rc


async def run():
    load_dotenv()
    cfg = load_config()

    symbols = cfg.get("symbols", []) or ([cfg.get("symbol")] if cfg.get("symbol") else [])
    base_dir = cfg["paths"]["base_dir"]
    merge_interval_seconds = int(cfg.get("merge_interval_seconds", 5))
    historical_mode = bool(cfg.get("historical_mode", False))
    bootstrap_ticks = int(cfg.get("bootstrap_ticks", 0))
    feature_sample_lines = int(cfg.get("feature_sample_lines", 200))
    auto_trigger_training = bool(cfg.get("auto_trigger_training", False))
    min_bidask_cov = float(cfg.get("min_bidask_cov_for_training", 0.0))

    coordinator = LaunchCoordinator(
        symbols=symbols,
        base_dir=base_dir,
        app_id=os.getenv("DERIV_APP_ID", ""),
        api_key=os.getenv("DERIV_API_KEY", ""),
        historical_mode=historical_mode,
        bootstrap_ticks=bootstrap_ticks,
        feature_sample_lines=feature_sample_lines,
        auto_trigger_training=auto_trigger_training,
        min_bidask_cov_for_training=min_bidask_cov,
    )

    await coordinator.start()
    logger.info("Phase1 started")

    # schedule Phase2 runner via subprocess if requested
    if cfg.get("run_phase2", False):
        phase2_delay = int(cfg.get("phase2_start_delay_seconds", 1))
        phase2_cmd = cfg.get("phase2_runner_cmd") or ["python", "training/runner.py"]
        phase2_env = cfg.get("phase2_env") or {}
        phase2_cwd = cfg.get("phase2_cwd") or None

        async def delayed_phase2():
            logger.info("Phase2 scheduled to start in %d seconds", phase2_delay)
            await asyncio.sleep(phase2_delay)
            logger.info("Starting Phase2 subprocess")
            try:
                rc = await _launch_phase2_subprocess(phase2_cmd, env=phase2_env, cwd=phase2_cwd)
                logger.info("Phase2 subprocess finished with code %d", rc)
            except Exception:
                logger.exception("Phase2 subprocess failed")

        asyncio.create_task(delayed_phase2())
    else:
        logger.info("run_phase2 disabled in config")

    try:
        while True:
            await asyncio.sleep(merge_interval_seconds)
            try:
                await coordinator.run_features_validate_stream()
            except Exception:
                logger.exception("Phase1 periodic task failed")
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Shutting down gracefully")
    finally:
        await coordinator.stop()
        logger.info("Stopped cleanly")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Exited by user.")
        sys.exit(0)
