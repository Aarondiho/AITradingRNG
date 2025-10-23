import asyncio
import os
import sys
import yaml
from dotenv import load_dotenv
import logging

from engine.launch_coordinator import LaunchCoordinator
from engine.simulator_runner import SimulatorRunner
from engine.adversarial_trainer import AdversarialTrainer



# Phase-level logging (no per-tick)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def load_config():
    with open(os.path.join("config", "settings.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def run():
    cfg = load_config()
    load_dotenv()
    env = {
        "DERIV_APP_ID": os.getenv("DERIV_APP_ID", ""),
        "DERIV_API_KEY": os.getenv("DERIV_API_KEY", ""),
    }

    symbols = cfg.get("symbols", ["R_10", "R_25", "R_50"])
    base_dir = cfg.get("base_dir", "data")
    historical_mode = cfg.get("historical_mode", "candles")
    bootstrap_ticks = bool(cfg.get("bootstrap_ticks", True))
    merge_interval_seconds = int(cfg.get("merge_interval_seconds", 60))

    coordinator = LaunchCoordinator(
        symbols,
        base_dir,
        env["DERIV_APP_ID"],
        env["DERIV_API_KEY"],
        historical_mode=historical_mode,
        bootstrap_ticks=bootstrap_ticks,
    )



    await coordinator.start()

    # Phase 3: Simulator
    if cfg.get("simulator", {}).get("enabled", False):
        sim_runner = SimulatorRunner(
            symbols=["R_10", "R_25", "R_50", "R_75", "R_100"],
            base_dir="data",
            reports_dir="reports",
            config=cfg["simulator"]
        )
        sim_runner.run_once()   # one-off run
        # or schedule it periodically if you want continuous synthetic validation


    #phase 3
    
    trainer = AdversarialTrainer(seed=42)
    result = trainer.train_and_evaluate(X, y)

    print("Best adversary:", result["best_name"])
    print("AUC:", result["best_auc"])
    print("All results:", result["all_results"])


    try:
        while True:
            await asyncio.sleep(merge_interval_seconds)
            await coordinator.run_merge_features_validate()
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("Shutting down gracefully...")
    finally:
        await coordinator.stop()
        print("Stopped cleanly.")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Exited by user.")
