# api/deriv_socket.py
import asyncio
import json
import logging
from typing import Callable, Dict, Optional

import websockets

logger = logging.getLogger("deriv_socket")


class DerivSocket:
    def __init__(self, app_id: str, api_key: Optional[str] = None, max_retries: int = 6):
        self.app_id = app_id
        self.api_key = api_key
        self._ws = None
        self._callbacks: Dict[str, Callable[[dict], None]] = {}
        self._running = False
        self._ready_event = asyncio.Event()
        self._recv_task = None
        self.max_retries = max_retries
        self._connect_lock = asyncio.Lock()

    def register(self, symbol: str, on_tick: Callable[[dict], None]):
        self._callbacks[symbol] = on_tick

    async def start(self):
        asyncio.create_task(self._connect_with_backoff())
        await asyncio.wait_for(self._ready_event.wait(), timeout=15)
        logger.info("[DERIV] Live socket ready (connected and subscribed)")

    async def stop(self):
        logger.info("[DERIV] Stopping live socket")
        self._running = False
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                logger.exception("[DERIV] Error closing websocket")
            self._ws = None
        self._ready_event.clear()
        logger.info("[DERIV] Live socket stopped")

    async def _connect_with_backoff(self):
        async with self._connect_lock:
            attempt = 0
            backoff = 1.0
            while attempt < self.max_retries:
                try:
                    url = f"wss://ws.derivws.com/websockets/v3?app_id={self.app_id}"
                    logger.info("[DERIV] Connecting (attempt %d)", attempt + 1)
                    self._ws = await websockets.connect(url)
                    self._running = True

                    if self.api_key:
                        await self._ws.send(json.dumps({"authorize": self.api_key}))
                        auth = json.loads(await self._ws.recv())
                        logger.info("[DERIV] Authorized: %s", auth.get("msg_type"))

                    for sym in self._callbacks:
                        await self._ws.send(json.dumps({"ticks": sym, "subscribe": 1}))
                    logger.info("[DERIV] Subscribed to %d symbols", len(self._callbacks))

                    self._ready_event.set()
                    if not self._recv_task or self._recv_task.done():
                        self._recv_task = asyncio.create_task(self._recv_loop())
                    return
                except Exception as exc:
                    attempt += 1
                    self._ready_event.clear()
                    logger.exception("[DERIV] Connect attempt %d failed: %s", attempt, exc)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)

            logger.error("[DERIV] Failed to connect after %d attempts", attempt)
            raise ConnectionError("DerivSocket failed to connect after retries")

    async def _recv_loop(self):
        try:
            while self._running and self._ws:
                try:
                    msg = await self._ws.recv()
                except websockets.ConnectionClosed:
                    logger.warning("[DERIV] WebSocket closed, scheduling reconnect")
                    self._ready_event.clear()
                    asyncio.create_task(self._connect_with_backoff())
                    return

                try:
                    data = json.loads(msg)
                except Exception:
                    logger.exception("[DERIV] Failed parsing message")
                    continue

                tick = self._normalize_tick(data)
                if not tick:
                    continue

                cb = self._callbacks.get(tick["symbol"])
                if cb:
                    try:
                        cb(tick)
                    except Exception:
                        logger.exception("[DERIV] Callback exception for %s", tick.get("symbol"))
        except asyncio.CancelledError:
            logger.info("[DERIV] Recv loop cancelled")
            raise
        except Exception:
            logger.exception("[DERIV] Recv loop unexpected error")
            self._ready_event.clear()
            asyncio.create_task(self._connect_with_backoff())

    def _normalize_tick(self, data: dict) -> Optional[dict]:
        try:
            tick = data.get("tick")
            if not tick:
                return None
            return {
                "symbol": tick["symbol"],
                "quote": float(tick.get("quote") or tick.get("quote")),
                "epoch": int(tick["epoch"]),
                "epoch_ms": int(tick["epoch"]) * 1000,
                "ask": float(tick.get("ask")) if tick.get("ask") is not None else None,
                "bid": float(tick.get("bid")) if tick.get("bid") is not None else None,
                "src": "deriv",
                "schema": "tick.v1",
            }
        except Exception:
            logger.exception("[DERIV] Failed to normalize tick")
            return None
