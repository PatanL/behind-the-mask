# Load test the live server: stages of N simulated viewers (40% of them tap), held 40 s each. A few probe viewers,
# connected first, time the gaps between word batches. Run several copies (PROBES=0, distinct TAG) to go past one
# client process. BTM_WS=ws://<host>:4340/live/ws python server/loadtest.py 100,300,600,1000,1500
import asyncio, json, os, random, sys, time, websockets, statistics as st
URL = os.environ.get("BTM_WS", "ws://127.0.0.1:8765/live/ws")
STAGES = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "100,300,600,1000,1500").split(",")]
HOLD = 40
PROBES = int(os.environ.get("PROBES", "3")); TAG = os.environ.get("TAG", "")
BTN = ["joy", "sadness", "anger", "fear", "calm", "curiosity", "unmask"]
stats = {"bytes": 0, "drops": 0, "full": 0, "lite": 0, "conns": 0}
probe_gaps = []; probe_words = [0]

async def viewer(i, probe, stop):
    try:
        async with websockets.connect(f"{URL}?cid=load{i}", max_size=2**22, open_timeout=30, ping_interval=None) as ws:
            stats["conns"] += 1
            tapper = (not probe) and random.random() < 0.4
            fav = random.choice(BTN)
            async def taps():
                while not stop.is_set():
                    await asyncio.sleep(0.4)
                    if random.random() < 0.6:
                        b = fav if random.random() < 0.7 else random.choice(BTN)
                        await ws.send(json.dumps({"type": "taps", "c": {b: random.randint(1, 3)}}))
            t = asyncio.create_task(taps()) if tapper else None
            last = None
            while not stop.is_set():
                try: raw = await asyncio.wait_for(ws.recv(), timeout=2)
                except asyncio.TimeoutError: continue
                stats["bytes"] += len(raw)
                if probe and raw.startswith('{"type":"ws"'):
                    now = time.time()
                    if last: probe_gaps.append(now - last)
                    if i == "probe0": probe_words[0] += raw.count('"t":')
                    last = now
                elif '"full"' in raw[:20]: stats["full"] += 1; break
                elif '"mode"' in raw[:20]: stats["lite"] += 1
            if t: t.cancel()
    except Exception:
        stats["drops"] += 1
    finally:
        stats["conns"] -= 1

async def main():
    stop = asyncio.Event(); tasks = []
    for k in range(PROBES):
        tasks.append(asyncio.create_task(viewer(f"probe{k}", True, stop)))
    n = 0
    for N in STAGES:
        while n < N:                                     # ramp up ~100 viewers a second
            tasks.append(asyncio.create_task(viewer(f"{TAG}{n}", False, stop))); n += 1
            if n % 100 == 0: await asyncio.sleep(1)
        probe_gaps.clear(); probe_words[0] = 0; b0, t0 = stats["bytes"], time.time()
        await asyncio.sleep(HOLD)
        g = sorted(probe_gaps)
        rate = (stats["bytes"] - b0) / (time.time() - t0) / max(1, stats["conns"])
        q = lambda p: g[min(len(g) - 1, int(p * len(g)))] if g else float('nan')
        print(f"[{time.strftime('%H:%M:%S')}] {N:5d} viewers: connected {stats['conns']}, drops {stats['drops']}, lite {stats['lite']}, full {stats['full']} | "
              f"batch gap s: median {q(.5):.2f} p99 {q(.99):.2f} max {g[-1] if g else float('nan'):.2f} | {rate/1024:.2f} KB/s per viewer | probe {probe_words[0]/(time.time()-t0):.1f} words/s", flush=True)
    stop.set(); await asyncio.gather(*tasks, return_exceptions=True)
asyncio.run(main())
