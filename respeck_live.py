"""
Live activity + social-signal recognition straight from a RESpeck over Bluetooth, in the
terminal (no phone app). Uses the same TFLite models and the same window/step/smoothing logic
as predict.py, so what you see here is what the app will show.

Setup (once):   pip install bleak
Find sensors:   python respeck_live.py --scan
Run:            python respeck_live.py --id F0:5B:12:34:56:78          # ID from the QR code
                python respeck_live.py                                  # nearest RESpeck
Also record:    python respeck_live.py --id ... --record session1.csv
Stop:           Ctrl+C (prints a summary)

Protocol (from the PDIoT app, lib/ble.dart and lib/utils.dart):
  service 00001523-1212-efde-1523-785feabcd125, notify characteristic 00001524-1212-efde-1523-785feabcd125
  packet = uint32 timestamp | uint16 sequence | uint8 battery | uint8 charging | N x (int16 x, y, z)
  all big-endian, acceleration in g = raw / 16384.

Windows/Linux: --id is the Bluetooth address printed on the QR code.
macOS hides addresses; the sensor is matched by the ID in its advertised service data instead
(same as the iPhone app), so --id still works there, or leave it out to take the nearest RESpeck.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # hide TensorFlow start-up noise
import argparse
import asyncio
import csv
import struct
import sys
import time
from collections import Counter, deque
from pathlib import Path

from predict import Classifier, STATIC, smoothed

SERVICE = "00001523-1212-efde-1523-785feabcd125"
ACCEL_CHAR = "00001524-1212-efde-1523-785feabcd125"
FEED_UUID = "0000feed-0000-1000-8000-00805f9b34fb"
RESPECK_NAMES = ("Res6AL", "Res6AM", "ResV5i")
DATA_OFFSET, SAMPLE_BYTES = 8, 6


def decode_packet(data: bytes):
    """-> (timestamp_ms, sequence, battery %, charging, [(x, y, z), ...] in g)."""
    if len(data) < DATA_OFFSET:
        return None
    ts, seq, batt, chg = struct.unpack_from(">IHBB", data, 0)
    n = (len(data) - DATA_OFFSET) // SAMPLE_BYTES
    raw = struct.unpack_from(f">{3 * n}h", data, DATA_OFFSET)
    samples = [(raw[3 * i] / 16384.0, raw[3 * i + 1] / 16384.0, raw[3 * i + 2] / 16384.0) for i in range(n)]
    return int(ts * 197 * 1000 / 32768), seq, batt, chg == 1, samples


def _service_id(adv):
    b = (adv.service_data or {}).get(FEED_UUID)
    return ":".join(f"{x:02X}" for x in b) if b else None


async def find_respeck(want_id: str | None, timeout: float = 15.0, list_only=False):
    from bleak import BleakScanner
    want = want_id.upper() if want_id else None
    print(f"scanning for {'RESpeck ' + want if want else 'RESpecks'} ({timeout:.0f} s)...")
    found = await BleakScanner.discover(timeout=timeout, return_adv=True)
    hits = []
    for dev, adv in found.values():
        name = adv.local_name or dev.name or ""
        sid = _service_id(adv)
        is_respeck = name.startswith("Res") or sid is not None
        if list_only and is_respeck:
            print(f"  {name:8s} address={dev.address}  id={sid or '-'}  rssi={adv.rssi}")
        if not is_respeck:
            continue
        if want and want not in (dev.address.upper(), sid):
            continue
        if name and name not in RESPECK_NAMES:
            print(f"  {name} at {dev.address}: firmware not supported by the app protocol")
            continue
        hits.append((adv.rssi, dev, name))
    if list_only:
        return None
    if not hits:
        sys.exit("no matching RESpeck found – is it charged, nearby and not connected to a phone?")
    hits.sort(key=lambda h: h[0], reverse=True)       # nearest first
    rssi, dev, name = hits[0]
    print(f"found {name} at {dev.address} (rssi {rssi})")
    return dev


class LiveSession:
    def __init__(self, models_dir: Path, smooth: int, record: str | None, quiet: bool,
                 har_name: str = "har_hybrid", social_name: str = "social_hybrid"):
        self.har = Classifier(models_dir, har_name)
        self.soc = Classifier(models_dir, social_name)
        if not self.har.ok:
            sys.exit(f"no {har_name} model in {models_dir} – run: python run_cw3.py --task activity --mode final")
        if not self.soc.ok:
            print(f"!! no {social_name}_meta.json in {models_dir} – social signals will not be shown. "
                  f"Export it first: python run_cw3.py --task social --mode final --model "
                  f"{social_name.removeprefix('social_')} --folds folds_social_signal.csv")
        self.act_hist, self.soc_hist = deque(maxlen=smooth), deque(maxlen=smooth)
        self.act, self.social = "…", "…"
        self.n_samples = self.n_packets = self.lost = 0
        self.last_seq = None
        self.battery = None
        self.t0 = time.time()
        self.log = Counter()                       # seconds spent per shown activity
        self.quiet = quiet
        self.rec = None
        if record:
            self.rec_f = open(record, "w", newline="")
            self.rec = csv.writer(self.rec_f)
            self.rec.writerow(["receivedTs", "respeckTs", "seq", "accelX", "accelY", "accelZ"])
            print(f"recording raw samples to {record}")

    def on_packet(self, _char, data: bytearray):
        t_recv = time.time()
        pkt = decode_packet(bytes(data))
        if pkt is None:
            return
        ts, seq, batt, charging, samples = pkt
        if self.last_seq is not None and seq != (self.last_seq + 1) % 65536:
            self.lost += (seq - self.last_seq - 1) % 65536
        self.last_seq, self.battery = seq, (batt, charging)
        self.n_packets += 1
        for x, y, z in samples:
            self.n_samples += 1
            if self.rec:
                self.rec.writerow([int(t_recv * 1000), ts, seq, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
            r_act = self.har.add((x, y, z))
            r_soc = self.soc.add((x, y, z)) if self.soc.ok else None
            if r_soc and self.act in STATIC:
                self.soc_hist.append(r_soc[0])
                self.social = f"{smoothed(self.soc_hist)} ({r_soc[1]:.0%})"
            if r_act:
                self.act_hist.append(r_act[0])
                new_act = smoothed(self.act_hist)
                self.log[new_act] += self.har.S / 12.5
                self.act = new_act
                if not self.soc.ok:
                    self.social = "– (no social model)"
                elif new_act not in STATIC:
                    self.soc_hist.clear(); self.social = "– (moving)"
                elif not self.soc_hist:
                    self.social = "– (filling 10 s window)"
                self.show(r_act[1], (time.time() - t_recv) * 1000)

    def show(self, conf, latency_ms):
        t = time.time() - self.t0
        batt = f"{self.battery[0]}%{'+' if self.battery[1] else ''}" if self.battery else "?"
        line = (f"{time.strftime('%H:%M:%S')} {t:6.0f}s  activity: {self.act + f' ({conf:.0%})':26s} "
                f"social: {self.social:28s} batt {batt:5s} rate {self.n_samples / max(t, 1e-6):4.1f} Hz "
                f"lost {self.lost}  proc {latency_ms:4.1f} ms")
        print(line, flush=True)

    def summary(self):
        if self.rec:
            self.rec_f.close()
        t = time.time() - self.t0
        print(f"\nsession {t:.0f} s, {self.n_samples} samples ({self.n_samples / max(t, 1):.1f} Hz), "
              f"{self.n_packets} packets, {self.lost} lost")
        if self.log:
            print("time per activity (as shown):")
            for a, s in self.log.most_common():
                print(f"  {a:16s} {s:6.0f} s")


async def run(args):
    from bleak import BleakClient
    if args.scan:
        await find_respeck(None, args.timeout, list_only=True)
        return
    dev = await find_respeck(args.id, args.timeout)
    session = LiveSession(Path(args.models), max(1, args.smooth), args.record, args.quiet, args.har, args.social)
    stop = asyncio.Event()

    def disconnected(_client):
        print("\n!! RESpeck disconnected")
        stop.set()

    async with BleakClient(dev, disconnected_callback=disconnected) as client:
        print("connected – waiting for data (first activity after 4 s, first social signal after 10 s)")
        await asyncio.sleep(1.0)                     # the app waits before subscribing too
        await client.start_notify(ACCEL_CHAR, session.on_packet)
        try:
            if args.duration:
                await asyncio.wait_for(stop.wait(), timeout=args.duration)
            else:
                await stop.wait()
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        finally:
            try:
                await client.stop_notify(ACCEL_CHAR)
            except Exception:
                pass
            session.summary()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", help="RESpeck ID from its QR code (Bluetooth address, e.g. F0:5B:12:34:56:78)")
    ap.add_argument("--scan", action="store_true", help="list nearby RESpecks and exit")
    ap.add_argument("--models", default="cw3_out")
    ap.add_argument("--har", default="har_hybrid", help="activity model name (e.g. har_gbm)")
    ap.add_argument("--social", default="social_hybrid", help="social model name (e.g. social_gbm, social_minirocket)")
    ap.add_argument("--smooth", type=int, default=3, help="majority vote over the last N predictions")
    ap.add_argument("--record", help="also save raw samples to this CSV")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = until Ctrl+C)")
    ap.add_argument("--timeout", type=float, default=15, help="scan time in seconds")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    try:
        import bleak  # noqa: F401
    except ImportError:
        sys.exit("bleak is not installed – run: pip install bleak")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
