"""Parse one openpilot rlog.zst offline with pycapnp; summarize message types and
extract the lateral-control-relevant signals to confirm feasibility."""
import sys, glob
import zstandard as zstd
import capnp
capnp.remove_import_hook()

CER = "/tmp/replay/cereal"
log_capnp = capnp.load(f"{CER}/log.capnp", imports=[CER, "/tmp/replay"])

def events(path):
    with open(path, "rb") as f:
        data = zstd.ZstdDecompressor().decompress(f.read(), max_output_size=400_000_000)
    return log_capnp.Event.read_multiple_bytes(data)

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/replay/logs/seg0.rlog.zst"
counts = {}
n_model = n_cs = 0
sample = {}
for ev in events(path):
    w = ev.which()
    counts[w] = counts.get(w, 0) + 1
    if w == "carState" and n_cs < 1:
        cs = ev.carState
        sample["carState"] = dict(vEgo=round(cs.vEgo,3), steeringAngleDeg=round(cs.steeringAngleDeg,3),
                                  steeringPressed=cs.steeringPressed, leftBlinker=cs.leftBlinker, rightBlinker=cs.rightBlinker)
        n_cs += 1
    if w == "modelV2" and n_model < 1:
        m = ev.modelV2
        try:
            dc = m.action.desiredCurvature
        except Exception:
            dc = None
        sample["modelV2"] = dict(pos_x_len=len(m.position.x), pos_y0=round(m.position.y[0],4) if len(m.position.y) else None,
                                 desiredCurvature=round(dc,6) if dc is not None else None,
                                 yStd_len=len(m.position.yStd), laneProbs=len(m.laneLineProbs))
        n_model += 1

print(f"parsed {sum(counts.values())} events, {len(counts)} types")
for k in sorted(counts, key=lambda x:-counts[x])[:18]:
    print(f"  {k:28s} {counts[k]}")
print("relevant present:", [k for k in ("carState","modelV2","liveParameters","liveDelay","carControl","liveCalibration","driverAssistance","selfdriveState","controlsState","liveTorqueParameters") if k in counts])
import json; print("samples:", json.dumps(sample, default=str))
