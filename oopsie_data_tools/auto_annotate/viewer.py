"""Generate a standalone HTML viewer for the model annotations.

One self-contained file, no server and no dependencies: it embeds the annotation records as
JSON and points ``<video>`` at the downloaded mp4s by relative path, so it works opened
straight from disk next to the data.

It shows what the repo's own annotation UI cannot: where each failure segment falls on the
video timeline. Segments are drawn as bands over a scrubber and seek the video when
clicked. The human annotation already present in every one of these episodes is shown
beside the model's, because the useful question here is not what the model said but where
it disagrees with the annotator.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py

from oopsie_data_tools.annotation_tool.annotation_schema import read_annotation_attrs
from oopsie_data_tools.auto_annotate import config, manifest


def human_annotation(h5_path: Path) -> Optional[Dict[str, Any]]:
    """The first human annotation in the file, normalised from v1 or v2."""
    with h5py.File(h5_path, "r") as handle:
        group = handle.get("episode_annotations")
        if not group:
            return None
        for name in group:
            if name == config.ANNOTATOR_NAME:
                continue
            annotation = read_annotation_attrs(group[name].attrs)
            annotation["annotator"] = name
            return annotation
    return None


def collect(root: Path) -> List[Dict[str, Any]]:
    """One record per annotated episode, pairing model output with the human label."""
    by_id = {episode["episode_id"]: episode for episode in manifest.load()}
    records: List[Dict[str, Any]] = []

    for sidecar in sorted(config.ANNOT_DIR.glob("*/*.json")):
        record = json.loads(sidecar.read_text())
        episode = by_id.get(record["episode_id"])
        if episode is None:
            continue

        h5_path = config.RAW_DIR / episode["release_path"]
        human = human_annotation(h5_path) or {}
        video = config.RAW_DIR / record["input"]["video"]

        records.append(
            {
                "episode_id": record["episode_id"],
                "source": record["source"],
                "video": video.relative_to(root).as_posix(),
                "duration": record["input"].get("duration_seconds") or 0.0,
                "camera": record["input"].get("camera", ""),
                "timesteps": record["input"].get("timestep_count", 0),
                "instruction": record["input"].get("provided_instruction") or "",
                "model": {
                    "task": record["episode"]["language_task"],
                    "outcome": record["episode"]["outcome"],
                    "rationale": record["episode"]["rationale"],
                },
                "segments": record["segments"],
                "warnings": record["warnings"],
                "human": {
                    "annotator": human.get("annotator", ""),
                    "outcome": human.get("outcome", ""),
                    "description": human.get("episode_description", "")
                    or human.get("failure_description", ""),
                    "categories": human.get("failure_category", []),
                    "severity": human.get("severity", ""),
                },
            }
        )
    return records


def build(out: Optional[Path] = None) -> Path:
    out = out or (config.DATA_ROOT / "viewer.html")
    records = collect(out.parent)
    payload = json.dumps(records, ensure_ascii=False)
    page = _TEMPLATE.replace("__DATA__", payload).replace(
        "__COUNT__", html.escape(str(len(records)))
    )
    out.write_text(page, encoding="utf-8")
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Oopsie Auto-Annotations</title>
<style>
  :root {
    --bg:#12141a; --panel:#1a1d26; --line:#2a2f3d; --text:#e6e8ee; --dim:#9aa3b5;
    --ok:#3ecf8e; --bad:#ff6b6b; --warn:#ffb454; --accent:#6aa8ff;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { padding:14px 18px; border-bottom:1px solid var(--line); display:flex;
           gap:18px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:16px; margin:0; font-weight:600; letter-spacing:.2px; }
  .stat { color:var(--dim); font-size:13px; }
  .stat b { color:var(--text); }
  select, input[type=search] { background:var(--panel); color:var(--text);
    border:1px solid var(--line); border-radius:6px; padding:5px 8px; font-size:13px; }
  .wrap { display:grid; grid-template-columns:330px 1fr; height:calc(100vh - 57px); }
  .list { overflow-y:auto; border-right:1px solid var(--line); }
  .item { padding:9px 12px; border-bottom:1px solid var(--line); cursor:pointer; }
  .item:hover { background:#1d2029; }
  .item.sel { background:#232838; border-left:3px solid var(--accent); padding-left:9px; }
  .item .eid { font-size:12px; color:var(--dim); word-break:break-all; }
  .item .row { display:flex; gap:6px; align-items:center; margin-bottom:3px; }
  .tag { font-size:10px; padding:1px 6px; border-radius:10px; border:1px solid var(--line);
         color:var(--dim); white-space:nowrap; }
  .tag.ok { color:var(--ok); border-color:#245c44; }
  .tag.bad { color:var(--bad); border-color:#5e2a2a; }
  .tag.dis { background:#4a2020; color:#ffc9c9; border-color:#7a3333; }
  .main { overflow-y:auto; padding:18px 22px; }
  .empty { color:var(--dim); padding:40px; text-align:center; }
  video { width:100%; max-height:52vh; background:#000; border-radius:8px; display:block; }
  .bar { position:relative; height:26px; margin:10px 0 4px; background:var(--panel);
         border:1px solid var(--line); border-radius:5px; overflow:hidden; cursor:pointer; }
  .seg { position:absolute; top:0; bottom:0; background:rgba(255,107,107,.45);
         border-left:2px solid var(--bad); border-right:1px solid rgba(255,107,107,.6); }
  .seg:hover { background:rgba(255,107,107,.7); }
  .play { position:absolute; top:0; bottom:0; width:2px; background:var(--accent); }
  .ticks { display:flex; justify-content:space-between; color:var(--dim); font-size:11px; }
  .cols { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:16px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }
  .card h3 { margin:0 0 8px; font-size:12px; text-transform:uppercase;
             letter-spacing:.7px; color:var(--dim); font-weight:600; }
  .kv { margin:6px 0; }
  .kv .k { color:var(--dim); font-size:12px; }
  .segcard { background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--bad);
             border-radius:8px; padding:11px 14px; margin-top:10px; cursor:pointer; }
  .segcard:hover { border-color:var(--accent); }
  .segcard .hd { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:6px; }
  .time { font-family:ui-monospace,Menlo,monospace; color:var(--accent); font-size:12px; }
  .warn { color:var(--warn); font-size:12px; margin-top:8px; }
  .none { color:var(--dim); font-style:italic; padding:8px 0; }
</style>
</head>
<body>
<header>
  <h1>Oopsie Auto-Annotations</h1>
  <span class="stat" id="summary"></span>
  <select id="fsource"><option value="">all sources</option></select>
  <select id="fagree">
    <option value="">model vs human: all</option>
    <option value="dis">disagreements only</option>
    <option value="agr">agreements only</option>
  </select>
  <select id="fseg">
    <option value="">any segments</option>
    <option value="has">has segments</option>
    <option value="none">no segments</option>
  </select>
  <input type="search" id="q" placeholder="search task / description">
</header>
<div class="wrap">
  <div class="list" id="list"></div>
  <div class="main" id="main"><div class="empty">Select an episode.</div></div>
</div>
<script>
const DATA = __DATA__;
const esc = s => String(s==null?"":s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const isFail = o => o === "failure";
// A four-way outcome vs the human's: compared on the success/failure split, since the
// human labels here mostly predate the qualified-success slugs.
const agrees = r => r.human.outcome ? (isFail(r.model.outcome) === isFail(r.human.outcome)) : null;
const fmt = t => { t = Math.max(0, t||0); const m = Math.floor(t/60), s = t - m*60;
                   return m + ":" + (s<10?"0":"") + s.toFixed(1); };

// Segment bounds in seconds. Video mode answers in seconds directly; frames mode answers
// in timesteps, so those are mapped back through the episode duration.
function secs(r, s, which) {
  const key = which + "_seconds";
  if (s[key] != null) return s[key];
  const step = s[which + "_timestep"];
  if (step == null || !r.timesteps) return 0;
  return (step / Math.max(r.timesteps - 1, 1)) * (r.duration || 0);
}

// Seeking before metadata is loaded is silently discarded by the browser, which makes an
// early click play from the start instead of the segment. Wait for it when necessary.
function seekTo(vid, t, play) {
  const go = () => {
    try { vid.currentTime = t; } catch (e) {}
    if (play) { const p = vid.play(); if (p && p.catch) p.catch(() => {}); }
  };
  if (vid.readyState >= 1) go();
  else vid.addEventListener("loadedmetadata", go, { once: true });
}

let sel = null;

function stats() {
  const scored = DATA.filter(r => agrees(r) !== null);
  const ok = scored.filter(agrees).length;
  const pct = scored.length ? (100*ok/scored.length).toFixed(0) : "-";
  document.getElementById("summary").innerHTML =
    "<b>" + DATA.length + "</b> episodes &nbsp;·&nbsp; outcome agreement with human: <b>" +
    ok + "/" + scored.length + "</b> (" + pct + "%)";
  const sources = [...new Set(DATA.map(r => r.source))].sort();
  const sel = document.getElementById("fsource");
  sources.forEach(s => { const o = document.createElement("option"); o.value = o.textContent = s; sel.appendChild(o); });
}

function visible() {
  const src = document.getElementById("fsource").value;
  const agr = document.getElementById("fagree").value;
  const seg = document.getElementById("fseg").value;
  const q = document.getElementById("q").value.toLowerCase().trim();
  return DATA.filter(r => {
    if (src && r.source !== src) return false;
    if (agr === "dis" && agrees(r) !== false) return false;
    if (agr === "agr" && agrees(r) !== true) return false;
    if (seg === "has" && !r.segments.length) return false;
    if (seg === "none" && r.segments.length) return false;
    if (q) {
      const hay = (r.model.task + " " + r.human.description + " " + r.episode_id + " " +
                   r.segments.map(s => s.what_happened).join(" ")).toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

function renderList() {
  const rows = visible();
  document.getElementById("list").innerHTML = rows.map(r => {
    const a = agrees(r);
    const dis = a === false ? '<span class="tag dis">differs</span>' : "";
    const oc = '<span class="tag ' + (isFail(r.model.outcome) ? "bad" : "ok") + '">' + esc(r.model.outcome) + "</span>";
    const sg = r.segments.length ? '<span class="tag">' + r.segments.length + " seg</span>" : "";
    return '<div class="item' + (sel === r.episode_id ? " sel" : "") + '" data-id="' + esc(r.episode_id) + '">' +
      '<div class="row">' + oc + sg + dis + '</div>' +
      '<div class="eid"><b>' + esc(r.source) + "</b> · " + esc(r.episode_id) + "</div></div>";
  }).join("") || '<div class="empty">No episodes match.</div>';
  document.querySelectorAll(".item").forEach(el =>
    el.onclick = () => { sel = el.dataset.id; renderList(); renderMain(); });
}

function renderMain() {
  const r = DATA.find(x => x.episode_id === sel);
  if (!r) return;
  const dur = r.duration || 0;
  const bands = r.segments.map((s, i) => {
    const a = secs(r, s, "start"), b = secs(r, s, "end");
    const L = dur ? 100*a/dur : 0, W = dur ? Math.max(0.8, 100*(b-a)/dur) : 0;
    return '<div class="seg" data-i="' + i + '" style="left:' + L + '%;width:' + W + '%" title="segment ' + (i+1) + '"></div>';
  }).join("");

  const segs = r.segments.length ? r.segments.map((s, i) => {
    const a = secs(r, s, "start"), b = secs(r, s, "end");
    return '<div class="segcard" data-i="' + i + '">' +
      '<div class="hd"><span class="time">▶ ' + fmt(a) + " – " + fmt(b) + '</span>' +
      '<span class="time">t' + s.start_timestep + "–" + s.end_timestep + "</span>" +
      s.failure_categories.map(c => '<span class="tag bad">' + esc(c) + "</span>").join("") +
      '<span class="tag">' + esc(s.severity) + '</span><span class="tag">' + esc(s.resetability) + "</span></div>" +
      "<div><b>What happened.</b> " + esc(s.what_happened) + "</div>" +
      '<div style="margin-top:5px"><b>Recovery.</b> ' + esc(s.how_to_recover) + "</div></div>";
  }).join("") : '<div class="none">No failure segments — the model judged this a clean success.</div>';

  const h = r.human;
  document.getElementById("main").innerHTML =
    '<video id="vid" controls preload="metadata" src="' + esc(r.video) + '"></video>' +
    '<div class="bar" id="bar">' + bands + '<div class="play" id="play" style="left:0"></div></div>' +
    '<div class="ticks"><span>0:00</span><span>' + esc(r.camera) + " · " + r.timesteps + " steps</span><span>" + fmt(dur) + "</span></div>" +
    '<div class="cols"><div class="card"><h3>Model</h3>' +
      '<div class="kv"><div class="k">task</div>' + esc(r.model.task) + "</div>" +
      '<div class="kv"><div class="k">outcome</div><span class="tag ' + (isFail(r.model.outcome)?"bad":"ok") + '">' + esc(r.model.outcome) + "</span></div>" +
      '<div class="kv"><div class="k">rationale</div>' + esc(r.model.rationale) + "</div></div>" +
    '<div class="card"><h3>Human' + (h.annotator ? " · " + esc(h.annotator) : "") + "</h3>" +
      '<div class="kv"><div class="k">outcome</div>' + (h.outcome ? '<span class="tag ' + (isFail(h.outcome)?"bad":"ok") + '">' + esc(h.outcome) + "</span>" : '<span class="none">none</span>') + "</div>" +
      '<div class="kv"><div class="k">description</div>' + (esc(h.description) || "—") + "</div>" +
      '<div class="kv"><div class="k">categories</div>' + (h.categories.length ? h.categories.map(c => '<span class="tag">' + esc(c) + "</span>").join(" ") : "—") + "</div>" +
      '<div class="kv"><div class="k">severity</div>' + (esc(h.severity) || "—") + "</div></div></div>" +
    (r.instruction ? '<div class="card" style="margin-top:14px"><h3>Recorded instruction</h3>' + esc(r.instruction) + "</div>" : "") +
    "<h3 style=\"margin:18px 0 0;font-size:12px;text-transform:uppercase;letter-spacing:.7px;color:var(--dim)\">Failure segments</h3>" +
    segs +
    (r.warnings.length ? '<div class="warn">⚠ ' + r.warnings.map(esc).join("<br>⚠ ") + "</div>" : "");

  const vid = document.getElementById("vid"), bar = document.getElementById("bar"), play = document.getElementById("play");
  vid.ontimeupdate = () => { const d = vid.duration || dur; if (d) play.style.left = (100*vid.currentTime/d) + "%"; };
  bar.onclick = e => { const d = vid.duration || dur;
    if (d) seekTo(vid, d * (e.clientX - bar.getBoundingClientRect().left) / bar.clientWidth, false); };
  const playSegment = i => {
    const s = r.segments[i];
    if (!s) return;
    seekTo(vid, secs(r, s, "start"), true);
    document.querySelectorAll(".segcard").forEach((el, j) =>
      el.style.borderLeftColor = j === i ? "var(--accent)" : "var(--bad)");
  };
  document.querySelectorAll(".seg").forEach(el =>
    el.onclick = e => { e.stopPropagation(); playSegment(+el.dataset.i); });
  document.querySelectorAll(".segcard").forEach(el =>
    el.onclick = () => playSegment(+el.dataset.i));
}

stats();
["fsource","fagree","fseg"].forEach(id => document.getElementById(id).onchange = renderList);
document.getElementById("q").oninput = renderList;
renderList();
</script>
</body>
</html>
"""


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build the standalone annotation viewer.")
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="output html path (default: <data root>/viewer.html)")
    args = parser.parse_args()
    path = build(args.out)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
