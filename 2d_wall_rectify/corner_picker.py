"""
Local tool to click each wall photo's 4 corners (top-left, top-right,
bottom-right, bottom-left) and save them, in original-image pixel
coordinates, to corners.json — the input wall_rectify.py builds each wall's
homography from.

Pure stdlib (http.server), nothing to install. Serves the photos straight
from disk at full resolution; a 6x magnifier next to the canvas makes the
clicks precise despite the downscaled main view. Clicked corners were what
took the marker sanity check from 30-52% error (eyeballed corners) to
2-6% — see README.md.

Usage:
    python 2d_wall_rectify/corner_picker.py
    then open http://localhost:8765
"""

from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from config import CORNERS_FILE, IMAGES_DIR, WALLS

PORT = 8765

WALLS_JSON = json.dumps([
    {"name": w.name, "file": w.file, "size_cm": f"{w.width_cm:.0f} x {w.height_cm:.0f} cm"}
    for w in WALLS
])

HTML_PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Wall corner picker</title>
<style>
  body { font-family: system-ui, sans-serif; background: #1e1e22; color: #eee; margin: 0; padding: 16px; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  #sub { color: #aaa; font-size: 13px; margin-bottom: 12px; }
  #layout { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
  #stage { position: relative; border: 1px solid #444; background: #000; cursor: crosshair; }
  #stage canvas { display: block; }
  #side { width: 260px; flex-shrink: 0; }
  #mag { border: 1px solid #444; background: #000; margin-bottom: 10px; }
  .btn { background: #3a3a42; color: #eee; border: 1px solid #555; border-radius: 4px;
         padding: 7px 12px; margin: 3px 4px 3px 0; cursor: pointer; font-size: 13px; }
  .btn:hover { background: #4a4a52; }
  .btn:disabled { opacity: 0.4; cursor: default; }
  .btn.primary { background: #2d6a4f; border-color: #3a8b66; }
  .btn.primary:hover { background: #35815e; }
  #coords { font-size: 13px; line-height: 1.6; margin-top: 8px; }
  #coords div { padding: 2px 6px; border-radius: 3px; }
  #coords div.done { background: #2a3a2a; }
  #progress { margin-top: 14px; font-size: 13px; }
  #progress span { display: inline-block; width: 22px; height: 22px; line-height: 22px; text-align: center;
                   border-radius: 50%; background: #333; margin-right: 4px; border: 1px solid #555; }
  #progress span.complete { background: #2d6a4f; border-color: #3a8b66; }
  #progress span.current { border-color: #eee; }
  #status { margin-top: 10px; font-size: 13px; color: #9dd; min-height: 18px; }
  #instructions { font-size: 12px; color: #bbb; max-width: 260px; }
</style>
</head>
<body>
<h1>Wall corner picker</h1>
<div id="sub">Click the 4 corners of the <b>flat wall face</b> in order — where it meets ceiling, then the two side corners, then floor. Used to build each wall's homography in wall_rectify.py.</div>

<div id="layout">
  <div id="stage"><canvas id="mainCanvas"></canvas></div>
  <div id="side">
    <canvas id="mag" width="220" height="220"></canvas>
    <div id="instructions">
      Move the mouse over the photo to zoom in on the corner before clicking.
      Click order: <b>1 top-left &rarr; 2 top-right &rarr; 3 bottom-right &rarr; 4 bottom-left</b>.
    </div>
    <div id="progress"></div>
    <div style="margin-top:10px;">
      <button class="btn" id="prevBtn">&larr; Prev wall</button>
      <button class="btn" id="nextBtn">Next wall &rarr;</button>
      <button class="btn" id="resetBtn">Reset this wall</button>
    </div>
    <div id="coords"></div>
    <button class="btn primary" id="saveBtn" disabled>Save all walls</button>
    <div id="status"></div>
  </div>
</div>

<script>
const LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"];
let walls = [];
let current = 0;
let points = {};   // name -> [[x,y],...] in ORIGINAL image pixel coords
let img = new Image();
let scale = 1;

const mainCanvas = document.getElementById("mainCanvas");
const ctx = mainCanvas.getContext("2d");
const mag = document.getElementById("mag");
const magCtx = mag.getContext("2d");
const stage = document.getElementById("stage");

async function init() {
  const res = await fetch("/walls.json");
  walls = await res.json();
  walls.forEach(w => points[w.name] = []);
  loadWall(0);
}

function loadWall(idx) {
  current = idx;
  const w = walls[idx];
  img = new Image();
  img.onload = () => {
    const maxW = Math.min(1100, window.innerWidth - 320);
    scale = Math.min(1, maxW / img.naturalWidth);
    mainCanvas.width = img.naturalWidth * scale;
    mainCanvas.height = img.naturalHeight * scale;
    redraw();
  };
  img.src = "/image/" + idx;
  document.getElementById("sub").innerHTML =
    `Wall <b>${idx+1}/${walls.length}</b> (${w.name}, ${w.file}, ${w.size_cm}) — click the 4 corners of the flat wall face.`;
  updateProgress();
  updateCoordsList();
  updateNav();
}

function redraw() {
  ctx.drawImage(img, 0, 0, mainCanvas.width, mainCanvas.height);
  const pts = points[walls[current].name];
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#3ddc84";
  ctx.fillStyle = "#3ddc84";
  ctx.beginPath();
  pts.forEach((p, i) => {
    const x = p[0] * scale, y = p[1] * scale;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  if (pts.length === 4) ctx.closePath();
  ctx.stroke();
  pts.forEach((p, i) => {
    const x = p[0] * scale, y = p[1] * scale;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.font = "12px sans-serif";
    ctx.fillText(String(i + 1), x + 8, y - 8);
    ctx.fillStyle = "#3ddc84";
  });
}

function updateCoordsList() {
  const pts = points[walls[current].name];
  const div = document.getElementById("coords");
  div.innerHTML = LABELS.map((label, i) => {
    const p = pts[i];
    const txt = p ? `${label}: (${Math.round(p[0])}, ${Math.round(p[1])})` : `${label}: —`;
    return `<div class="${p ? 'done' : ''}">${txt}</div>`;
  }).join("");
}

function updateProgress() {
  const div = document.getElementById("progress");
  div.innerHTML = walls.map((w, i) => {
    const done = points[w.name].length === 4;
    const cls = (i === current ? "current " : "") + (done ? "complete" : "");
    return `<span class="${cls}" title="${w.name}">${i + 1}</span>`;
  }).join("");
  const allDone = walls.every(w => points[w.name].length === 4);
  document.getElementById("saveBtn").disabled = !allDone;
}

function updateNav() {
  document.getElementById("prevBtn").disabled = current === 0;
  document.getElementById("nextBtn").disabled = current === walls.length - 1;
}

mainCanvas.addEventListener("mousemove", (e) => {
  const rect = mainCanvas.getBoundingClientRect();
  const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
  const ox = cx / scale, oy = cy / scale;
  const zoom = 6;
  const half = mag.width / (2 * zoom);
  magCtx.imageSmoothingEnabled = false;
  magCtx.clearRect(0, 0, mag.width, mag.height);
  magCtx.drawImage(img, ox - half, oy - half, half * 2, half * 2, 0, 0, mag.width, mag.height);
  magCtx.strokeStyle = "#ff3b3b";
  magCtx.lineWidth = 1;
  magCtx.beginPath();
  magCtx.moveTo(mag.width / 2, 0); magCtx.lineTo(mag.width / 2, mag.height);
  magCtx.moveTo(0, mag.height / 2); magCtx.lineTo(mag.width, mag.height / 2);
  magCtx.stroke();
});

mainCanvas.addEventListener("click", (e) => {
  const name = walls[current].name;
  if (points[name].length >= 4) return;
  const rect = mainCanvas.getBoundingClientRect();
  const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
  points[name].push([cx / scale, cy / scale]);
  redraw();
  updateCoordsList();
  updateProgress();
});

document.getElementById("resetBtn").onclick = () => {
  points[walls[current].name] = [];
  redraw();
  updateCoordsList();
  updateProgress();
};
document.getElementById("prevBtn").onclick = () => { if (current > 0) loadWall(current - 1); };
document.getElementById("nextBtn").onclick = () => { if (current < walls.length - 1) loadWall(current + 1); };

document.getElementById("saveBtn").onclick = async () => {
  const status = document.getElementById("status");
  status.textContent = "Saving...";
  const res = await fetch("/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(points),
  });
  if (res.ok) {
    status.textContent = "Saved to corners.json.";
  } else {
    status.textContent = "Save failed — check the server terminal.";
  }
};

init();
</script>
</body>
</html>
"""


def format_corners(corners: dict) -> str:
    """One line per wall, rounded to 0.1px — sub-pixel digits beyond that are click noise."""
    lines = [f'  "{name}": {json.dumps([[round(x, 1), round(y, 1)] for x, y in pts])}'
             for name, pts in corners.items()]
    return "{\n" + ",\n".join(lines) + "\n}\n"


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = unquote(self.path.split("?")[0])
        if path == "/":
            self._send(200, HTML_PAGE.encode("utf-8"))
        elif path == "/walls.json":
            self._send(200, WALLS_JSON.encode("utf-8"), "application/json")
        elif path.startswith("/image/"):
            try:
                fpath = IMAGES_DIR / WALLS[int(path.rsplit("/", 1)[-1])].file
                data = fpath.read_bytes()
            except (ValueError, IndexError, FileNotFoundError):
                self._send(404, b"not found")
                return
            self._send(200, data, mimetypes.guess_type(str(fpath))[0] or "image/jpeg")
        else:
            self._send(404, b"not found")

    def do_POST(self) -> None:
        if self.path != "/save":
            self._send(404, b"not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        CORNERS_FILE.write_text(format_corners(json.loads(self.rfile.read(length))))
        print(f"Saved corners to {CORNERS_FILE}")
        self._send(200, b'{"ok": true}', "application/json")

    def log_message(self, format: str, *args) -> None:   # quiet the default access log
        pass


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Corner picker running at http://localhost:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
