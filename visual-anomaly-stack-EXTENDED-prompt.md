# Self-Hosted Visual Anomaly Testing Stack — **Extended Prompt** (Qdrant + YOLO + Transformers + OCR + Triage)

> **Goal:** Build a *Percy/Applitools–like* visual regression system using **Playwright** (record → replay),
> a **Node.js API** (storage, indexing, triage), and a **Python FastAPI ML service** (SSIM + pixel diff + LPIPS + YOLO + ViT distance + OCR)
> to score screenshots, semantically explain diffs, retrieve similar cases from **Qdrant**, and output triage suggestions.
> No SaaS required; everything runs locally or in CI via Docker Compose.

---

## ✅ Deliverables

1. **Dockerized services**
   - `api` (Node.js/Express): stores baselines/candidates, calls ML, **upserts to Qdrant**, exposes approve/triage endpoints, serves artifacts
   - `ml` (Python/FastAPI): computes SSIM, pixel-diff, LPIPS, **YOLO detections**, **ViT cosine distance**, **OCR text**, **heatmap**
   - `qdrant` (VectorDB): stores **CLIP embeddings** (and optional blob/crop embeddings) + metadata for retrieval
2. **Playwright test suite** (`ui-tests/`) that records flows and uploads snapshots to the API
3. **CI pipeline example** (GitHub Actions) that fails on high anomaly scores and attaches heatmaps
4. **Docs** (this file) describing setup, run, and extension roadmap (SAM, active learning, labeling, LLM triage)

---

## 🏗️ Project Structure (create exactly this tree)

```
.
├─ docker-compose.yml
├─ services/
│  ├─ api/
│  │  ├─ Dockerfile
│  │  ├─ package.json
│  │  └─ index.js
│  └─ ml/
│     ├─ Dockerfile
│     ├─ requirements.txt
│     └─ main.py
├─ ui-tests/
│  ├─ package.json
│  ├─ playwright.config.ts
│  └─ tests/
│     └─ visual.spec.ts
└─ data/                # (created at runtime) baselines & runs
```
> **Note:** All directories/files below must be generated **verbatim** by the agent.

---

## 🧩 docker-compose.yml (extended)

```yaml
version: "3.9"
services:
  api:
    build: ./services/api
    ports: ["8080:8080"]
    environment:
      - ML_URL=http://ml:8000
      - QDRANT_URL=http://qdrant:6333
      - DATA_DIR=/app/data
    volumes:
      - ./data:/app/data
    depends_on: [ml, qdrant]

  ml:
    build: ./services/ml
    ports: ["8000:8000"]

  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes:
      - ./qdrant_storage:/qdrant/storage
```
---

## 🐍 Python ML Scorer/Analyzer (FastAPI)

### `services/ml/Dockerfile`
```dockerfile
FROM python:3.10-slim

# System deps (OpenCV headless, fonts/glib for cv2, easyocr torch deps handled via pip)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `services/ml/requirements.txt`
```
fastapi==0.114.2
uvicorn[standard]==0.30.6
numpy==2.1.1
Pillow==10.4.0
scikit-image==0.24.0
opencv-python-headless==4.10.0.84

# Deep learning / metrics
torch==2.3.1
torchvision==0.18.1
lpips==0.1.4
timm==1.0.8

# Image embeddings & OCR
sentence-transformers==3.0.1
easyocr==1.7.1

# Object detection
ultralytics==8.2.86
```

### `services/ml/main.py`
```python
import base64, io, math
import numpy as np
import cv2
import torch
import lpips as lp
import timm
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from skimage.metrics import structural_similarity as ssim
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO
import easyocr

app = FastAPI(title="Visual Anomaly Scorer & Analyzer")

# --- Models ---
_lpips = lp.LPIPS(net='vgg')  # LPIPS perceptual distance
_vit = timm.create_model('vit_base_patch16_224', pretrained=True).eval()
_clip = SentenceTransformer("clip-ViT-B-32")  # 512-d embeddings
_yolo = YOLO("yolov8n.pt")  # swap with a fine-tuned model for UI classes
_ocr = easyocr.Reader(['en'], gpu=False)

# --- Schemas ---
class Rect(BaseModel):
    x: int; y: int; width: int; height: int

class ScoreRequest(BaseModel):
    baseline_png_b64: str
    candidate_png_b64: str
    masks: list[Rect] = []
    ssim_weight: float = 0.4
    lpips_weight: float = 0.3
    vit_weight: float = 0.2
    pix_weight: float = 0.1
    anomaly_threshold: float = 0.25  # tune per app

def _b64_to_rgba(b64):
    arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Cannot decode image")
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return img

def _apply_masks(img_rgba, rects):
    img = img_rgba.copy()
    for r in rects:
        x, y, w, h = r.x, r.y, r.width, r.height
        patch = img[max(0,y):y+h, max(0,x):x+w, :3]
        if patch.size == 0: continue
        med = np.median(patch.reshape(-1,3), axis=0).astype(np.uint8)
        img[y:y+h, x:x+w, :3] = med
    return img

def _to_rgb_tensor(img_rgba):
    rgb = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
    t = torch.from_numpy(rgb).permute(2,0,1).float() / 255.0  # 3xHxW [0,1]
    return t

def _lpips_score(a_rgba, b_rgba):
    with torch.no_grad():
        a = (_to_rgb_tensor(a_rgba)*2-1)[None]
        b = (_to_rgb_tensor(b_rgba)*2-1)[None]
        return float(_lpips(a, b).cpu().numpy())

def _vit_distance(a_rgba, b_rgba):
    with torch.no_grad():
        def feats(t):
            x = F.interpolate(t[None], size=(224,224), mode='bilinear', align_corners=False)
            f = _vit.forward_features(x)  # [1, tokens, dim]
            f = F.normalize(f.flatten(1), dim=1)  # [1, D]
            return f
        a = _to_rgb_tensor(a_rgba); b = _to_rgb_tensor(b_rgba)
        fa, fb = feats(a), feats(b)
        cos = float((fa @ fb.T).item())
        return float(1.0 - cos)  # cosine distance (0=same)

def _diff_heatmap(a_rgba, b_rgba):
    a = cv2.cvtColor(a_rgba, cv2.COLOR_RGBA2GRAY)
    b = cv2.cvtColor(b_rgba, cv2.COLOR_RGBA2GRAY)
    d = cv2.absdiff(a, b)
    d = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
    heat = cv2.applyColorMap(d, cv2.COLORMAP_JET)
    base_rgb = cv2.cvtColor(a_rgba, cv2.COLOR_RGBA2RGB)
    overlay = cv2.addWeighted(base_rgb, 0.6, heat, 0.4, 0)
    ok, buf = cv2.imencode(".png", overlay)
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

def _yolo_detect(img_rgba):
    rgb = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
    res = _yolo.predict(source=rgb, verbose=False)[0]
    dets = []
    for box, conf, c in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
        dets.append({
            "box": [float(x) for x in box],
            "conf": float(conf),
            "class_id": int(c),
            "class_name": res.names[int(c)]
        })
    return dets

def _ocr_regions(img_rgba, heat_threshold=25):
    # OCR only where heatmap is strong to save time
    gray_a = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)
    # crude: use edges as proxy for text-rich areas; could be guided by heatmap
    edges = cv2.Canny(gray_a, 50, 150)
    ys, xs = np.where(edges>0)
    if len(xs)==0: return []
    # tile sampling around edges
    H, W = gray_a.shape
    tiles = []
    step = max(64, min(H,W)//10)
    for y in range(0, H, step):
        for x in range(0, W, step):
            tiles.append((x,y, min(step,W-x), min(step,H-y)))
    results = []
    for (x,y,w,h) in tiles[:30]:  # cap work
        crop = cv2.cvtColor(img_rgba[y:y+h, x:x+w], cv2.COLOR_RGBA2RGB)
        if crop.size==0: continue
        try:
            ocr = _ocr.readtext(crop)
            text = " ".join([t[1] for t in ocr]).strip()
            if text:
                results.append({"box":[x,y,x+w,y+h], "text":text[:200]})
        except Exception:
            pass
    return results

def _clip_embed(img_rgba):
    # returns 512-d normalized vector
    img = Image.fromarray(cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB))
    vec = _clip.encode([img], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vec.astype(np.float32).tolist()

@app.post("/score")
def score(req: ScoreRequest):
    base = _b64_to_rgba(req.baseline_png_b64)
    cand = _b64_to_rgba(req.candidate_png_b64)
    if cand.shape[:2] != base.shape[:2]:
        cand = cv2.resize(cand, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)

    base_m = _apply_masks(base, req.masks)
    cand_m = _apply_masks(cand, req.masks)

    base_g = cv2.cvtColor(base_m, cv2.COLOR_RGBA2GRAY)
    cand_g = cv2.cvtColor(cand_m, cv2.COLOR_RGBA2GRAY)
    ssim_val = ssim(base_g, cand_g, data_range=255)

    delta = cv2.absdiff(base_g, cand_g)
    pix_ratio = float((delta > 8).sum()) / float(delta.size)

    lpips_val = _lpips_score(base_m, cand_m)
    vit_dist = _vit_distance(base_m, cand_m)

    anomaly = (req.ssim_weight * (1.0 - ssim_val)) + \
              (req.lpips_weight * lpips_val) + \
              (req.vit_weight * vit_dist) + \
              (req.pix_weight  * min(1.0, pix_ratio))

    heat = _diff_heatmap(base, cand)
    # Light analysis add-ons:
    dets = _yolo_detect(cand)
    ocr = _ocr_regions(cand)
    clip_vec = _clip_embed(cand)

    return {
        "metrics": {
            "ssim": float(ssim_val),
            "lpips": float(lpips_val),
            "vit_distance": float(vit_dist),
            "pixel_diff_ratio": float(pix_ratio),
            "anomaly_score": float(anomaly),
            "is_anomaly": anomaly >= req.anomaly_threshold
        },
        "analysis": {
            "yolo": dets,
            "ocr": ocr
        },
        "embeddings": {
            "clip_512": clip_vec
        },
        "artifacts": {
            "heatmap_png_b64": heat
        }
    }
```
> *Notes:*  
> - You can later swap YOLO model to your **fine-tuned UI classes** (export `.pt` and mount it).  
> - OCR routine is intentionally light-weight; for production, guide OCR using heatmap blobs to reduce calls.  
> - Add a SAM/FastSAM step to segment exact diff blobs and recompute metrics locally if desired.

---

## 🟩 Node API (Express) with **Qdrant indexing** + **triage stub**

### `services/api/Dockerfile`
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY index.js .
ENV PORT=8080
CMD ["node", "index.js"]
```

### `services/api/package.json`
```json
{
  "name": "visual-api",
  "version": "0.2.0",
  "type": "module",
  "dependencies": {
    "axios": "1.7.4",
    "express": "4.19.2",
    "mkdirp": "3.0.1",
    "multer": "1.4.5-lts.1"
  }
}
```

### `services/api/index.js`
```js
import express from "express";
import multer from "multer";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import axios from "axios";
import mkdirp from "mkdirp";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json({ limit: "50mb" }));
const upload = multer({ storage: multer.memoryStorage() });

const DATA_DIR = process.env.DATA_DIR || path.join(__dirname, "data");
const ML_URL = process.env.ML_URL || "http://localhost:8000";
const QDRANT_URL = process.env.QDRANT_URL || "http://localhost:6333";
const COLLECTION = "ui_diffs";

function b64(buf) { return buf.toString("base64"); }
function baselinePath(testId, name) { return path.join(DATA_DIR, "baselines", testId, `${name}.png`); }
function candidatePath(runId, name) { return path.join(DATA_DIR, "runs", runId, `${name}.png`); }

async function ensureCollection() {
  try {
    await axios.put(`${QDRANT_URL}/collections/${COLLECTION}`, {
      vectors: { size: 512, distance: "Cosine" }
    });
  } catch (e) {
    // ignore if exists
  }
}

app.post("/runs", (_req, res) => {
  const runId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  res.json({ runId });
});

app.post("/snapshots", upload.single("image"), async (req, res) => {
  try {
    const { runId, testId, name, masks = "[]" } = req.body;
    if (!runId || !testId || !name) return res.status(400).json({ error: "runId,testId,name required" });
    const img = req.file?.buffer;
    if (!img) return res.status(400).json({ error: "image required" });

    const candPath = candidatePath(runId, name);
    await mkdirp(path.dirname(candPath));
    fs.writeFileSync(candPath, img);

    const basePath = baselinePath(testId, name);
    if (!fs.existsSync(basePath)) {
      await mkdirp(path.dirname(basePath));
      fs.writeFileSync(basePath, img); // first run becomes baseline
      return res.json({ status: "baseline_created", is_anomaly: false });
    }

    // Call ML for score + analysis + embeddings
    const payload = {
      baseline_png_b64: b64(fs.readFileSync(basePath)),
      candidate_png_b64: b64(img),
      masks: JSON.parse(masks)
    };
    const { data } = await axios.post(`${ML_URL}/score`, payload, { timeout: 240000 });

    // Save heatmap
    const heatOut = candPath.replace(/\.png$/, ".heat.png");
    const heatB64 = data.artifacts?.heatmap_png_b64;
    if (heatB64) fs.writeFileSync(heatOut, Buffer.from(heatB64, "base64"));

    // Upsert to Qdrant
    await ensureCollection();
    const point = {
      points: [{
        id: `${runId}::${testId}::${name}`,
        vector: data.embeddings.clip_512,
        payload: {
          run_id: runId, test_id: testId, name,
          files: { baseline: basePath, candidate: candPath, heatmap: heatOut },
          metrics: data.metrics,
          yolo: data.analysis?.yolo || [],
          ocr: data.analysis?.ocr || []
        }
      }]
    };
    await axios.put(`${QDRANT_URL}/collections/${COLLECTION}/points`, point);

    res.json({ status: "scored", ...data, files: { baseline: basePath, candidate: candPath, heatmap: heatOut } });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "scoring_failed", detail: String(e) });
  }
});

// Approve: replace baseline with chosen candidate
app.post("/approve", (req, res) => {
  const { testId, name, runId } = req.body;
  if (!testId || !name || !runId) return res.status(400).json({ error: "testId,name,runId required" });
  const src = candidatePath(runId, name);
  const dst = baselinePath(testId, name);
  if (!fs.existsSync(src)) return res.status(404).json({ error: "candidate_not_found" });
  mkdirp.sync(path.dirname(dst));
  fs.copyFileSync(src, dst);
  res.json({ status: "baseline_updated", baseline: dst });
});

// Retrieve similar historical diffs from Qdrant
app.post("/similar", async (req, res) => {
  try {
    const { vector, limit = 5 } = req.body;
    if (!vector) return res.status(400).json({ error: "vector required" });
    const { data } = await axios.post(`${QDRANT_URL}/collections/${COLLECTION}/points/search`, {
      vector, limit, with_payload: true
    });
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: "similar_failed", detail: String(e) });
  }
});

// Rule-based triage stub (swap to LLM later)
app.post("/triage", (req, res) => {
  const m = req.body?.metrics || {};
  const yolo = req.body?.yolo || [];
  const ocr = req.body?.ocr || [];

  let severity = 0;
  const score = Number(m.anomaly_score ?? 0);
  if (score >= 0.35) severity = 3;
  else if (score >= 0.28) severity = 2;
  else if (score >= 0.22) severity = 1;

  // Downgrade if only text-like changes detected and small pixel ratio
  const texty = ocr.length > 0;
  const banners = yolo.filter(d => /banner|ad/i.test(d.class_name)).length;
  if (texty && (m.pixel_diff_ratio ?? 0) < 0.05) severity = Math.max(0, severity - 1);
  if (banners > 0) severity = Math.max(0, severity - 1);

  const recs = [];
  if (texty) recs.push("Consider masking dynamic text (timestamps/prices).");
  if (banners > 0) recs.push("Mask `.ad-banner` region.");
  if ((m.ssim ?? 1) < 0.9) recs.push("Layout changed; confirm CSS/spacing updates.");

  res.json({
    severity, pass_ci: severity <= 1,
    summary: `Score=${score.toFixed(3)}; pixel=${(m.pixel_diff_ratio ?? 0).toFixed(3)}; text=${texty}; banners=${banners}`,
    recommendations: recs
  });
});

// Artifact server
app.get("/artifacts/*", (req, res) => {
  const p = path.join(DATA_DIR, req.params[0] || "");
  if (!fs.existsSync(p)) return res.status(404).send("Not found");
  res.sendFile(p);
});

const port = process.env.PORT || 8080;
app.listen(port, () => console.log(`API listening on ${port}`));
```

---

## 🎭 Playwright Tests (unchanged usage; now with retrieval option)

### `ui-tests/package.json`
```json
{
  "name": "ui-tests",
  "private": true,
  "devDependencies": {
    "@playwright/test": "1.46.0",
    "axios": "1.7.4"
  },
  "scripts": {
    "test": "playwright test"
  }
}
```

### `ui-tests/playwright.config.ts`
```ts
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 60000,
  use: {
    headless: true,
    viewport: { width: 1280, height: 800 }
  }
});
```

### `ui-tests/tests/visual.spec.ts`
```ts
import { test, expect, Page } from '@playwright/test';
import axios from 'axios';

const API = process.env.VISUAL_API || 'http://localhost:8080';
const APP_URL = process.env.APP_URL || 'http://localhost:3000';

async function startRun() {
  const { data } = await axios.post(`${API}/runs`);
  return data.runId as string;
}

type Mask = { x: number; y: number; width: number; height: number };

async function visualSnapshot(page: Page, runId: string, testId: string, name: string, maskSelectors: string[] = []) {
  // compute masks
  const masks: Mask[] = [];
  for (const sel of maskSelectors) {
    const loc = page.locator(sel).first();
    const box = await loc.boundingBox();
    if (box) masks.push({ x: Math.round(box.x), y: Math.round(box.y), width: Math.round(box.width), height: Math.round(box.height) });
  }

  await page.addStyleTag({ content: `*{animation:none !important; transition:none !important}` });
  await page.evaluate(() => { Date.now = () => 1735600000000; });

  const shot = await page.screenshot({ fullPage: true });
  const form = new FormData();
  form.set('runId', runId);
  form.set('testId', testId);
  form.set('name', name);
  form.set('masks', JSON.stringify(masks));
  form.set('image', new Blob([shot], { type: 'image/png' }), `${name}.png`);

  const resp = await fetch(`${API}/snapshots`, { method: 'POST', body: form as any });
  const json = await resp.json();

  if (json.embeddings?.clip_512) {
    const { data: knn } = await axios.post(`${API}/similar`, { vector: json.embeddings.clip_512, limit: 5 });
    console.log("Similar past diffs:", knn);
  }

  const triage = await axios.post(`${API}/triage`, {
    metrics: json.metrics, yolo: json.analysis?.yolo, ocr: json.analysis?.ocr
  });
  console.log("Triage:", triage.data);

  return json;
}

test('Homepage visual stays OK', async ({ page }) => {
  const runId = await startRun();
  await page.goto(APP_URL);

  const result = await visualSnapshot(page, runId, 'webapp@home', 'home', [
    '[data-testid=clock]',
    '.ad-banner'
  ]);

  console.log('Visual result:', result);
  if (result.status === 'scored') {
    expect(result.metrics.anomaly_score).toBeLessThan(0.25);
  }
});
```

---

## ▶️ How to Run Locally

```bash
# 0) Ensure Docker & Node.js 20+ are installed

# 1) Boot services (API, ML, Qdrant)
docker compose up --build -d

# 2) Install Playwright deps
cd ui-tests
npm i
npx playwright install --with-deps

# 3) Run tests against your app (set APP_URL to your local/staging)
APP_URL=http://localhost:3000 VISUAL_API=http://localhost:8080 npx playwright test
```

Artifacts:
- Baselines: `./data/baselines/<testId>/<name>.png` (created on first run)
- Candidates & heatmaps: `./data/runs/<runId>/*.png`
- Heatmap: `GET http://localhost:8080/artifacts/runs/<runId>/<name>.heat.png`
- Similar cases: `POST /similar` with `embeddings.clip_512`
- Triage: `POST /triage` with metrics/detections/ocr returns severity + recommendations

---

## ✅ Approve Intentional Changes

```bash
curl -X POST http://localhost:8080/approve \
  -H 'Content-Type: application/json' \
  -d '{"testId":"webapp@home","name":"home","runId":"<RUN_ID_FROM_LOG>"}'
```

---

## 🔒 CI (GitHub Actions example)

`.github/workflows/visual.yml`
```yaml
name: ui-visual
on: [pull_request]

jobs:
  visual:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - name: Boot stack
        run: docker compose up --build -d
      - name: Install tests
        working-directory: ui-tests
        run: |
          npm i
          npx playwright install --with-deps
      - name: Run visual tests
        working-directory: ui-tests
        env:
          APP_URL: ${{ secrets.PREVIEW_URL }}
          VISUAL_API: http://localhost:8080
        run: npx playwright test
      - name: Upload heatmaps (artifact)
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: visual-heatmaps
          path: data/runs/**/**.heat.png
```

---

## 🧠 Data & Training Enhancements (optional but recommended)

- **Labeling UI**: Run **Label Studio** externally to review diffs and assign labels (actionable/flake/severity). Export CSV/JSON to `offline_train/`.
- **Features dataset**: Build `features.parquet` from historical runs (metrics, #diff blobs, YOLO histogram, OCR delta length, viewport, browser).
- **Classifier**: Train an XGBoost/LightGBM model to predict severity and feed it back into `/triage`. Track with **MLflow**.
- **Active Learning**: Use **Qdrant + modAL** to fetch the most uncertain/novel diffs for labeling.
- **Drift & validation**: Add **Great Expectations**/**Evidently** checks to ensure incoming data quality and monitor score drift.

---

## 🗺️ Roadmap

- **SAM/FastSAM** segmentation guided by heatmap for precise diff masks; recompute metrics inside mask.
- **Device/browser grids** using Playwright projects.
- **Per-page/per-cluster thresholds**: store thresholds in Qdrant payload by layout cluster.
- **LLM Triage**: Replace stub `/triage` with a call to your chosen LLM (e.g., DeepSeek) using prompt that includes: metrics, YOLO tags, OCR snippets, and links to artifacts + top-3 similar cases from Qdrant.
- **Reviewer UI**: Next.js dashboard listing runs with side-by-side baseline/candidate + heatmap + similar cases + Approve.
- **S3/MinIO storage** for artifacts; **Postgres** for rich metadata if desired.

---

## 📦 License

MIT for all generated files unless otherwise specified.

---

**End of extended prompt.** Generate the project exactly as specified above.
