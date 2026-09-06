# PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware

**Smart India Hackathon 2026 · Problem Statement SIH26122 · Oil India Limited**

PlanBridge reconciles messy, free-text field Daily Progress Reports (DPRs)
against a Primavera P6 schedule, using a hybrid entity + semantic matching
pipeline and a three-way confidence gate (auto-accept / human review /
unmatched), then tracks dual (physical-claimed vs QA-verified) progress in
a SHA-256 hash-chained, CVC/CAG-audit-ready evidence ledger.

---

## 1. Project structure

```
planbridge/
├── requirements.txt
├── run_demo.py                One-click demo launcher (Phase 8)
├── test_benchmark.py          Automated safety benchmark (Phase 8)
│
├── synthetic_generator.py     Phase 1 — synthetic activities.json + dprs.json generator
│
├── schemas.py                 Shared Pydantic models for every phase
├── unit_normalizer.py         Phase 2 — deterministic unit conversion
├── ingestion.py                Phase 2 — DPR parsing + QA-clearance detection
│
├── entity_extractor.py         Phase 3 — regex + spaCy entity extraction
├── candidate_narrower.py       Phase 3 — hard entity filter (500+ activities -> 3-5 candidates)
│
├── vector_ranker.py            Phase 4 — Sentence-BERT semantic similarity (+ TF-IDF/SequenceMatcher fallback)
├── scoring_engine.py           Phase 4 — hybrid entity+semantic+quantity scoring formula
├── confidence_gate.py          Phase 4 — three-way AUTO_ACCEPT / HUMAN_REVIEW / UNMATCHED gate
│
├── progress_engine.py           Phase 5 — dual physical/QA-verified progress tracking
├── audit_logger.py              Phase 5 — SHA-256 hash-chained append-only evidence ledger
│
├── xer_exporter.py               Phase 6 — Primavera P6 .XER schedule delta exporter
├── main.py                       Phase 6 — FastAPI REST backend wrapping the full pipeline
│
├── frontend/
│   └── index.html                 Phase 7 — self-contained dashboard SPA (online + offline demo mode)
│
├── test_phase2.py ... test_phase6.py   Per-phase unit/integration test suites
│
└── data/
    ├── activities.json            Generated: 500+ synthetic Primavera P6 activities
    ├── dprs.json                  Generated: 100+ synthetic labeled DPR test cases
    ├── benchmark_results.json     Generated: full benchmark run output (Phase 8)
    └── evidence_ledger.json       Generated at runtime: the live audit ledger
```

---

## 2. Setup

### 2.1 Prerequisites

- Python 3.10+
- `pip`
- ~2 GB free disk space (mainly for `torch`)
- Internet access is only required once, to download the Sentence-BERT
  model on first run — see [Environment notes](#5-environment-notes--known-limitations)
  for what happens if you don't have it.

### 2.2 Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The `spacy download` step is required separately from `requirements.txt` —
`entity_extractor.py` uses the `en_core_web_sm` model for lemma-based
action/discipline detection. If it's missing, `EntityExtractor` logs a
warning and falls back to its regex-only keyword path automatically; it
won't crash, but detection is more robust with the model installed.

### 2.3 Generate the synthetic benchmark dataset

```bash
python synthetic_generator.py
```

This creates `data/activities.json` (500+ synthetic P6 activities) and
`data/dprs.json` (100+ labeled DPR test cases). Every other script in this
project depends on these two files existing — `run_demo.py` and
`test_benchmark.py` will both generate them automatically if you skip this
step, but running it explicitly first is faster for iterating.

---

## 3. Running the tests

### 3.1 Per-phase test suites

```bash
python -m unittest test_phase2.py -v
python -m unittest test_phase3.py -v
python -m unittest test_phase4.py -v
python -m unittest test_phase5.py -v
python -m unittest test_phase6.py -v
```

Or run all of them at once with pytest:

```bash
pytest test_phase2.py test_phase3.py test_phase4.py test_phase5.py test_phase6.py -v
```

`test_phase3.py` and later suites load the real spaCy/Sentence-BERT/torch
stack, so the *first* test in a fresh process takes longer (model
loading); subsequent tests in the same run are fast.

### 3.2 The safety benchmark (Phase 8)

This is the headline deliverable: it runs every labeled DPR in
`data/dprs.json` through the full pipeline and reports the **False
Auto-Accept Rate**, which must be 0.00%.

```bash
python test_benchmark.py                # full human-readable report
python test_benchmark.py --verbose       # also prints each false-accept/error case as found
pytest test_benchmark.py -s              # same benchmark, as a single pass/fail pytest case
```

Exit code is `0` if the safety gate passes (false auto-accept rate is
exactly 0.00%), `1` otherwise — wire this into CI as a hard gate on any
change to the matching pipeline. The full per-case results (not just the
summary) are saved to `data/benchmark_results.json` on every run.

---

## 4. Running the demo

### 4.1 One-click launch

```bash
python run_demo.py
```

This will:
1. Check that all required Python packages are importable (offers to
   `pip install -r requirements.txt` for you if anything's missing).
2. Generate `data/activities.json` / `data/dprs.json` automatically if
   they don't exist yet.
3. Start `uvicorn main:app --port 8000` in the background and wait until
   it reports healthy.
4. Open `frontend/index.html` in your default browser.
5. Print a quick demo script and instructions, then wait for `Ctrl+C` to
   shut everything down cleanly.

Useful flags:

```bash
python run_demo.py --port 8080        # use a different port
python run_demo.py --no-browser       # don't auto-open a browser (headless/CI)
python run_demo.py --skip-checks      # skip dependency/data checks (fast restart)
python run_demo.py --yes              # auto-confirm any prompts, non-interactive
```

### 4.2 Manual launch (equivalent, if you want to see backend logs directly)

```bash
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` directly in a browser (double-click it,
or `open frontend/index.html` / `xdg-open frontend/index.html`). The
frontend auto-detects whether the backend is reachable at
`http://localhost:8000`; if it isn't, it transparently switches to a
fully-functional **offline demo mode** with its own local simulation
engine — the presentation never breaks just because the backend didn't
start in time or isn't running at all.

- API docs (interactive OpenAPI/Swagger UI): **http://localhost:8000/docs**

---

## 5. Environment notes & known limitations

- **Sentence-BERT model download**: `VectorRanker` tries to load
  `all-MiniLM-L6-v2` (then `paraphrase-MiniLM-L6-v2`) from Hugging Face on
  first use. If there's no internet access, or the model isn't cached, it
  automatically falls back to TF-IDF cosine similarity (`scikit-learn`),
  and further to `difflib.SequenceMatcher` if even that's unavailable —
  the system never crashes for lack of the model, but semantic-similarity
  scores are measurably more conservative under the fallback than they
  would be with real embeddings. Check the startup log line
  `VectorRanker backend: ...` to see which one is active.
- **First-run latency**: loading spaCy + sentence-transformers + torch
  takes a few seconds on a fresh process — this is a one-time cost
  (`ProgressEngine`/`VectorRanker` are instantiated once at server/
  benchmark startup, not per-request).
- **CORS**: `main.py` sets `allow_origins=["*"]` with
  `allow_credentials=False` — this is the correct, actually-functional
  configuration for "allow all origins"; browsers reject credentialed
  requests against a wildcard origin regardless, so setting
  `allow_credentials=True` alongside `"*"` would silently not work.
- **`data/evidence_ledger.json`** is created automatically on first write
  and grows over the life of a running server — delete it to reset the
  audit trail for a fresh demo session.

---

## 6. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `run_demo.py` says "Port 8000 is already in use" | Another instance is already running — either use it as-is, or stop it, or run with `--port 8080`. |
| `ModuleNotFoundError: No module named 'spacy'` (or similar) | `pip install -r requirements.txt` wasn't run, or you're not in the virtual environment. |
| `EntityExtractor` logs "spaCy model not installed" | Run `python -m spacy download en_core_web_sm`. Not fatal — regex-only fallback still works. |
| Frontend shows "Offline · Demo Mode" even though the server is running | Confirm the backend is actually reachable at `http://localhost:8000/` in a browser tab; check for a firewall or a non-default `--port` (the frontend's `API_BASE` is hardcoded to port 8000). |
| `test_benchmark.py` / `run_demo.py` says dataset not found | Run `python synthetic_generator.py`, or just let `run_demo.py` generate it automatically. |
| Backend takes ~1-2 minutes to become "healthy" on first launch | Expected on a machine with no internet access to huggingface.co — `VectorRanker` retries twice before falling back to TF-IDF. Subsequent restarts are faster once you're past the network timeout, or if you pre-download the model. |
