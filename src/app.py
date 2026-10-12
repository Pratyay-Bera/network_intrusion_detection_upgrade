import random
from pathlib import Path
from textwrap import dedent

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ============================================================
# NETGUARD AI — SINGLE-FILE STREAMLIT APPLICATION
# Professional corporate-style interface. Model inference
# remains in this app.py.
# ============================================================

st.set_page_config(
    page_title="NetGuard AI | Intrusion Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent

# Project root: if app.py lives in a "src" folder, the root is its parent.
# Otherwise (app.py already in the root) the root is BASE_DIR itself.
ROOT_DIR = BASE_DIR.parent if BASE_DIR.name.lower() == "src" else BASE_DIR

# The "outputs" folder sits in the ROOT folder, not inside src.
OUTPUT_DIR = ROOT_DIR / "outputs"

ARTIFACT_NAMES = [
    "best_model.pkl",
    "preprocessor.pkl",
    "scaler.pkl",
    "pca.pkl",
]

# ------------------------- CSS --------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --primary:#1d4ed8; --primary-dark:#1e3a8a; --primary-soft:#eff4ff;
  --green:#059669; --green-soft:#ecfdf5;
  --red:#dc2626; --red-soft:#fef2f2;
  --amber:#d97706; --purple:#7c3aed; --teal:#0d9488; --blue:#2563eb;
  --bg:#f6f8fb; --surface:#ffffff; --surface2:#f1f5f9;
  --border:#e2e8f0; --border2:#cbd5e1;
  --text:#334155; --heading:#0f172a; --muted:#94a3b8; --muted2:#64748b;
  --shadow:0 1px 2px rgba(15,23,42,.05),0 4px 14px rgba(15,23,42,.05);
}
*,*::before,*::after { box-sizing:border-box; }
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],.stApp {
  background:var(--bg)!important; color:var(--text);
  font-family:'Inter',sans-serif;
}
[data-testid="stHeader"],footer,#MainMenu { background:transparent!important; }
[data-testid="stToolbar"] { right:1rem; }
[data-testid="stMainBlockContainer"] { max-width:1280px!important; padding:0 2rem 1.5rem!important; margin:0 auto; }
.block-container { max-width:1280px!important; padding:0 2rem 2rem!important; }
[data-testid="stVerticalBlock"] { gap:.9rem; }
[data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--border); }
h1,h2,h3,h4,h5 { color:var(--heading)!important; font-family:'Inter',sans-serif!important; font-weight:700!important; letter-spacing:-.01em; }
p,li,span,label { color:var(--text); }
hr { border-color:var(--border)!important; margin:14px 0!important; }
.stMarkdown { position:relative; z-index:2; }
div[data-testid="stAlert"] { border-radius:8px; }
[data-testid="stMetric"] {
  background:var(--surface); border:1px solid var(--border); border-radius:10px;
  padding:18px 20px; box-shadow:var(--shadow);
}
[data-testid="stMetricLabel"] p {
  color:var(--muted2)!important; font-family:'Inter',sans-serif!important;
  text-transform:uppercase; letter-spacing:.06em; font-size:11px!important; font-weight:600;
}
[data-testid="stMetricValue"] { color:var(--heading)!important; font-family:'Inter',sans-serif; font-weight:700; }
[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:10px; overflow:hidden; box-shadow:var(--shadow); }
[data-testid="stFileUploader"] {
  border:1.5px dashed var(--border2); border-radius:12px;
  background:var(--surface); padding:10px; transition:border-color .2s,background .2s;
}
[data-testid="stFileUploader"]:hover { border-color:var(--primary); background:var(--primary-soft); }
[data-testid="stFileUploader"] section { background:transparent!important; }
[data-testid="stFileUploader"] small,[data-testid="stFileUploader"] span { color:var(--muted2)!important; }
.stButton>button,.stDownloadButton>button {
  width:100%; border-radius:8px!important; border:1px solid var(--primary)!important;
  background:var(--primary)!important; color:#ffffff!important;
  font-family:'Inter',sans-serif!important; font-size:12px!important; font-weight:600!important;
  letter-spacing:.04em; text-transform:uppercase; transition:all .2s ease; padding:.6rem 1rem;
}
.stButton>button:hover,.stDownloadButton>button:hover {
  background:var(--primary-dark)!important; border-color:var(--primary-dark)!important;
  box-shadow:0 6px 16px rgba(29,78,216,.25); transform:translateY(-1px); color:#fff!important;
}
.stDownloadButton>button p,.stButton>button p { color:#fff!important; }
[data-testid="stSelectbox"] label p,[data-testid="stFileUploader"] label p {
  color:var(--muted2)!important; font-family:'Inter',sans-serif!important;
  font-size:11px!important; text-transform:uppercase; letter-spacing:.06em; font-weight:600;
}
[data-baseweb="select"]>div { background:var(--surface); border-color:var(--border2); border-radius:8px; color:var(--heading); }
[data-testid="stExpander"] { background:var(--surface); border:1px solid var(--border)!important; border-radius:10px; box-shadow:var(--shadow); }
[data-testid="stExpander"] summary p { color:var(--heading)!important; font-weight:600; }
code,pre { font-family:'JetBrains Mono',monospace!important; }
[data-testid="stCaptionContainer"] { color:var(--muted2)!important; }

/* Info strip (formerly ticker) */
.ticker-wrap {position:relative;z-index:5;overflow:hidden;height:38px;display:flex;align-items:center;
 background:var(--primary-dark);margin:0 -2rem;}
.ticker-track {display:flex;width:max-content;animation:tickerMove 70s linear infinite;}
.ticker-wrap:hover .ticker-track {animation-play-state:paused;}
.ticker-item {display:flex;align-items:center;gap:9px;padding:0 24px;white-space:nowrap;
 font:500 11px 'Inter',sans-serif;border-right:1px solid rgba(255,255,255,.14);}
.ticker-item .symbol {color:rgba(255,255,255,.6);letter-spacing:.08em;font-weight:500;}
.ticker-item .value {color:#ffffff;font-weight:600;}
.ticker-item .green {color:#6ee7b7;}
.ticker-item .blue {color:#93c5fd;}
.ticker-item .purple {color:#c4b5fd;}
.ticker-item .red {color:#fca5a5;}
@keyframes tickerMove {from{transform:translateX(0)}to{transform:translateX(-50%)}}

/* Navigation and hero */
.navbar {position:relative;z-index:5;display:flex;align-items:center;justify-content:space-between;
 padding:16px 28px;margin:0 -2rem;background:var(--surface);border-bottom:1px solid var(--border);
 box-shadow:0 1px 3px rgba(15,23,42,.04);}
.brand {font:800 19px 'Inter',sans-serif;letter-spacing:.02em;color:var(--heading);text-transform:uppercase;}
.brand span {color:var(--primary);}
.nav-meta {font:500 11px 'Inter',sans-serif;letter-spacing:.12em;color:var(--muted2);text-transform:uppercase;}
.nav-chip {border:1px solid #bfd0fb;padding:7px 14px;color:var(--primary);border-radius:999px;
 background:var(--primary-soft);font:600 11px 'Inter',sans-serif;letter-spacing:.06em;}
.hero {position:relative;z-index:2;text-align:center;padding:72px 5% 56px;margin:0 -2rem;
 background:linear-gradient(180deg,#ffffff 0%,#eef3fc 100%);border-bottom:1px solid var(--border);
 animation:fadeUp .7s ease both;}
.eyebrow {display:inline-flex;align-items:center;gap:12px;color:var(--primary);
 font:600 11px 'Inter',sans-serif;letter-spacing:.2em;text-transform:uppercase;margin-bottom:22px;}
.eyebrow:before,.eyebrow:after {content:'';height:2px;width:26px;background:var(--primary);border-radius:2px;}
.hero-title {font:800 clamp(38px,6vw,68px)/1.05 'Inter',sans-serif;color:var(--heading);letter-spacing:-.035em;}
.hero-title .accent {background:linear-gradient(100deg,#1d4ed8,#0ea5e9);
 -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;}
.hero-sub {max-width:700px;margin:22px auto 0;color:var(--muted2);
 font-size:16px;line-height:1.75;font-weight:400;}
.hero-badges {display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:28px;}
.hero-badge {font:600 11px 'Inter',sans-serif;letter-spacing:.05em;padding:8px 14px;
 border:1px solid var(--border2);border-radius:999px;background:var(--surface);}
.badge-green {color:var(--green);border-color:#a7f3d0;background:var(--green-soft);}
.badge-blue {color:var(--blue);border-color:#bfdbfe;background:#eff6ff;}
.badge-purple {color:var(--purple);border-color:#ddd6fe;background:#f5f3ff;}
.pulse-dot {display:inline-block;width:7px;height:7px;background:var(--green);border-radius:50%;
 margin-right:7px;animation:pulse 1.8s infinite;}
@keyframes pulse {50%{opacity:.35}}
@keyframes fadeUp {from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}

/* Sections/cards */
.section-label {position:relative;z-index:2;display:flex;align-items:center;gap:16px;
 margin:40px 0 18px;color:var(--heading);font:700 13px 'Inter',sans-serif;
 letter-spacing:.1em;text-transform:uppercase;}
.section-label:after {content:'';height:1px;flex:1;background:var(--border2);}
.module-grid {position:relative;z-index:2;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin:0;}
.module-card {position:relative;min-height:220px;overflow:hidden;background:var(--surface);
 border:1px solid var(--border);border-radius:12px;box-shadow:var(--shadow);
 padding:24px 24px 20px;transition:box-shadow .25s,transform .25s,border-color .25s;}
.module-card:hover {transform:translateY(-3px);box-shadow:0 10px 28px rgba(15,23,42,.10);border-color:var(--accent);}
.module-card:after {content:'';position:absolute;top:0;left:0;width:100%;height:3px;background:var(--accent);}
.card-top {display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;}
.card-index {color:var(--muted);font:600 11px 'Inter',sans-serif;letter-spacing:.1em;}
.card-status {font:700 9px 'Inter',sans-serif;letter-spacing:.1em;padding:4px 9px;border-radius:999px;
 background:color-mix(in srgb,var(--accent) 10%,white);
 border:1px solid color-mix(in srgb,var(--accent) 30%,white);color:var(--accent);}
.card-icon {font-size:26px;margin-bottom:10px;}
.card-title {font:700 18px 'Inter',sans-serif;color:var(--heading);margin-bottom:8px;}
.card-desc {font-size:13px;line-height:1.7;color:var(--muted2);max-width:420px;}
.card-tags {display:flex;gap:6px;flex-wrap:wrap;margin-top:16px;}
.card-tag {font:600 10px 'Inter',sans-serif;color:var(--muted2);background:var(--surface2);
 border:1px solid var(--border);padding:4px 9px;border-radius:6px;letter-spacing:.04em;}
.card-number {position:absolute;bottom:-26px;right:12px;font:800 100px 'Inter',sans-serif;
 color:rgba(15,23,42,.035);pointer-events:none;}
.stats-bar {position:relative;z-index:2;display:grid;grid-template-columns:repeat(4,1fr);
 margin:24px 0 0;border:1px solid var(--border);background:var(--surface);border-radius:12px;box-shadow:var(--shadow);}
.stat-item {padding:22px 12px;text-align:center;border-right:1px solid var(--border);}
.stat-item:nth-child(4n) {border-right:0;}
.stat-val {font:800 28px 'Inter',sans-serif;color:var(--primary);}
.stat-label {margin-top:6px;font:600 11px 'Inter',sans-serif;color:var(--muted2);letter-spacing:.08em;text-transform:uppercase;}
.content-wrap {position:relative;z-index:2;margin:0;}
.panel {border:1px solid var(--border);background:var(--surface);padding:22px;margin-bottom:16px;
 border-radius:12px;box-shadow:var(--shadow);}
.info-card {border:1px solid #a7f3d0;background:var(--green-soft);border-radius:10px;padding:12px 14px;}
.panel-title {font:700 15px 'Inter',sans-serif;color:var(--heading);margin-bottom:6px;letter-spacing:.01em;}
.panel-subtitle {font:500 11px 'Inter',sans-serif;color:var(--muted2);letter-spacing:.06em;margin-bottom:16px;text-transform:uppercase;}
.small-note {font:400 12px 'Inter',sans-serif;color:var(--muted2);line-height:1.7;}
.disclaimer {position:relative;z-index:2;margin:44px 0 0;border:1px solid #fecaca;
 border-left:5px solid var(--red);background:var(--red-soft);padding:22px 26px;border-radius:10px;}
.disclaimer-heading {font:700 13px 'Inter',sans-serif;letter-spacing:.06em;color:#b91c1c;margin-bottom:10px;}
.disclaimer-text {font-size:13px;line-height:1.8;color:#475569;}
.disclaimer-tags {display:flex;gap:8px;flex-wrap:wrap;margin-top:14px;}
.disclaimer-tag {font:600 10px 'Inter',sans-serif;padding:5px 10px;border:1px solid #fecaca;
 color:#b91c1c;background:#fff;border-radius:999px;letter-spacing:.04em;}
.footer {position:relative;z-index:2;text-align:center;padding:30px 5%;margin:36px -2rem -1.5rem;
 background:var(--primary-dark);}
.footer-brand {font:800 14px 'Inter',sans-serif;letter-spacing:.1em;color:#ffffff;}
.footer-brand span {color:#93c5fd;}
.footer-note {font:500 10px 'Inter',sans-serif;letter-spacing:.1em;color:rgba(255,255,255,.6);margin-top:9px;}

@media(max-width:950px) {
 .module-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
 .hero {padding:56px 5% 40px;}
}
@media(max-width:600px) {
 .module-grid {grid-template-columns:1fr;}
 .stats-bar {grid-template-columns:repeat(2,1fr);}
 .stat-item:nth-child(2n) {border-right:0;}
 .navbar {padding:14px 5%;}
 .nav-meta {display:none;}
 .hero-title {font-size:38px;}
 .panel {padding:14px;}
 .disclaimer {padding:16px;}
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------- Ticker (illustrative) -------------------
# These are deliberately labeled as illustrative values, not live data.
ticker_data = [
    ("NETWORK", "TRAFFIC", "green"),
    ("MODEL", "XGBOOST", "blue"),
    ("DATASET", "UNSW-NB15", "purple"),
    ("PIPELINE", "PREPROCESS → PCA", "green"),
    ("MODE", "BATCH ANALYSIS", "blue"),
    ("OUTPUT", "NORMAL / ATTACK", "purple"),
]
ticker_items = "".join(
    f'<div class="ticker-item"><span class="symbol">{name}</span>'
    f'<span class="value">{value}</span><span class="{color}">◆</span></div>'
    for name, value, color in ticker_data
)
st.markdown(
    f'<div class="ticker-wrap"><div class="ticker-track">{ticker_items}{ticker_items}</div></div>',
    unsafe_allow_html=True,
)

# --------------------- Navigation / hero -----------------------
st.markdown(
    """
<div class="navbar">
  <div class="brand">NET<span>GUARD</span> AI</div>
  <div class="nav-meta">NETWORK DEFENSE / ML INTELLIGENCE</div>
  <div class="nav-chip">UNSW-NB15 · XGBOOST</div>
</div>
<div class="hero">
  <div class="eyebrow">Network Security · Machine Learning</div>
  <div class="hero-title">TRAFFIC,<br><span class="accent">DECODED.</span></div>
  <p class="hero-sub">
    Analyze network flow records, identify potentially malicious traffic,
    and inspect model outputs through one unified intrusion intelligence dashboard.
    Built for educational research and preliminary analysis.
  </p>
  <div class="hero-badges">
    <div class="hero-badge badge-green"><span class="pulse-dot"></span>MODEL INFERENCE</div>
    <div class="hero-badge badge-purple">⬡ XGBOOST CLASSIFIER</div>
    <div class="hero-badge badge-blue">◈ LOCAL CSV ANALYSIS</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------- Module cards ----------------------------
st.markdown(
    '<div class="section-label"><span>01 / Detection Modules</span></div>',
    unsafe_allow_html=True,
)

modules = [
    ("01", "🧠", "Traffic Classification",
     "Apply the saved preprocessing pipeline, scaler, PCA transformation, and XGBoost classifier.",
     "MODEL INFERENCE", "#1d4ed8", ["XGBOOST", "PREDICTION"]),
    ("02", "📂", "CSV Intake",
     "Upload a network traffic dataset and preview records before running the model.",
     "DATA INPUT", "#0284c7", ["CSV", "VALIDATION"]),
    ("03", "🚨", "Threat Summary",
     "Review normal and potential attack counts, percentages, and individual predictions.",
     "CLASSIFICATION", "#dc2626", ["NORMAL", "ATTACK"]),
    ("04", "📊", "Traffic Analytics",
     "Explore class distribution charts and export prediction results for further review.",
     "VISUALIZATION", "#7c3aed", ["CHARTS", "EXPORT"]),
    ("05", "🧪", "Model Evaluation",
     "Compare predictions against supplied ground-truth labels when valid binary labels are available.",
     "OPTIONAL", "#d97706", ["METRICS", "CONFUSION MATRIX"]),
    ("06", "🗂️", "Training Insights",
     "View saved training-time visualizations for SMOTE, feature importance, correlation, and model comparison.",
     "ARTIFACTS", "#0d9488", ["SMOTE", "FEATURES"]),
]
cards = "".join(
    dedent(f"""
        <div class="module-card" style="--accent:{accent}">
          <div class="card-top">
            <span class="card-index">{idx} / 06</span>
            <span class="card-status">{status}</span>
          </div>
          <div class="card-icon">{icon}</div>
          <div class="card-title">{title}</div>
          <div class="card-desc">{desc}</div>
          <div class="card-tags">{''.join(f'<span class="card-tag">{tag}</span>' for tag in tags)}</div>
          <div class="card-number">{idx}</div>
        </div>
    """).strip()
    for idx, icon, title, desc, status, accent, tags in modules
)
st.markdown(
    f'<div class="module-grid">{cards}</div>',
    unsafe_allow_html=True,
)

# --------------------- Static stats bar ------------------------
stats = [
    ("XGB", "Classifier"),
    ("4", "Saved Artifacts"),
    ("CSV", "Input Format"),
    ("0 / 1", "Output Classes"),
]
stats_html = "".join(
    f'<div class="stat-item"><div class="stat-val">{value}</div>'
    f'<div class="stat-label">{label}</div></div>'
    for value, label in stats
)
st.markdown(f'<div class="stats-bar">{stats_html}</div>', unsafe_allow_html=True)


# --------------------- Artifact loading ------------------------
def find_artifact(filename):
    # Look in the root-level "models" folder first, then fall back to
    # src-level / working-directory locations.
    candidates = [
        ROOT_DIR / "models" / filename,
        BASE_DIR / "models" / filename,
        Path.cwd() / "models" / filename,
        ROOT_DIR / filename,
        BASE_DIR / filename,
        Path.cwd() / filename,
    ]
    checked = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in checked:
            checked.add(candidate)
            if candidate.is_file():
                return candidate
    return None


@st.cache_resource(show_spinner=False)
def load_artifacts():
    paths = {name: find_artifact(name) for name in ARTIFACT_NAMES}
    missing = [name for name, path in paths.items() if path is None]
    if missing:
        expected = ROOT_DIR / "models"
        raise FileNotFoundError(
            f"Missing artifact(s): {', '.join(missing)}. "
            f"Expected files in {expected}"
        )
    try:
        model = joblib.load(paths["best_model.pkl"])
        preprocessor = joblib.load(paths["preprocessor.pkl"])
        scaler = joblib.load(paths["scaler.pkl"])
        pca = joblib.load(paths["pca.pkl"])
    except Exception as exc:
        raise RuntimeError(
            f"{type(exc).__name__}: {exc}. "
            "Check artifact integrity and the scikit-learn/XGBoost versions "
            "used when the files were saved."
        ) from exc
    return model, preprocessor, scaler, pca, paths


# --------------------- Main workspace --------------------------
st.markdown(
    '<div class="section-label"><span>02 / Live Workspace</span></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown('<div class="panel-title">▣ DATA INGESTION</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-subtitle">UPLOAD A COMPATIBLE NETWORK TRAFFIC CSV</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"],
        help="The columns must match the feature schema expected by training.",
        label_visibility="collapsed",
    )
    st.markdown(
        '<p class="small-note">INPUT: CSV · OUTPUT: NORMAL / POTENTIAL ATTACK'
        '<br>Predictions require the same feature pipeline used during training.</p>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="panel-title">◈ SYSTEM STATUS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-subtitle">MODEL ARTIFACTS AND INFERENCE READINESS</div>',
        unsafe_allow_html=True,
    )
    try:
        with st.spinner("Loading saved model artifacts..."):
            model, preprocessor, scaler, pca, artifact_paths = load_artifacts()
        artifacts_loaded = True
        st.success("Model artifacts loaded. Prediction pipeline is ready.")
        status_cols = st.columns(4)
        for col, filename in zip(status_cols, ARTIFACT_NAMES):
            col.markdown(
                f'<div class="info-card"><div class="small-note">ARTIFACT</div>'
                f'<div style="color:#0f172a;font-family:JetBrains Mono,monospace;font-weight:500;'
                f'font-size:11px;margin-top:8px;overflow-wrap:anywhere;">'
                f'{filename}</div><div style="color:#059669;font-weight:600;font-size:11px;margin-top:8px;">'
                f'● FOUND</div></div>',
                unsafe_allow_html=True,
            )
    except Exception as exc:
        artifacts_loaded = False
        st.error("Model loading failed. Predictions are disabled until the issue is fixed.")
        with st.expander("Technical details — click to diagnose", expanded=True):
            st.code(f"{type(exc).__name__}: {exc}")
            st.write("Application directory:")
            st.code(str(BASE_DIR))
            st.write("Project root directory:")
            st.code(str(ROOT_DIR))
            st.write("Expected artifact paths:")
            for filename in ARTIFACT_NAMES:
                path = find_artifact(filename)
                if path:
                    st.success(f"{filename}: {path}")
                else:
                    st.error(f"{filename}: NOT FOUND — {ROOT_DIR / 'models' / filename}")
            st.markdown(
                "**Troubleshooting:** Check the `models` directory and filenames. "
                "If every file exists, check whether the saved artifacts were created "
                "with compatible Python, scikit-learn, and XGBoost versions."
            )

# --------------------- Prediction workflow ---------------------
if uploaded_file is not None:
    if not artifacts_loaded:
        st.warning("CSV received, but inference cannot run until model loading succeeds.")
    else:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.astype(str).str.strip()

            if df.empty:
                st.warning("The uploaded CSV contains no records.")
            elif df.columns.duplicated().any():
                st.error("Duplicate column names found after trimming whitespace.")
            else:
                st.markdown(
                    '<div class="section-label"><span>03 / Prediction Results</span></div>',
                    unsafe_allow_html=True,
                )
                a, b, c = st.columns(3)
                a.metric("Records", f"{len(df):,}")
                b.metric("Input Columns", f"{len(df.columns):,}")
                c.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

                with st.expander("Preview input records", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True)

                # A label column is optional and is never passed as a feature.
                if "label" in df.columns:
                    X = df.drop(columns=["label"]).copy()
                    has_label = True
                else:
                    X = df.copy()
                    has_label = False

                # Keep this list consistent with the actual training script.
                cols_to_drop = [col for col in ["id", "stcpb", "dtcpb"] if col in X.columns]
                if cols_to_drop:
                    X = X.drop(columns=cols_to_drop)

                if X.shape[1] == 0:
                    st.error("No feature columns remain after feature selection.")
                else:
                    with st.spinner("Running saved preprocessing and XGBoost inference..."):
                        X_processed = preprocessor.transform(X)
                        X_scaled = scaler.transform(X_processed)
                        X_pca = pca.transform(X_scaled)
                        predictions = model.predict(X_pca)

                    if len(predictions) != len(df):
                        raise ValueError("Prediction count does not match input row count.")

                    try:
                        pred_codes = pd.Series(predictions).astype(int)
                    except (ValueError, TypeError) as exc:
                        raise ValueError("Model returned non-numeric class labels.") from exc

                    if not set(pred_codes.unique()).issubset({0, 1}):
                        raise ValueError(
                            "Unexpected model classes. This dashboard expects "
                            "0 = Normal and 1 = Attack."
                        )

                    results = df.copy()
                    results["Prediction Code"] = pred_codes.values
                    results["Prediction"] = pred_codes.map({
                        0: "Normal",
                        1: "Potential Attack",
                    }).values

                    normal_count = int((pred_codes == 0).sum())
                    attack_count = int((pred_codes == 1).sum())
                    total = len(pred_codes)
                    normal_pct = normal_count / total * 100
                    attack_pct = attack_count / total * 100

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Analyzed", f"{total:,}")
                    m2.metric("Normal", f"{normal_count:,}")
                    m3.metric("Potential Attacks", f"{attack_count:,}")
                    m4.metric("Attack Rate", f"{attack_pct:.2f}%")

                    if attack_count:
                        st.warning(
                            f"{attack_count:,} record(s) were flagged as potential attacks. "
                            "These predictions are not confirmed incidents; investigate independently."
                        )
                    else:
                        st.success(
                            "No records were flagged as attacks. This does not guarantee "
                            "that the uploaded traffic is safe."
                        )

                    filter_choice = st.selectbox(
                        "Filter results",
                        ["All Records", "Normal Traffic", "Potential Attacks"],
                    )
                    if filter_choice == "Normal Traffic":
                        visible = results[results["Prediction Code"] == 0]
                    elif filter_choice == "Potential Attacks":
                        visible = results[results["Prediction Code"] == 1]
                    else:
                        visible = results

                    st.caption(f"Displaying {len(visible):,} of {total:,} records.")
                    st.dataframe(visible, use_container_width=True, height=380)

                    st.download_button(
                        "↓ EXPORT PREDICTIONS AS CSV",
                        data=results.to_csv(index=False).encode("utf-8"),
                        file_name="netguard_intrusion_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    st.markdown("#### Predicted traffic distribution")
                    chart_col, summary_col = st.columns([1, 1])
                    with chart_col:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        fig.patch.set_facecolor("#ffffff")
                        ax.set_facecolor("#ffffff")
                        ax.pie(
                            [normal_count, attack_count],
                            labels=["Normal", "Potential Attack"],
                            autopct="%1.1f%%",
                            startangle=90,
                            colors=["#10b981", "#ef4444"],
                            textprops={"color": "#1e293b", "fontsize": 10},
                            wedgeprops={"edgecolor": "#ffffff", "linewidth": 2},
                        )
                        ax.axis("equal")
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                    with summary_col:
                        st.markdown(
                            f"""
                            <div class="panel">
                              <div style="color:#059669;font:700 11px Inter,sans-serif;
                              letter-spacing:.1em;">NORMAL TRAFFIC</div>
                              <div style="font:800 32px Inter,sans-serif;color:#0f172a;margin:5px 0;">
                              {normal_count:,}</div>
                              <div class="small-note">{normal_pct:.2f}% of uploaded records</div>
                              <hr>
                              <div style="color:#dc2626;font:700 11px Inter,sans-serif;
                              letter-spacing:.1em;">POTENTIAL ATTACKS</div>
                              <div style="font:800 32px Inter,sans-serif;color:#0f172a;margin:5px 0;">
                              {attack_count:,}</div>
                              <div class="small-note">{attack_pct:.2f}% of uploaded records</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    if has_label:
                        st.markdown("#### Optional ground-truth evaluation")
                        st.info(
                            "Only use these metrics when `label` contains genuine binary "
                            "ground-truth values (0/1)."
                        )
                        try:
                            from sklearn.metrics import (
                                accuracy_score, f1_score, recall_score, confusion_matrix
                            )
                            y_true = pd.to_numeric(df["label"], errors="raise")
                            if y_true.isin([0, 1]).all():
                                e1, e2, e3 = st.columns(3)
                                e1.metric(
                                    "Accuracy",
                                    f"{accuracy_score(y_true, pred_codes) * 100:.2f}%",
                                )
                                e2.metric(
                                    "Macro F1",
                                    f"{f1_score(y_true, pred_codes, average='macro', zero_division=0) * 100:.2f}%",
                                )
                                e3.metric(
                                    "Attack Recall",
                                    f"{recall_score(y_true, pred_codes, pos_label=1, zero_division=0) * 100:.2f}%",
                                )
                                cm = confusion_matrix(y_true, pred_codes, labels=[0, 1])
                                st.dataframe(
                                    pd.DataFrame(
                                        cm,
                                        index=["Actual Normal", "Actual Attack"],
                                        columns=["Predicted Normal", "Predicted Attack"],
                                    ),
                                    use_container_width=True,
                                )
                                st.caption(
                                    "Metrics are not independent validation if these records "
                                    "were used during model training or tuning."
                                )
                            else:
                                st.warning("Labels are not all encoded as binary 0/1; metrics skipped.")
                        except (ValueError, TypeError):
                            st.warning("Could not interpret the label column as binary ground truth.")

        except Exception as exc:
            st.error(
                "Prediction failed. Verify the uploaded CSV schema and ensure the saved "
                "preprocessor, scaler, PCA, and model match the training pipeline."
            )
            with st.expander("Technical details — prediction error"):
                st.code(f"{type(exc).__name__}: {exc}")
            st.caption("A failed prediction is not evidence that network traffic is safe.")

# --------------------- Training-time insights ------------------
st.markdown(
    '<div class="section-label"><span>04 / Training Intelligence</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="content-wrap"><p class="small-note">These visualizations are saved '
    'training-time artifacts. They are not recalculated from the uploaded file.</p></div>',
    unsafe_allow_html=True,
)


def show_image_if_present(filename, caption=None):
    # Images live in <project root>/outputs (not inside src).
    path = OUTPUT_DIR / filename
    if path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Visualization not found: {path}")



with st.expander("Class balance / before SMOTE", expanded=False):
    show_image_if_present("before_smote.png")

with st.expander("Class balance / after SMOTE", expanded=False):
    show_image_if_present("after_smote.png")

with st.expander("Correlation heatmap"):
    show_image_if_present("correlation_heatmap.png")

with st.expander("Feature importance"):
    show_image_if_present("feature_importance.png")

with st.expander("Model performance comparison"):
    show_image_if_present("model_comparison.png")
    st.markdown(
        """
        Previously reported XGBoost scores (verify against your evaluation report):

        | Metric | Reported score |
        |---|---:|
        | Accuracy | 88.28% |
        | Macro F1 | 87.99% |
        | Attack Recall | 94.08% |
        | ROC-AUC | 97.34% |

        These historical values are not recalculated by this dashboard and do not
        guarantee future model performance.
        """
    )

# --------------------- Risk disclosure at END -----------------
st.markdown(
    """
<div class="disclaimer">
  <div class="disclaimer-heading">⚠ &nbsp; DISCLAIMER — USE THIS MODEL AT YOUR OWN RISK</div>
  <div class="disclaimer-text">
    This application provides machine-learning-based network traffic predictions for
    educational, research, and preliminary analysis purposes only. Predictions may be
    inaccurate and can produce both false positives and false negatives.
    <strong style="color:#b91c1c">Do not blindly trust or rely solely on these predictions</strong>
    for cybersecurity decisions, incident response, or critical actions. Independently
    verify findings using trusted security tools and qualified security professionals.
    No guarantee is made regarding the accuracy, completeness, or reliability of the
    results. The developers are not responsible for decisions made solely on the basis
    of these predictions. Use this system at your own risk.
  </div>
  <div class="disclaimer-tags">
    <span class="disclaimer-tag">NOT A SECURITY GUARANTEE</span>
    <span class="disclaimer-tag">FALSE POSITIVES POSSIBLE</span>
    <span class="disclaimer-tag">FALSE NEGATIVES POSSIBLE</span>
    <span class="disclaimer-tag">INDEPENDENT VERIFICATION REQUIRED</span>
  </div>
</div>
<div class="footer">
  <div class="footer-brand">NET<span>GUARD</span> AI</div>
  <div class="footer-note">UNSW-NB15 · XGBOOST · NETWORK INTRUSION INTELLIGENCE</div>
  <div class="footer-note">EDUCATIONAL &amp; RESEARCH USE · VERIFY ALL MODEL OUTPUTS</div>
</div>
""",
    unsafe_allow_html=True,
)