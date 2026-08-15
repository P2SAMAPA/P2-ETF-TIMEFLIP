"""
config.py  —  Configuration for TIMEFLIP Engine
================================================

Defines:
  - UNIVERSES: ETF ticker sets
  - MODEL: Foundation model selection and architecture
  - LORA: Low-Rank Adaptation parameters
  - TRAINING: Finetuning parameters
  - FORECAST: Forecasting parameters
  - WINDOWS: Time windows for analysis
"""

# ── HuggingFace ──────────────────────────────────────────────────────────────

HF_TOKEN = ""
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-timeflip-results"


# ── ETF Universes ────────────────────────────────────────────────────────────

UNIVERSES = {
    "FI_COMMODITIES": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
    ],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}


# ── Windows ──────────────────────────────────────────────────────────────────

WINDOWS = [63, 126, 252, 504]
WINDOW_LABELS = {
    63: "63d  (~3 months) — Short-term",
    126: "126d (~6 months) — Medium-term",
    252: "252d (~1 year) — Core Signal",
    504: "504d (~2 years) — Long-term",
}
PRIMARY_WINDOW = 252


# ── Foundation Model Selection ─────────────────────────────────────────────

MODEL = {
    # Choose one: "timesfm", "moirai", "chronos"
    "model_type": "timesfm",   # or "moirai", "chronos"
    "repo_id": "google/timesfm-1.0-200m-pytorch",
    "context_len": 512,        # Context window for forecasting
    "horizon_len": 128,        # Forecast horizon
    "input_patch_len": 32,     # Patching length
    "output_patch_len": 128,   # Output patch length
    "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
}


# ── LoRA Parameters ────────────────────────────────────────────────────────

LORA = {
    "rank": 8,                 # LoRA rank (r)
    "alpha": 16,               # LoRA alpha scaling
    "dropout": 0.1,            # LoRA dropout
    "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],  # Attention modules
    "include_mlp": True,       # Include MLP layers
}


# ── Training Parameters ─────────────────────────────────────────────────────

TRAINING = {
    "learning_rate": 1e-4,     # Learning rate for LoRA
    "n_epochs": 20,            # Training epochs
    "batch_size": 16,          # Batch size
    "weight_decay": 0.01,      # Weight decay
    "warmup_steps": 100,       # Warmup steps
    "gradient_clip": 1.0,      # Gradient clipping
    "early_stopping": True,    # Early stopping
    "patience": 5,             # Patience for early stopping
}


# ── Forecasting Parameters ──────────────────────────────────────────────────

FORECAST = {
    "context_window": 512,     # Context window for forecasting
    "forecast_horizon": 32,    # Forecast horizon (days ahead)
    "n_samples": 20,           # Number of samples for probabilistic forecast
}


# ── Macro Signals ────────────────────────────────────────────────────────────

MACRO_SIGNALS = [
    ("VIX",       "VIX",           0.30, -1.0),
    ("T10Y2Y",    "10Y–2Y Spread", 0.25, +1.0),
    ("DXY",       "DXY",           0.20, -1.0),
    ("IG_SPREAD", "IG Spread",     0.15, -1.0),
    ("HY_SPREAD", "HY Spread",     0.10, -1.0),
]

MACRO_COLS_CORE = ["VIX", "T10Y2Y", "DXY"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]
