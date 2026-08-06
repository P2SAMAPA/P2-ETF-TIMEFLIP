# P2-TIMEFLIP

**Foundation Model Finetuner — Parameter-Efficient Adaptation of Time-Series Foundation Models**

Part of the **P2Quant Engine Suite** · P2SAMAPA

---

## What This Engine Does

This engine attaches a lightweight **LoRA (Low-Rank Adaptation)** adapter to a pre-trained Time-Series Foundation Model (TimesFM, Moirai, or Chronos) and finetunes it on your specific ETF/Equity universe.

### Theory

**Time-Series Foundation Models:**
- Pre-trained on massive time-series datasets
- Capture universal patterns in sequential data
- Can be adapted to specific domains

**LoRA (Low-Rank Adaptation):**
- Injects trainable low-rank matrices into the model
- W' = W + α * B @ A (where r << d)
- Only < 2% of parameters are trainable
- Prevents catastrophic forgetting

**Parameter-Efficient Finetuning:**
- Adapt foundation models with minimal parameters
- Fast training and inference
- Domain adaptation without losing general capabilities

---

## Key Metrics

| Metric | What it tells you |
|--------|-------------------|
| **z-score** | Cross-sectional ranking of foundation model forecast |
| **Forecast** | Predicted future return |
| **Loss** | Finetuning loss (lower = better) |
| **Finetuned** | Whether the model was adapted to your data |
| **LoRA Params** | Number of trainable parameters |

---

## Supported Models

| Model | Repository | Description |
|-------|------------|-------------|
| **TimesFM** | google/timesfm-1.0-200m-pytorch | Google's Time-Series Foundation Model |
| **Moirai** | Salesforce/moirai-1.0-R-large | Salesforce's time-series model |
| **Chronos** | amazon/chronos-t5-small | Amazon's time-series forecasting |

---

## Windows

| Window | Purpose |
|--------|---------|
| 63d | Short-term adaptation |
| 126d | Medium-term adaptation |
| 252d | Core signal (primary) |
| 504d | Long-term adaptation |

---

## Interpretation

| z-score | Action | Meaning |
|---------|--------|---------|
| **> 0.1** | BUY | Foundation model predicts positive returns |
| **-0.1 to 0.1** | HOLD | Neutral forecast |
| **< -0.1** | SELL | Foundation model predicts negative returns |

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-TIMEFLIP
cd P2-TIMEFLIP
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py

streamlit run streamlit_app.py
