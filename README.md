# 🏦 Loan Approval Predictor — Streamlit Demo

A small web app (bonus task of the Data Science final project) that takes an applicant's details and predicts whether the loan is likely to be **approved** or **rejected**, explains **why**, and shows **what could change** the decision.

The model is **Logistic Regression**, the best model from the project notebook. The app repeats the notebook's cleaning, train/test split and preprocessing pipeline, so its test scores match the notebook: **accuracy 85.0%**, **precision 86.0%**, **recall 90.2%**.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `loan_approval_dataset.csv` | The dataset the model is trained on when the app starts |
| `requirements.txt` | Python packages needed |
| `.streamlit/config.toml` | Theme colour |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)

1. Create a **public GitHub repository** and upload the contents of this folder (`app.py`, `loan_approval_dataset.csv`, `requirements.txt`, `.streamlit/config.toml`).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Choose the repository and branch, and set **Main file path** to `app.py`.
5. Under **Advanced settings**, choose **Python 3.11**.
6. Click **Deploy**. The first start takes a few minutes while packages install.

---
*Educational demo trained on a synthetic dataset. Not for real lending decisions.*
