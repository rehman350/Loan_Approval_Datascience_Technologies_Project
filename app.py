"""
Loan Approval Predictor - Streamlit demo (Bonus task)
====================================================
Takes an applicant's details and predicts whether the loan will be approved,
using the best model from the project notebook (Logistic Regression).

The app repeats the notebook's cleaning, split and pipeline, so its model and
test scores are the same as in the notebook.

Run locally:  streamlit run app.py
"""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
ROLL_NUMBER = 2022105002
DATA_FILE = Path(__file__).parent / "loan_approval_dataset.csv"

APPROVED_COLOR = "#2a78d6"   # blue   -> approved / helps approval
REJECTED_COLOR = "#eb6834"   # orange -> rejected / pushes towards rejection
YOU_COLOR = "#4a3aa7"        # violet -> the applicant being checked
MUTED_COLOR = "#8a8a8a"

EDUCATION_LEVELS = ["High School", "Bachelor", "Master", "Doctorate"]
MARITAL_OPTIONS = ["Single", "Married", "Divorced"]
EMPLOYMENT_OPTIONS = ["Salaried", "Self-Employed", "Contract"]
HOME_OPTIONS = ["Rent", "Mortgage", "Own"]
LOAN_TERMS = [12, 24, 36, 48, 60]

TEXT_COLUMNS = ["Marital_Status", "Education", "Employment_Type",
                "Home_Ownership", "Previous_Default", "Loan_Approved"]
MEAN_COLUMNS = ["Age", "Credit_Score", "Debt_to_Income"]
MEDIAN_COLUMNS = ["Employment_Years", "Annual_Income", "Existing_Debt",
                  "Loan_Amount", "Loan_Term", "Loan_to_Income"]
ORDINAL_COLUMNS = ["Education"]
NOMINAL_COLUMNS = ["Marital_Status", "Employment_Type", "Home_Ownership", "Previous_Default"]

FRIENDLY_NAMES = {
    "Age": "Age",
    "Marital_Status": "Marital status",
    "Education": "Education",
    "Employment_Type": "Employment type",
    "Employment_Years": "Years employed",
    "Annual_Income": "Annual income",
    "Credit_Score": "Credit score",
    "Existing_Debt": "Existing debt",
    "Loan_Amount": "Loan amount",
    "Loan_Term": "Loan term",
    "Home_Ownership": "Home ownership",
    "Previous_Default": "Previous default",
    "Debt_to_Income": "Debt-to-income ratio",
    "Loan_to_Income": "Loan-to-income ratio",
}

# Ready-made applicants so a first-time user can see the app working at once
EXAMPLES = {
    "strong": {"Age": 42, "Marital_Status": "Married", "Education": "Master",
               "Employment_Type": "Salaried", "Employment_Years": 14, "Annual_Income": 85000,
               "Credit_Score": 760, "Existing_Debt": 6000, "Loan_Amount": 20000,
               "Loan_Term": 36, "Home_Ownership": "Own", "Previous_Default": "No"},
    "borderline": {"Age": 35, "Marital_Status": "Single", "Education": "Bachelor",
                   "Employment_Type": "Salaried", "Employment_Years": 6, "Annual_Income": 45000,
                   "Credit_Score": 650, "Existing_Debt": 12000, "Loan_Amount": 18000,
                   "Loan_Term": 36, "Home_Ownership": "Rent", "Previous_Default": "No"},
    "risky": {"Age": 29, "Marital_Status": "Single", "Education": "High School",
              "Employment_Type": "Contract", "Employment_Years": 2, "Annual_Income": 28000,
              "Credit_Score": 590, "Existing_Debt": 14000, "Loan_Amount": 30000,
              "Loan_Term": 24, "Home_Ownership": "Rent", "Previous_Default": "Yes"},
}
DEFAULT_APPLICANT = EXAMPLES["borderline"]
INPUT_COLUMNS = list(DEFAULT_APPLICANT.keys())


# ----------------------------------------------------------------------------
# Data and model (same steps as the notebook)
# ----------------------------------------------------------------------------
def add_ratios(data):
    data = data.copy()
    data["Debt_to_Income"] = data["Existing_Debt"] / data["Annual_Income"]
    data["Loan_to_Income"] = data["Loan_Amount"] / data["Annual_Income"]
    return data


@st.cache_data
def load_clean_data():
    df = pd.read_csv(DATA_FILE)
    df = df.drop_duplicates()
    for col in TEXT_COLUMNS:
        df[col] = df[col].str.strip().str.title()
    df = df.drop_duplicates().reset_index(drop=True)
    df["Loan_Term"] = df["Loan_Term"].str.replace(" months", "").astype(int)
    df.loc[(df["Age"] < 18) | (df["Age"] > 100), "Age"] = np.nan
    df.loc[(df["Credit_Score"] < 300) | (df["Credit_Score"] > 850), "Credit_Score"] = np.nan
    df.loc[df["Existing_Debt"] < 0, "Existing_Debt"] = np.nan
    return df


@st.cache_resource
def train_model():
    df = load_clean_data()
    y = df["Loan_Approved"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Loan_Approved", "Application_ID", "Application_Date"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=ROLL_NUMBER, stratify=y)
    X_train = add_ratios(X_train)
    X_test = add_ratios(X_test)

    preprocessor = ColumnTransformer([
        ("mean_numeric", Pipeline([("imputer", SimpleImputer(strategy="mean")),
                                   ("scaler", StandardScaler())]), MEAN_COLUMNS),
        ("median_numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                     ("scaler", StandardScaler())]), MEDIAN_COLUMNS),
        ("ordinal", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encoder", OrdinalEncoder(categories=[EDUCATION_LEVELS]))]), ORDINAL_COLUMNS),
        ("nominal", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         NOMINAL_COLUMNS),
    ])

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=ROLL_NUMBER)),
    ])
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)
    train_ratios = X_train[["Debt_to_Income", "Loan_to_Income"]].copy()
    train_ratios["Loan_Approved"] = y_train

    info = {
        "accuracy": accuracy_score(y_test, test_predictions),
        "precision": precision_score(y_test, test_predictions),
        "recall": recall_score(y_test, test_predictions),
        "f1": f1_score(y_test, test_predictions),
        "confusion": confusion_matrix(y_test, test_predictions),
        "feature_names": model.named_steps["preprocessing"].get_feature_names_out(),
        "train_means": model.named_steps["preprocessing"].transform(X_train).mean(axis=0),
        "ratio_medians": train_ratios.groupby("Loan_Approved").median(),
        "credit_medians": pd.concat([X_train["Credit_Score"], y_train], axis=1)
                            .groupby("Loan_Approved")["Credit_Score"].median(),
        "n_rows": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "approval_rate": y.mean(),
    }
    return model, info


# ----------------------------------------------------------------------------
# Explanations
# ----------------------------------------------------------------------------
def original_feature(transformed_name):
    """'nominal__Employment_Type_Salaried' -> 'Employment_Type'."""
    name = transformed_name.split("__")[1]
    for col in NOMINAL_COLUMNS:
        if name.startswith(col + "_"):
            return col
    return name


def explain_decision(model, info, applicant):
    """Contribution of each feature compared with an average training applicant.

    For Logistic Regression: contribution = coefficient x (applicant value - average value),
    measured in log-odds. Positive pushes towards approval, negative towards rejection.
    """
    transformed = model.named_steps["preprocessing"].transform(add_ratios(applicant))[0]
    coefficients = model.named_steps["model"].coef_[0]
    contributions = coefficients * (transformed - info["train_means"])

    totals = {}
    for name, value in zip(info["feature_names"], contributions):
        feature = original_feature(name)
        totals[feature] = totals.get(feature, 0.0) + value

    table = pd.DataFrame({"feature": list(totals.keys()), "impact": list(totals.values())})
    table["label"] = table["feature"].map(FRIENDLY_NAMES)
    table["abs_impact"] = table["impact"].abs()
    table["direction"] = np.where(table["impact"] >= 0, "Helps approval", "Pushes towards rejection")
    return table.sort_values("abs_impact", ascending=False).reset_index(drop=True)


def first_flip(model, applicant, feature, values, target):
    """First value (in the given order) that makes the model predict `target`."""
    if len(values) == 0:
        return None
    candidates = pd.concat([applicant] * len(values), ignore_index=True)
    candidates[feature] = values
    predictions = model.predict(add_ratios(candidates))
    hits = np.where(predictions == target)[0]
    if len(hits) == 0:
        return None
    return values[hits[0]]


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def probability_chart(probability):
    color = APPROVED_COLOR if probability >= 0.5 else REJECTED_COLOR
    row = "Approval probability"

    track = alt.Chart(pd.DataFrame({"row": [row], "value": [100]})).mark_bar(
        color="rgba(138,138,138,0.25)", cornerRadius=8, size=38
    ).encode(
        x=alt.X("value:Q", scale=alt.Scale(domain=[0, 100]), title="Probability of approval (%)"),
        y=alt.Y("row:N", axis=None, title=None),
    )
    fill = alt.Chart(pd.DataFrame({"row": [row], "value": [probability * 100]})).mark_bar(
        color=color, cornerRadius=8, size=38
    ).encode(
        x="value:Q", y="row:N",
        tooltip=[alt.Tooltip("value:Q", title="Probability of approval (%)", format=".1f")],
    )
    threshold = alt.Chart(pd.DataFrame({"x": [50]})).mark_rule(
        color=MUTED_COLOR, strokeDash=[5, 4], size=2
    ).encode(x="x:Q")
    threshold_label = alt.Chart(pd.DataFrame({"x": [50], "text": ["50% decision line"]})).mark_text(
        dy=-30, color=MUTED_COLOR, fontSize=12
    ).encode(x="x:Q", text="text:N")

    return (track + fill + threshold + threshold_label).properties(height=120)


def impact_chart(table):
    base = alt.Chart(table).encode(
        y=alt.Y("label:N", sort=alt.EncodingSortField("abs_impact", order="descending"), title=None),
    )
    bars = base.mark_bar(cornerRadius=4).encode(
        x=alt.X("impact:Q", title="Effect on the decision (left = towards rejection, right = towards approval)"),
        color=alt.Color("direction:N",
                        scale=alt.Scale(domain=["Helps approval", "Pushes towards rejection"],
                                        range=[APPROVED_COLOR, REJECTED_COLOR]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("label:N", title="Feature"),
                 alt.Tooltip("direction:N", title="Effect"),
                 alt.Tooltip("impact:Q", title="Strength (log-odds)", format="+.2f")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED_COLOR).encode(x="x:Q")
    return (bars + zero).properties(height=420)


def ratio_chart(applicant, info):
    medians = info["ratio_medians"]
    rows = []
    for ratio, label in [("Debt_to_Income", "Debt-to-income"), ("Loan_to_Income", "Loan-to-income")]:
        rows.append({"ratio": label, "group": "Your application", "value": add_ratios(applicant)[ratio].iloc[0]})
        rows.append({"ratio": label, "group": "Typical approved", "value": medians.loc[1, ratio]})
        rows.append({"ratio": label, "group": "Typical rejected", "value": medians.loc[0, ratio]})
    data = pd.DataFrame(rows)
    order = ["Your application", "Typical approved", "Typical rejected"]

    return alt.Chart(data).mark_bar(cornerRadius=4).encode(
        x=alt.X("ratio:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("group:N", sort=order),
        y=alt.Y("value:Q", title="Ratio (lower is safer)"),
        color=alt.Color("group:N", sort=order,
                        scale=alt.Scale(domain=order, range=[YOU_COLOR, APPROVED_COLOR, REJECTED_COLOR]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("group:N", title="Group"), alt.Tooltip("ratio:N", title="Ratio"),
                 alt.Tooltip("value:Q", title="Value", format=".2f")],
    ).properties(height=420)


def global_chart(model, info):
    coefficients = np.abs(model.named_steps["model"].coef_[0])
    totals = {}
    for name, value in zip(info["feature_names"], coefficients):
        feature = original_feature(name)
        totals[feature] = totals.get(feature, 0.0) + value
    data = pd.DataFrame({"label": [FRIENDLY_NAMES[f] for f in totals], "weight": list(totals.values())})

    return alt.Chart(data).mark_bar(cornerRadius=4, color=YOU_COLOR).encode(
        x=alt.X("weight:Q", title="Total coefficient size (bigger = more influence)"),
        y=alt.Y("label:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("label:N", title="Feature"), alt.Tooltip("weight:Q", title="Weight", format=".2f")],
    ).properties(height=420)


# ----------------------------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------------------------
def load_example(name):
    for key, value in EXAMPLES[name].items():
        st.session_state[key] = value
    st.session_state["show_result"] = True


def decision_card(approved, probability):
    color = APPROVED_COLOR if approved else REJECTED_COLOR
    title = "✅ Likely APPROVED" if approved else "❌ Likely REJECTED"
    distance = abs(probability - 0.5)
    if distance < 0.1:
        confidence = "This is a <b>borderline</b> case: a small change in the details could flip the decision."
    elif distance < 0.3:
        confidence = "The model is <b>fairly confident</b> in this decision."
    else:
        confidence = "The model is <b>very confident</b> in this decision."

    st.markdown(
        f"""
        <div style="border-left: 8px solid {color}; background: {color}1f;
                    padding: 1.1rem 1.4rem; border-radius: 14px; height: 100%;">
            <div style="font-size: 0.85rem; letter-spacing: .05em; text-transform: uppercase; opacity: .75;">
                Model decision
            </div>
            <div style="font-size: 2.1rem; font-weight: 800; color: {color}; margin: .15rem 0 .35rem 0;">
                {title}
            </div>
            <div style="font-size: 1.1rem;">Probability of approval: <b>{probability:.1%}</b></div>
            <div style="font-size: 0.95rem; opacity: .85; margin-top: .45rem;">{confidence}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def money(value):
    return f"{value:,.0f}"


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }
        div[data-testid="stForm"] { border-radius: 16px; padding: 1.2rem 1.4rem; }
        div[data-testid="stMetric"] { background: rgba(138,138,138,0.08); border-radius: 12px; padding: .7rem 1rem; }
        .step-card { background: rgba(138,138,138,0.08); border-radius: 12px; padding: .8rem 1rem; margin-bottom: .6rem; }
        .hint { font-size: .9rem; opacity: .8; }
    </style>
    """,
    unsafe_allow_html=True,
)

model, info = train_model()

for key, value in DEFAULT_APPLICANT.items():
    if key not in st.session_state:
        st.session_state[key] = value
if "show_result" not in st.session_state:
    st.session_state["show_result"] = False

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 🏦 Loan Approval Predictor")
    st.markdown("### How to use")
    st.markdown(
        """
        <div class="step-card"><b>1. Fill in the form</b><br><span class="hint">Personal, work and loan details.</span></div>
        <div class="step-card"><b>2. Click “Predict loan decision”</b><br><span class="hint">Or load a ready-made example.</span></div>
        <div class="step-card"><b>3. Read the result</b><br><span class="hint">Decision, reasons, and what could change it.</span></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Model")
    st.markdown(f"**Logistic Regression** — the best model from the project notebook.")
    col_a, col_b = st.columns(2)
    col_a.metric("Test accuracy", f"{info['accuracy']:.1%}")
    col_b.metric("Precision", f"{info['precision']:.1%}")
    st.divider()
    st.caption("Final Project — Introduction to Data Science  \nAbdul Rehman · F2022105002")
    st.caption("⚠️ Educational demo trained on a synthetic dataset. Not for real lending decisions.")

# ---------------- Header ----------------
st.title("🏦 Loan Approval Predictor")
st.markdown(
    "Enter an applicant's details to see whether the loan is **likely to be approved**, "
    "**why** the model decided that, and **what could change** the decision."
)

tab_predict, tab_about = st.tabs(["🔮 Predict", "📊 About the model"])

# ============================================================================
# Tab 1 - Predict
# ============================================================================
with tab_predict:
    st.markdown("##### ⚡ Quick start — load an example applicant")
    ex1, ex2, ex3, _ = st.columns([1, 1, 1, 1.4])
    ex1.button("💪 Strong applicant", on_click=load_example, args=("strong",), use_container_width=True)
    ex2.button("⚖️ Borderline applicant", on_click=load_example, args=("borderline",), use_container_width=True)
    ex3.button("⚠️ Risky applicant", on_click=load_example, args=("risky",), use_container_width=True)

    with st.form("applicant_form"):
        col_personal, col_work, col_loan = st.columns(3, gap="large")

        with col_personal:
            st.markdown("#### 👤 Personal")
            st.slider("Age (years)", 18, 75, key="Age",
                      help="Applicant's age in years.")
            st.selectbox("Marital status", MARITAL_OPTIONS, key="Marital_Status")
            st.selectbox("Education (highest level)", EDUCATION_LEVELS, key="Education",
                         help="Ordered from lowest to highest.")
            st.radio("Home ownership", HOME_OPTIONS, key="Home_Ownership", horizontal=True)

        with col_work:
            st.markdown("#### 💼 Work & income")
            st.selectbox("Employment type", EMPLOYMENT_OPTIONS, key="Employment_Type")
            st.slider("Years employed", 0, 45, key="Employment_Years",
                      help="Total years of work experience.")
            st.number_input("Annual income", min_value=5000, max_value=1_000_000, step=1000,
                            key="Annual_Income", help="Yearly income before tax.")
            st.number_input("Existing debt", min_value=0, max_value=1_000_000, step=500,
                            key="Existing_Debt", help="Total amount the applicant already owes.")

        with col_loan:
            st.markdown("#### 💳 Loan & credit")
            st.number_input("Loan amount requested", min_value=500, max_value=1_000_000, step=500,
                            key="Loan_Amount")
            st.select_slider("Loan term", options=LOAN_TERMS, key="Loan_Term",
                             format_func=lambda months: f"{months} months")
            st.slider("Credit score", 300, 850, key="Credit_Score",
                      help="300 = very poor, 850 = excellent. Most applicants are between 600 and 750.")
            st.radio("Previous loan default?", ["No", "Yes"], key="Previous_Default", horizontal=True,
                     help="Has the applicant ever failed to repay a loan?")

        submitted = st.form_submit_button("🔮 Predict loan decision", type="primary", use_container_width=True)

    if submitted:
        st.session_state["show_result"] = True

    if st.session_state["show_result"]:
        applicant = pd.DataFrame([{col: st.session_state[col] for col in INPUT_COLUMNS}])
        applicant_ready = add_ratios(applicant)

        probability = model.predict_proba(applicant_ready)[0][1]
        approved = model.predict(applicant_ready)[0] == 1

        if st.session_state["Employment_Years"] > st.session_state["Age"] - 16:
            st.warning("Years employed look too high for this age. Please check the inputs.")

        st.divider()
        st.markdown("## 📋 Result")

        card_col, chart_col = st.columns([1, 1.25], gap="large")
        with card_col:
            decision_card(approved, probability)
        with chart_col:
            st.markdown("**Where this applicant falls**")
            st.altair_chart(probability_chart(probability), use_container_width=True)
            st.caption("The bar shows the probability of approval. Above the dashed 50% line the model says **Approved**.")

        # Key numbers compared with typical approved applicants
        medians = info["ratio_medians"]
        dti = applicant_ready["Debt_to_Income"].iloc[0]
        lti = applicant_ready["Loan_to_Income"].iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Debt-to-income ratio", f"{dti:.2f}",
                  delta=f"{dti - medians.loc[1, 'Debt_to_Income']:+.2f} vs typical approved", delta_color="inverse",
                  help="Existing debt ÷ annual income. Lower is safer.")
        m2.metric("Loan-to-income ratio", f"{lti:.2f}",
                  delta=f"{lti - medians.loc[1, 'Loan_to_Income']:+.2f} vs typical approved", delta_color="inverse",
                  help="Loan amount ÷ annual income. Lower is safer.")
        credit_gap = st.session_state["Credit_Score"] - info["credit_medians"].loc[1]
        m3.metric("Credit score", f"{st.session_state['Credit_Score']}",
                  delta=f"{credit_gap:+.0f} vs typical approved", delta_color="normal",
                  help="Higher is better.")

        # Why?
        st.markdown("### 🧠 Why did the model decide this?")
        explanation = explain_decision(model, info, applicant)
        why_col, ratio_col = st.columns([1.35, 1], gap="large")

        with why_col:
            st.altair_chart(impact_chart(explanation), use_container_width=True)
            st.caption("Each bar compares this applicant with an **average applicant**. "
                       "Blue bars help approval, orange bars push towards rejection. Longer bars matter more.")

        with ratio_col:
            st.altair_chart(ratio_chart(applicant, info), use_container_width=True)
            st.caption("How this applicant's ratios compare with the typical (median) approved and rejected applicants.")

        # Only list reasons with a noticeable effect (at least 0.1 log-odds)
        helping = explanation[explanation["impact"] >= 0.1].head(3)
        hurting = explanation[explanation["impact"] <= -0.1].head(3)
        reason_help, reason_hurt = st.columns(2, gap="large")
        with reason_help:
            st.markdown("**👍 Main reasons helping approval**")
            if len(helping) == 0:
                st.markdown("- Nothing stands out as helping.")
            for label in helping["label"]:
                st.markdown(f"- {label}")
        with reason_hurt:
            st.markdown("**👎 Main reasons pushing towards rejection**")
            if len(hurting) == 0:
                st.markdown("- Nothing stands out as hurting.")
            for label in hurting["label"]:
                st.markdown(f"- {label}")

        # What could change the decision? (counterfactuals)
        st.markdown("### 🔁 What could change the decision?")
        loan = st.session_state["Loan_Amount"]
        debt = st.session_state["Existing_Debt"]
        income = st.session_state["Annual_Income"]
        credit = st.session_state["Credit_Score"]

        if not approved:
            st.markdown("Each option changes **one thing only** and shows the smallest change that would make the model say **Approved**.")
            options = [
                ("💸 Ask for a smaller loan", "Loan_Amount", np.arange(loan - 500, 0, -500), loan, "Loan amount"),
                ("🧾 Pay off existing debt", "Existing_Debt", np.append(np.arange(debt - 500, -1, -500), 0), debt, "Existing debt"),
                ("📈 Increase annual income", "Annual_Income", np.arange(income + 1000, 1_000_001, 1000), income, "Annual income"),
                ("⭐ Improve credit score", "Credit_Score", np.arange(credit + 1, 851, 1), credit, "Credit score"),
            ]
            cards = st.columns(4)
            best_change = None
            results = []
            for (title, feature, values, current, label) in options:
                new_value = first_flip(model, applicant, feature, values, target=1)
                change = None if new_value is None or current == 0 else (new_value - current) / current * 100
                results.append((title, feature, current, new_value, change, label))
                if change is not None and (best_change is None or abs(change) < abs(best_change[4])):
                    best_change = (title, feature, current, new_value, change, label)

            for card, (title, feature, current, new_value, change, label) in zip(cards, results):
                with card:
                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        if new_value is None:
                            st.markdown("Not enough on its own.")
                            st.caption("Even the maximum change here does not flip the decision.")
                        else:
                            if feature == "Credit_Score":
                                st.markdown(f"From **{current}** to **{int(new_value)}**")
                            else:
                                st.markdown(f"From **{money(current)}** to **{money(new_value)}**")
                            if change is not None:
                                st.caption(f"Change: {change:+.1f}%")

            if best_change is not None and abs(best_change[4]) <= 100:
                st.success(f"**Easiest single change:** {best_change[0].split(' ', 1)[1].lower()} — "
                           f"{best_change[5].lower()} from {money(best_change[2])} to {money(best_change[3])} "
                           f"({best_change[4]:+.1f}%).")
            else:
                st.info("**No realistic single change is enough here.** Any one change would have to be more than "
                        "double the current value, so several things (loan size, debt, credit history) "
                        "would need to improve together.")
        else:
            st.markdown("The application is approved. These are the **limits** before the model would change its mind (one thing at a time):")
            max_loan_hit = first_flip(model, applicant, "Loan_Amount", np.arange(loan + 500, 1_000_001, 500), target=0)
            min_credit_hit = first_flip(model, applicant, "Credit_Score", np.arange(credit - 1, 299, -1), target=0)
            max_debt_hit = first_flip(model, applicant, "Existing_Debt", np.arange(debt + 500, 1_000_001, 500), target=0)

            c1, c2, c3 = st.columns(3)
            with c1:
                with st.container(border=True):
                    st.markdown("**💸 Largest loan still approved**")
                    st.markdown(f"**{money(max_loan_hit - 500)}**" if max_loan_hit is not None else "Above 1,000,000")
                    st.caption(f"Current request: {money(loan)}")
            with c2:
                with st.container(border=True):
                    st.markdown("**🧾 Most debt still approved**")
                    st.markdown(f"**{money(max_debt_hit - 500)}**" if max_debt_hit is not None else "Above 1,000,000")
                    st.caption(f"Current debt: {money(debt)}")
            with c3:
                with st.container(border=True):
                    st.markdown("**⭐ Lowest credit score still approved**")
                    st.markdown(f"**{int(min_credit_hit) + 1}**" if min_credit_hit is not None else "Any score (300+)")
                    st.caption(f"Current score: {credit}")

        st.caption("Note: these are the model's estimates, based on patterns in the training data. "
                   "They show what the model reacts to, not guaranteed real-world causes.")
    else:
        st.info("👆 Fill in the form and click **Predict loan decision**, or load one of the example applicants above.")

# ============================================================================
# Tab 2 - About the model
# ============================================================================
with tab_about:
    st.markdown("### How good is the model?")
    st.markdown(f"Scores on the **{info['n_test']} test applications** that the model never saw during training.")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Accuracy", f"{info['accuracy']:.1%}", help="Share of all decisions that were correct.")
    a2.metric("Precision", f"{info['precision']:.1%}", help="Of the applicants it approved, how many deserved it.")
    a3.metric("Recall", f"{info['recall']:.1%}", help="Of the applicants who deserved approval, how many it approved.")
    a4.metric("F1-score", f"{info['f1']:.3f}", help="Balance of precision and recall.")

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.markdown("#### Confusion matrix (test set)")
        cm = info["confusion"]
        st.markdown("\n".join([
            "| | Predicted rejected | Predicted approved |",
            "|---|---|---|",
            f"| **Actually rejected** | {cm[0][0]} ✅ | {cm[0][1]} ⚠️ |",
            f"| **Actually approved** | {cm[1][0]} ⚠️ | {cm[1][1]} ✅ |",
        ]))
        st.markdown(
            f"- **{cm[0][1]} risky applicants were approved** (false positives) — the costly mistake for a bank.\n"
            f"- **{cm[1][0]} good applicants were rejected** (false negatives)."
        )
        st.markdown("#### How it was built")
        st.markdown(
            f"""
            1. **Data:** {info['n_rows']:,} loan applications after cleaning (duplicates, messy text, impossible values fixed).
            2. **Split:** {info['n_train']} for training and {info['n_test']} for testing, keeping {info['approval_rate']:.1%} approved in both.
            3. **Features:** 12 inputs + 2 engineered ratios (debt-to-income, loan-to-income).
            4. **Pipeline:** missing-value filling, encoding and scaling learned from the training data only.
            5. **Model:** Logistic Regression, chosen for the best F1-score and precision among the models tested.
            """
        )
    with right:
        st.markdown("#### What the model pays most attention to")
        st.altair_chart(global_chart(model, info), use_container_width=True)
        st.caption("Size of the model's coefficients, grouped by input. Features are scaled, so sizes can be compared.")

    st.warning("**Limitations:** the model is trained on a synthetic dataset and a small test set (200 rows). "
               "It is a learning demo and must not be used for real lending decisions.")
