"""
Sri Lankan Used-Car Price - Ensemble Learning Dashboard
Author: Serasinghe S T H C H (224179H)

Pages
  Overview | Model comparison | Feature importance | Bagging lab | Live prediction
All numbers come from the files created by the notebook
(results/*.csv, models/*.pkl, models/metadata.json).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ensemble_utils as eu  # needed so the pickled pipelines can be loaded

ROOT = Path(__file__).parent
FAMILY_COLORS = {"Baseline": "#8c8c8c", "Bagging": "#4c72b0", "Boosting": "#dd8452",
                 "Stacking": "#55a868", "Hybrid": "#c44e52", "Other": "#937860"}

st.set_page_config(page_title="SL Car Price - Ensemble Lab", page_icon="🚗", layout="wide")

# Make the main action button easy to see (bright green, white bold text, clear hover state)
st.markdown(
    """
    <style>
    button[kind="primary"], [data-testid="stBaseButton-primary"] {
        background-color: #16a34a !important;
        border: 2px solid #86efac !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0.6rem 2rem !important;
        box-shadow: 0 0 14px rgba(34, 197, 94, 0.55);
    }
    button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
        background-color: #22c55e !important;
        border-color: #ffffff !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data loading
@st.cache_data
def load_tables():
    r = ROOT / "results"
    return {
        "cmp": pd.read_csv(r / "model_comparison.csv"),
        "seg": pd.read_csv(r / "error_by_segment.csv"),
        "pred": pd.read_csv(r / "test_predictions.csv"),
        "perm": pd.read_csv(r / "permutation_importance.csv"),
        "rfimp": pd.read_csv(r / "rf_impurity_importance.csv"),
        "oob": pd.read_csv(r / "bagging_oob_vs_trees.csv"),
        "rho": pd.read_csv(r / "bagging_rho_vs_max_features.csv"),
    }


@st.cache_data
def load_meta():
    return json.load(open(ROOT / "models" / "metadata.json"))


@st.cache_resource(show_spinner="Loading model...")
def load_model(file):
    return joblib.load(ROOT / "models" / file)


T, META = load_tables(), load_meta()
cmp_df = T["cmp"]
fmt_m = lambda lakhs: f"Rs. {lakhs / 10:,.2f} M"  # lakhs -> millions of rupees

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🚗 SL Car Price Lab")
st.sidebar.caption("Ensemble learning on 9.5k Sri Lankan used-car adverts (ikman.lk + riyasewana.com)")
PAGES = ["Overview", "Model comparison", "Feature importance", "Bagging lab", "Live prediction"]
_start = st.query_params.get("page", "Overview")          # lets me link straight to a page, e.g. ?page=Bagging%20lab
page = st.sidebar.radio("Go to", PAGES, index=PAGES.index(_start) if _start in PAGES else 0)
st.sidebar.markdown("---")
st.sidebar.markdown("**Serasinghe S T H C H**  \nIndex 224179H  \nMachine Learning - Ensemble Project")

# ---------------------------------------------------------------- pages
if page == "Overview":
    st.title("Predicting used-car prices in Sri Lanka with ensemble learning")
    st.write(
        "I compare **Bagging** (Random Forest), **Boosting** (XGBoost, LightGBM, CatBoost), **Stacking** and a "
        "**hybrid Random Forest + ANN** on real Sri Lankan advert data. Use the sidebar to explore the results "
        "or try a live prediction.")
    best = cmp_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Adverts used", f"{META['n_rows']:,}")
    c2.metric("Models compared", len(cmp_df))
    c3.metric("Best model (RMSLE)", best["Model"].split(" (")[0])
    c4.metric("Best typical error (MAPE)", f"{cmp_df['MAPE_%'].min():.1f} %")

    st.subheader("Bagging, boosting and stacking")
    a, b, c = st.columns(3)
    a.markdown("#### Bagging\nMany deep trees on bootstrap samples, **averaged**. Cuts *variance*.  \n`Random Forest`")
    b.markdown("#### Boosting\nShallow trees built **one after another**, each fixing the last one's errors. Cuts *bias*.  \n`XGBoost · LightGBM · CatBoost`")
    c.markdown("#### Stacking\nA meta-learner **learns how to weight** very different models (RF, boosters, SVR, ANN).  \n`Ridge on top`")
    st.info("Prices are shown in **Rs. millions**. In the notebook they are in lakhs (1 lakh = Rs. 100,000).")
    st.caption(f"Dataset: {META['price_unit']}; listings scraped Dec-2024 to Feb-2025.")

elif page == "Model comparison":
    st.title("Model comparison on the held-out test set")
    metric = st.selectbox("Metric", ["RMSLE", "MAE", "RMSE", "R2", "MAPE_%"], index=0,
                          help="RMSLE is the main tuning metric: it measures *relative* price error.")
    asc = metric != "R2"
    d = cmp_df.sort_values(metric, ascending=asc)
    fig = px.bar(d, x=metric, y="Model", color="Family", orientation="h", color_discrete_map=FAMILY_COLORS,
                 text=d[metric].round(3))
    fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": d["Model"][::-1]}, height=520, margin=dict(l=0, r=0, t=10, b=0))
    if metric == "R2":
        fig.update_xaxes(range=[max(0.0, d[metric].min() - .02), d[metric].max() + .01])
    st.plotly_chart(fig, width="stretch")
    st.caption("MAE / RMSE are in lakhs of rupees. RMSE 95% CI comes from 1000 bootstrap resamples of the test set.")
    st.dataframe(cmp_df.round(3), width="stretch", hide_index=True)

    st.subheader("Where do the models fail? Error by price segment")
    seg = T["seg"]
    seg_metric = st.radio("Segment metric", ["MAPE_%", "MAE"], horizontal=True)
    default_pick = [m for m in ["Decision Tree", "Random Forest", "XGBoost", "CatBoost", "Stacking (RF+XGB+Cat+SVR+ANN)", META["best_hybrid"]] if m in cmp_df["Model"].values]
    pick = st.multiselect("Models", cmp_df["Model"].tolist(), default=default_pick)
    order = ["< Rs 2.5M", "2.5 - 5M", "5 - 10M", "> Rs 10M"]
    sp = seg[seg.Model.isin(pick)].pivot(index="Model", columns="Segment", values=seg_metric)[order]
    st.plotly_chart(px.imshow(sp, text_auto=".1f", color_continuous_scale="YlOrRd", aspect="auto"), width="stretch")

    st.subheader("Predicted vs actual")
    m = st.selectbox("Model", cmp_df["Model"].tolist())
    pr = T["pred"]
    fig = px.scatter(pr, x="actual", y=m, log_x=True, log_y=True, opacity=.4, labels={"actual": "actual price (lakhs)", m: "predicted (lakhs)"})
    fig.add_trace(go.Scatter(x=[2, 800], y=[2, 800], mode="lines", line=dict(color="red", dash="dash"), name="perfect"))
    st.plotly_chart(fig, width="stretch")

elif page == "Feature importance":
    st.title("What drives the price?")
    tab1, tab2 = st.tabs(["Permutation importance (all models)", "Random Forest impurity importance"])
    with tab1:
        perm = T["perm"].set_index("feature")
        models = st.multiselect("Models", perm.columns.tolist(), default=perm.columns.tolist()[:4])
        long = perm[models].reset_index().melt(id_vars="feature", var_name="Model", value_name="Increase in RMSLE")
        order = perm[models].mean(axis=1).sort_values().index.tolist()
        fig = px.bar(long, y="feature", x="Increase in RMSLE", color="Model", barmode="group", orientation="h",
                     category_orders={"feature": order}, height=560)
        st.plotly_chart(fig, width="stretch")
        st.caption("How much the test error grows when one column is shuffled. Bigger = the model relies on it more.")
    with tab2:
        d = T["rfimp"].sort_values("importance")
        st.plotly_chart(px.bar(d, x="importance", y="feature", orientation="h", height=520), width="stretch")
        st.caption("Impurity importance, with one-hot columns added back together. It is biased towards high-cardinality features, so I cross-check it with permutation importance.")

elif page == "Bagging lab":
    st.title("Bagging lab")
    st.latex(r"\mathrm{Var}(\bar f)=\rho\,\sigma^{2}+\frac{1-\rho}{B}\,\sigma^{2}")
    st.write("Two things lower the variance of a forest, **more trees B** and **less correlation ρ** between the trees.")
    c1, c2 = st.columns(2)
    o = T["oob"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=o.B, y=o.Test, name="test error", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=o.B, y=o.OOB, name="out-of-bag error", mode="lines+markers", line=dict(dash="dash")))
    fig.update_layout(title="Number of trees (B)", xaxis_type="log", xaxis=dict(type="log", title="number of trees", tickvals=[1, 3, 10, 30, 100, 500], ticktext=["1", "3", "10", "30", "100", "500"]),
                      yaxis_title="RMSLE (log price)", height=420, legend=dict(orientation="h", y=-0.28), margin=dict(t=50))
    c1.plotly_chart(fig, width="stretch")
    r = T["rho"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.max_features, y=r.rho, name="rho", mode="lines+markers", line=dict(color="#c44e52")))
    fig.add_trace(go.Scatter(x=r.max_features, y=r["Test RMSLE"], name="test RMSLE", mode="lines+markers", yaxis="y2", line=dict(color="#4c72b0")))
    fig.update_layout(title="Features per split (max_features)", xaxis_title="max_features (fraction tried per split)", yaxis=dict(title="estimated ρ (tree correlation)"),
                      yaxis2=dict(title="test RMSLE", overlaying="y", side="right"), height=420, legend=dict(orientation="h", y=-0.28), margin=dict(t=50))
    c2.plotly_chart(fig, width="stretch")
    st.caption("The out-of-bag error is computed from the rows each tree did NOT see in its bootstrap sample, so it tracks the test error without needing a test set.")
    st.dataframe(r.round(4), hide_index=True, width="stretch")

else:  # Live prediction
    st.title("Live price prediction")
    st.write("Describe a car and every selected model estimates its price. If the models disagree, the price is less certain.")
    brands = sorted(META["brand_models"])
    with st.container():
        c1, c2, c3 = st.columns(3)
        brand = c1.selectbox("Brand", brands, index=brands.index("TOYOTA") if "TOYOTA" in brands else 0)
        model_opts = META["brand_models"][brand]
        model = c2.selectbox("Model", model_opts, index=model_opts.index("VITZ") if "VITZ" in model_opts else 0)
        yom = c3.slider("Year of manufacture", META["yom_range"][0], META["yom_range"][1], 2015)
        c1, c2, c3 = st.columns(3)
        engine = c1.number_input("Engine capacity (cc)", META["engine_range"][0], META["engine_range"][1],
                                 int(META["median_engine_by_model"].get(model, 1300)), step=50, key=f"cc_{brand}_{model}")
        gear = c2.selectbox("Gearbox", ["Automatic", "Manual"])
        fuel = c3.selectbox("Fuel", ["Petrol", "Hybrid", "Diesel", "Electric"])
        c1, c2, c3 = st.columns(3)
        town = c1.selectbox("Town", META["towns"], index=META["towns"].index("Colombo") if "Colombo" in META["towns"] else 0)
        condition = c2.selectbox("Condition", ["USED", "NEW"])
        leasing = c3.selectbox("Leasing", ["No Leasing", "Ongoing Lease"])
        e1, e2, e3, e4 = st.columns(4)
        ac, ps = e1.checkbox("Air conditioning", True), e2.checkbox("Power steering", True)
        pm, pw = e3.checkbox("Power mirrors", True), e4.checkbox("Power windows", True)
        chosen = st.multiselect("Models to use", list(META["model_files"]),
                                default=[m for m in META["model_files"] if m.split(" (")[0] in ("Random Forest", "XGBoost", "CatBoost") or m.startswith("Hybrid")])
        go_btn = st.button("Predict price", type="primary")

    if go_btn and chosen:
        flag = lambda b: "Available" if b else "Not_Available"
        row = pd.DataFrame([dict(Brand=brand, Model=model, YOM=yom, Engine_cc=float(engine), Gear=gear, Fuel=fuel, Town=town,
                                 Leasing=leasing, Condition=condition, AC=flag(ac), PowerSteering=flag(ps),
                                 PowerMirror=flag(pm), PowerWindow=flag(pw))])
        preds = {m: float(load_model(META["model_files"][m]).predict(row)[0]) for m in chosen}
        vals = np.array(list(preds.values()))
        med = float(np.median(vals))
        c1, c2, c3 = st.columns(3)
        c1.metric("Consensus (median of models)", fmt_m(med))
        c2.metric("Lowest model", fmt_m(vals.min()))
        c3.metric("Highest model", fmt_m(vals.max()))
        spread = (vals.max() - vals.min()) / med * 100
        (st.success if spread < 15 else st.warning)(f"Models agree within {spread:.0f}% of the consensus price." if spread < 15 else
                                                    f"Models disagree by {spread:.0f}% of the consensus price - treat this estimate with care (rare or unusual car?).")
        d = pd.DataFrame({"Model": list(preds), "Predicted (Rs. M)": [v / 10 for v in preds.values()]})
        st.plotly_chart(px.bar(d, x="Predicted (Rs. M)", y="Model", orientation="h", text=d["Predicted (Rs. M)"].round(2), height=60 + 55 * len(d)),
                        width="stretch")

        if "Random Forest" in chosen:  # bagging gives a free uncertainty band: look at the individual trees
            rf = load_model(META["model_files"]["Random Forest"]).regressor_
            Xt = rf[:-1].transform(row).astype(np.float32)
            trees = np.expm1(np.array([t.predict(Xt)[0] for t in rf.named_steps["model"].estimators_])) / 10
            lo, hi = np.percentile(trees, [10, 90])
            st.markdown(f"**Random Forest tree spread:** 80% of the {len(trees)} individual trees predict between "
                        f"**Rs. {lo:.2f} M** and **Rs. {hi:.2f} M**.")
            fig = px.histogram(x=trees, nbins=30, labels={"x": "price predicted by one tree (Rs. M)"}, height=300)
            fig.add_vline(x=preds["Random Forest"] / 10, line_color="red", annotation_text="forest prediction")
            st.plotly_chart(fig, width="stretch")
    elif go_btn:
        st.warning("Select at least one model.")


