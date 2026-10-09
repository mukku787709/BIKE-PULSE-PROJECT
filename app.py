from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from bikepulse.insights import answer

st.set_page_config(page_title="BikePulse AI",page_icon="🚲",layout="wide")
st.title("🚲 BikePulse AI")
st.caption("Online data → wrangling → machine learning → deep learning → actionable insights")
base=Path(__file__).parent/"artifacts"
if not (base/"metrics.json").exists():
    st.info("Train the models first: python -m bikepulse.train")
    st.stop()
report=json.loads((base/"metrics.json").read_text())
pred=pd.read_csv(base/"predictions.csv",parse_dates=["timestamp"])
a,b,c,d=st.columns(4)
a.metric("Online observations",f"{report['rows']:,}")
b.metric("Selected model",report["winner"])
c.metric("Test MAE",f"{report['test'][report['winner']]['MAE']:.1f} rentals")
d.metric("Interval coverage",f"{report['interval']['observed_test_coverage']:.1%}")
forecast,models,insights,data=st.tabs(["Demand explorer","Model laboratory","AI insights","Data & methodology"])
with forecast:
    days=st.slider("Show last test days",1,110,14)
    subset=pred[pred.timestamp>=pred.timestamp.max()-pd.Timedelta(days=int(days))]
    st.plotly_chart(px.line(subset,x="timestamp",y=["actual","prediction","lower","upper"],title="Observed demand and calibrated prediction interval"),width="stretch")
    st.dataframe(subset.tail(24),hide_index=True,width="stretch")
    st.download_button("Download predictions",pred.to_csv(index=False),"predictions.csv","text/csv")
with models:
    st.write("Selection uses validation MAE; test results below are for final evaluation.")
    st.dataframe(pd.DataFrame(report["test"]).T, width="stretch")
    st.plotly_chart(px.bar(pd.read_csv(base/"importance.csv"),x="importance",y="feature",orientation="h",title="Random forest feature importance (not causal effects)"),width="stretch")
    st.plotly_chart(px.line(pd.read_csv(base/"training_history.csv"),x="epoch",y="validation_mae",title="Neural network training"),width="stretch")
with insights:
    st.subheader("Ask your experiment")
    st.caption("Local intent-based assistant; no API key needed. Answers are grounded in this experiment.")
    question=st.text_input("Question",placeholder="Which model performs best?")
    if question:
        st.write(answer(question,report,pred))
    st.subheader("Unusual conditions")
    st.dataframe(pred[pred.anomaly],hide_index=True)
with data:
    st.markdown("Data: [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275), Fanaee-T (2013), CC BY 4.0. Historical hourly observations from 2011–2012; these are not live forecasts.")
    st.json(report["splits"])
    st.write("Chronological 60% training / 15% validation / 10% calibration / 15% test. Imputation and scaling fit on training only. Casual and registered counts are excluded to prevent target leakage. Prediction intervals use held-out residual calibration; exchangeability is not guaranteed for temporal data.")
