# CommissionLens

Commission-adjusted alpha prediction for Indian equity mutual funds.

Most retail investors in India buy the *regular* plan of a mutual fund, which carries an extra
0.5 to 1.5 percent of annual distributor commission versus the *direct* plan of the identical
portfolio. Over a multi-year SIP that gap compounds into a real drag. CommissionLens answers one
question for every fund, every quarter:

> Will this fund generate enough alpha next quarter to justify the commission you pay for its
> regular plan?

It is framed as a joint learning problem: predict next-quarter **net alpha** (CAPM alpha
annualised, minus the regular-vs-direct expense gap) as regression, and the binary
**commission-justified** label (net alpha above zero) as classification.

## The model

Fund data is a panel of funds observed over consecutive quarters with a forward-looking label,
so instead of the usual gradient-boosted trees the estimator is **CommissionNet**, a PyTorch
multi-task network:

- a bidirectional GRU reads the last eight quarters of fund and macro features,
- an attention layer pools the timesteps into one fund-quarter embedding,
- two heads share that embedding to predict net alpha (Smooth L1) and the commission-justified
  label (weighted BCE) at once.

SHAP `GradientExplainer` ranks the features that drive the classification.

## Files

```
config.py       configuration dataclasses and yaml loader
config.yaml      run settings
data.py         synthetic fund-panel generator, mfapi.in client, macro loader
features.py     risk metrics and the quarterly panel builder
model.py        sequences, CommissionNet, training loop, evaluation
explain.py      SHAP feature importance
simulation.py   XIRR solver and SIP back-validation
main.py         one CLI: build / train / explain / simulate / all
app.py          Streamlit dashboard
notebook.ipynb  the full workflow as a notebook
test_commissionlens.py   unit tests
docs/           research report, problem statement, resource pool
```

## Setup

Python is invoked as `py` on this machine.

```
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
```

## Run

```
py main.py build       # generate the panel -> data/panel.parquet
py main.py train       # train CommissionNet -> artifacts/commissionnet.pt, metrics.json
py main.py explain     # SHAP importance   -> artifacts/shap_importance.{csv,png}
py main.py simulate    # SIP back-test     -> artifacts/simulation.json, sip_xirr.png
py main.py all         # everything in order

py -m streamlit run app.py   # optional dashboard
py -m pytest                 # tests (run without torch installed)
```

Outputs land in `data/` and `artifacts/`.

## Data

The pipeline runs fully offline through a synthetic generator (`data.py`) that produces 220
Indian equity funds over 2013-2023 with realistic NAV paths, expense ratios, AUM, manager
tenure, turnover, and a macro block (repo rate, CPI, yield-curve slope, FII and DII flows). Each
fund carries a latent, regime-sensitive skill so the problem has real signal without being
trivial. For live data set `data.source: mfapi` in the config and wire `mfapi_fetch_many` and
`load_macro_csv` from `data.py`.

## Evaluation

- Regression: RMSE, MAE, R^2. Classification: AUC-ROC, F1, precision at the top decile.
- Splits are strictly temporal. The test set is the most recent quarters, and the SIP back-test
  trains CommissionNet only on quarters that close before the simulation window, so no future
  information leaks backwards.

## Notes

- The default dataset is synthetic; the numbers show the pipeline works end to end, they are not
  claims about any real fund.
- Net alpha annualises a single quarter of return with beta fixed at the trailing estimate; this
  simplification is documented in `docs/report.md`.
