# ❤️ Heart Disease Risk Screening (Streamlit)

Streamlit app with three models trained on the CDC BRFSS *Personal Key Indicators of Heart Disease* data:

- Logistic Regression (`models/heart_logreg_model.pkl`)
- LightGBM (`models/heart_lgbm_model.pkl`)
- XGBoost (`models/heart_xgb_model.pkl`)

Pick one model in the sidebar, or choose **Compare all models** to see them side by side. A model whose file is missing is left out of the menu.

> Educational project on a self-reported survey. Not a medical device and not medical advice.

## Repository layout

```
heart-disease-app/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    ├── heart_logreg_model.pkl   <- copy from Colab
    ├── heart_lgbm_model.pkl     <- copy from Colab
    └── heart_xgb_model.pkl      <- copy from Colab
```

## Deploy on Streamlit Community Cloud

1. Download the `.pkl` files from Colab (each notebook's Step 12 downloads its file) and put them in `models/`.
2. **Match the library versions.** Open `requirements.txt` and make it equal to the versions printed in your Colab notebooks (Step 1 and the last line of Step 13, "Saved with versions"). The default file pins scikit-learn 1.6.1, pandas 2.2.3, numpy 2.1.3 and lightgbm 4.6.0. **XGBoost is not pinned**: replace the `xgboost>=2.1` line with the exact version printed in Step 1 of your XGBoost notebook (for example `xgboost==3.0.5`).
3. Create a new **public GitHub repository** and upload everything, including the `models/` folder.
4. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**, choose your repository and branch (`main`), set **Main file path** to `app.py`.
5. Under **Advanced settings**, choose Python 3.12 or 3.13, then click **Deploy**. The first build takes a few minutes.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Troubleshooting

| Problem | Fix |
|---|---|
| "No model files found" | The `.pkl` files must be in `models/` (or the repo root) with exactly these names |
| "Could not load ..." or wrong results | Library versions in `requirements.txt` differ from training; pin the same versions |
| A model is missing from the menu | Its `.pkl` is missing or failed to load; see the sidebar message |
| Yellow version warning in the sidebar | The installed library version differs from the one used in training; pin the version shown in the warning |
| Build fails on `lightgbm` | Pick a different Python version in Advanced settings and redeploy |

## Security note

The app uses `pickle` to load the models. Only load `.pkl` files that you created yourself.
