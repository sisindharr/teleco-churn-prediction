# Deployment guide

## Option A: Streamlit Community Cloud (recommended, free)

1. **Train locally** so the app has something to load:
   ```bash
   python -m src.train
   ```
2. **Commit the artefacts** (they are *not* git-ignored on purpose):
   ```bash
   git add models/ data/processed/ .
   git commit -m "Add trained model and processed data"
   ```
3. **Push to GitHub:**
   ```bash
   git remote add origin https://github.com/<you>/telco-churn-prediction.git
   git branch -M main
   git push -u origin main
   ```
4. Go to **https://share.streamlit.io**, sign in with GitHub and choose **Create app**.
5. Fill in: Repository `<you>/telco-churn-prediction`, Branch `main`, Main file path `app/main.py`.
6. Open **Advanced settings** and select Python **3.11** or **3.12**.
7. Click **Deploy**. The first build installs `requirements.txt` (a few minutes). Add the live URL to the README.

### Keep the model and library versions in sync
A pickled scikit-learn model must be loaded with a compatible scikit-learn version. If the cloud installs a different version than you trained with, retrain and commit, or pin versions:
```bash
pip freeze | grep -i -E "^(scikit-learn|xgboost|pandas|numpy|streamlit|plotly|joblib)==" > requirements.lock.txt
```
Then copy those pins into `requirements.txt`. The app shows a warning in the sidebar when versions differ.

### Updating the app
Retrain, commit the new `models/` and `data/processed/`, push. Streamlit Cloud redeploys automatically.

### Troubleshooting
| Symptom | Fix |
|---|---|
| "Almost there" screen | `models/` or `data/processed/` is missing from the repo. Train and commit them. |
| `ModuleNotFoundError: src` | Main file path must be `app/main.py` from the repo root. |
| Version warning in sidebar | Retrain with the installed versions, or pin versions as above. |
| Fonts look different | Google Fonts blocked by the network. The app falls back to system fonts. |

## Option B: Docker

```bash
docker build -t churniq .
docker run -p 8501:8501 churniq
# open http://localhost:8501
```
Train before building so `models/` and `data/processed/` are copied into the image (or mount them with `-v $(pwd)/models:/app/models`).

Deploy the image to any container platform (Cloud Run, Azure Container Apps, AWS App Runner, Fly.io, Render). Expose port 8501.

## Option C: GitHub Actions (CI)

`.github/workflows/ci.yml` runs on every push and pull request:
1. installs dependencies on Python 3.11 and 3.12,
2. compiles all code,
3. runs `pytest` (trains a tiny model on synthetic data, so no Kaggle download is needed),
4. builds the Docker image.

To add continuous deployment later: Streamlit Cloud redeploys automatically on push to `main`, so CI plus branch protection (require green checks before merge) gives you a safe release flow.

## Production checklist

- [ ] Pin dependency versions
- [ ] Add authentication if real customer data is used (never commit real customer data to GitHub)
- [ ] Log predictions to a database instead of session memory
- [ ] Monitor data drift and model performance, and schedule retraining
- [ ] Calibrate probabilities before using them for financial decisions
