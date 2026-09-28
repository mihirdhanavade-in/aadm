# Quick commerce analytics for Streamlit

This repository-ready folder is the deployed dashboard component of an Advanced Analytics for Decision Making project. It displays the supplied 2,000-order sample, a chronological model holdout, an ETA correction scenario, six recommendations and data-quality checks.

## Local run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app reads `data/orders.csv` and `data/summary.json` relative to its own file. These were exported from the companion analysis; there is no network dependency, secret, database or live order feed. The data file omits customer and order IDs.

## Streamlit Community Cloud

Create a new GitHub repository containing **the contents of this folder at the repository root**. In Community Cloud, choose that repository, its `main` branch, and `streamlit_app.py` as the entrypoint. The `requirements.txt` and `.streamlit/config.toml` files belong in the root and `.streamlit/` folder respectively. The site will be a static analysis exhibit over packaged data; updated source data requires a new analysis export and repository commit.

The report and Colab notebook explain model design, evaluation and limitations. In particular, the ETA scenario is an exploratory display; only the training-derived +5.03-minute correction is a valid preselected candidate for a controlled pilot.
