# Industrial Pipe Roundness Inspector

Streamlit app for pipe roundness inspection, pixel measurement, optional millimeter scaling, and OSTB standards checks.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

- Main file path: `app.py`
- Dependencies: `requirements.txt`
- Python version: choose it in Streamlit Community Cloud `Advanced settings`
- Recommended Python version for this app: `3.12`

`runtime.txt` is not used by Streamlit Community Cloud for Python selection.
If the app was deployed with the wrong Python version, delete the app and redeploy it with the correct version selected in the UI.

Do not commit local virtual environments, cache files, test images, or generated scale files. They are ignored by `.gitignore`.

# pipe_ostp
