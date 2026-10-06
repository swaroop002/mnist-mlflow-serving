# MNIST digit classifier — CNN, MLflow registry, Flask serving

[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](train.ipynb)
[![MLflow](https://img.shields.io/badge/MLflow-model%20registry-0194E2)](https://mlflow.org)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

A small end-to-end MLOps exercise: train a convolutional network on MNIST,
track the run and register the model with MLflow, serve it from the registry,
and draw digits in the browser to test it.

```
train.ipynb ── mlflow.tensorflow.autolog() ──▶ MLflow tracking server (runs, metrics, registry)
                                                        │  models:/digit-classifier-cnn/1
                                                        ▼
                                            predict_api.py  (Flask, :5001, CORS)
                                                        ▲
                                                        │  POST /predict  {"image": [784 floats]}
                                            index.html  (draw on a canvas, see the prediction)
```

## The model

Two `Conv2D` + `MaxPooling2D` blocks (32 and 64 filters), a 128-unit dense layer
with 30% dropout, softmax over 10 classes. Adam, sparse categorical
cross-entropy, 5 epochs, batch 64, 10% validation split. Everything the run
produces — parameters, per-epoch metrics, the test accuracy and the model
itself — is logged by MLflow autologging and the model is registered as
`digit-classifier-cnn`.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1. tracking server + model registry
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# 2. train, log and register (run the notebook; MNIST downloads itself)
jupyter lab train.ipynb

# 3. serve the registered model with CORS so a browser page can call it
python predict_api.py              # http://localhost:5001/predict, /health

# 4. draw digits
open index.html                    # or just double-click it
```

MLflow's own `mlflow models serve -m "models:/digit-classifier-cnn/1" -p 1234`
also works (`POST /invocations`, see the last notebook cells), but it sends no
CORS headers, so the browser page cannot call it. `predict_api.py` is the thin
layer that fixes that: it loads the model once from the registry and exposes
one endpoint.

## Why this layout

- **The registry is the source of truth.** The API never loads a file path;
  it asks MLflow for `models:/digit-classifier-cnn/1`, so promoting a new
  version is a registry change, not a deploy.
- **Autologging over hand-written logging.** One line captures what would
  otherwise be a dozen `log_param`/`log_metric` calls, and the model with it.
- **Preprocessing lives in one place.** The canvas is downsampled to the same
  28×28, 0–1 greyscale the network was trained on, client-side, so the API
  stays a pure model call.

MIT licensed. Built by [Swaroop Rangle](https://github.com/swaroop002).
