# Sudoku Solver

A computer-vision pipeline that reads a Sudoku puzzle from a photograph and returns the completed board in under a second.

[![deploy to render](https://img.shields.io/badge/deploy%20to-render-430098?logo=render&logoColor=white)](https://sudokusolverrr.onrender.com)
[![license: mit](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

## Features

| | |
|---|---|
| **Grid detection** | OpenCV locates the grid, corrects perspective, and segments all 81 cells. |
| **Digit recognition** | A custom CNN trained on MNIST classifies each printed digit. |
| **Puzzle solving** | Backtracking solver with bitmask constraints and MRV heuristic — most puzzles solved in < 1 ms. |
| **Web UI** | Responsive single-page interface with no frontend frameworks or build tools. |
| **Docker ready** | Fully containerized; system-level OpenCV dependencies are handled automatically. |

## How It Works

1. **Preprocessing** — OpenCV converts the image to grayscale, applies Gaussian blur, and uses adaptive thresholding to isolate grid lines.
2. **Grid extraction** — Contour detection finds the largest quadrilateral (the board). A perspective transform flattens it into a canonical top-down view.
3. **Digit recognition** — The board is sliced into 81 cells. Non-empty cells are fed through a CNN that returns a digit in the range 1–9.
4. **Solving** — The parsed matrix is solved in place by a backtracking algorithm accelerated with bitmask constraint propagation and the Minimum Remaining Values (MRV) heuristic.
5. **Response** — The completed board is returned as JSON and rendered on the client.

## Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | Python 3.12, FastAPI, Uvicorn, Gunicorn |
| **AI / Vision** | TensorFlow 2, Keras, OpenCV 4, NumPy |
| **Frontend** | HTML5, Vanilla CSS, Vanilla JavaScript |
| **Deployment** | Docker (python:3.12-slim / Debian 12 Bookworm) |

## Project Structure

```text
sudoku/
├── api/
│   └── main.py                    # FastAPI app — routes and static file serving
├── core/
│   ├── digit_predictor.py         # CNN inference wrapper
│   ├── image_processing.py        # OpenCV grid detection and cell extraction
│   └── solver.py                  # Backtracking solver with bitmask + MRV
├── models/
│   └── digit_model.keras          # Pre-trained CNN weights
├── notebooks/
│   └── train_digit_model.ipynb    # Model training notebook
├── assets/                        # Sample images for testing
├── tests/
│   ├── test_image.py              # Image processing tests
│   ├── test_predictor.py          # Digit prediction tests
│   └── test_solver.py             # Solver unit tests
├── static/
│   ├── css/style.css              # UI styling
│   ├── img/logo.svg               # Site logo
│   └── js/app.js                  # Client-side interaction logic
├── templates/
│   └── index.html                 # Jinja2 HTML template
├── Dockerfile                     # Production container definition
└── requirements.txt               # Pinned Python dependencies
```

## Running Locally

**Prerequisites:** Python 3.10+ · Git

```bash
# 1. Clone and enter the repository
git clone https://github.com/roseliaevanny/sudoku.git
cd sudoku

# 2. Create and activate a virtual environment
python -m venv env
env\Scripts\activate        # Windows
# source env/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the development server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in your browser.

## Deployment

The project ships with a `Dockerfile` that installs the system libraries required by OpenCV (`libgl1`, `libglib2.0-0`) before installing Python dependencies. No additional configuration is needed.

**Deploy to Render (or any Docker-compatible host):**

1. Push the repository to GitHub.
2. Create a new **Web Service** and connect the repository.
3. Set the **Runtime** to **Docker**.
4. Click **Deploy** — Render reads the `Dockerfile` automatically. The `PORT` environment variable is injected at runtime; no manual configuration is required.