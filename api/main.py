import copy
import logging
import os
import sys
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import extract_digit_cells, solve_sudoku
from core.digit_predictor import DigitPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_HERE          = os.path.dirname(os.path.abspath(__file__))
_BASE          = os.path.join(_HERE, "..")
_STATIC_DIR    = os.path.normpath(os.path.join(_BASE, "static"))
_TEMPLATES_DIR = os.path.normpath(os.path.join(_BASE, "templates"))

# Load the CNN model once at startup
_predictor: DigitPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    logger.info("Loading digit recognition model…")
    _predictor = DigitPredictor()
    logger.info("Model ready.")
    yield
    _predictor = None


# App
app = FastAPI(
    title="Sudoku AI Solver",
    description="Upload a photo of a Sudoku puzzle and get the solved grid back.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static files at /static
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

templates = Jinja2Templates(directory=_TEMPLATES_DIR)


# Frontend
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# API endpoints
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/solve")
async def solve(file: UploadFile = File(...)):
    """
    Accepts a Sudoku image (JPEG / PNG / any OpenCV-readable format).
    Returns the detected initial board and the solved board.
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    # 1. Decode image bytes → OpenCV BGR array
    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=422,
            detail="Could not decode the uploaded file as an image."
        )

    # 2. Extract 81 cell images via the CV pipeline
    try:
        cells = extract_digit_cells(img)
    except Exception as exc:
        logger.exception("Grid detection failed")
        raise HTTPException(
            status_code=422,
            detail=f"Grid detection failed: {exc}"
        )

    clue_count = sum(1 for c in cells if c is not None)
    warning = None
    if clue_count < 17:
        warning = (
            f"Only {clue_count} digit(s) detected — the mathematical minimum "
            "for a uniquely solvable puzzle is 17. The grid may not have been "
            "read correctly."
        )

    # 3. Predict digits with the CNN
    try:
        initial_board = _predictor.predict_cells(cells)
    except Exception as exc:
        logger.exception("Digit prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Digit prediction failed: {exc}"
        )

    # 4. Solve
    solved_board = copy.deepcopy(initial_board)
    solved = solve_sudoku(solved_board)
    if not solved:
        solved_board = initial_board

    return {
        "clues_found": clue_count,
        "initial_board": initial_board,
        "solved_board": solved_board,
        "solved": solved,
        "warning": warning,
    }
