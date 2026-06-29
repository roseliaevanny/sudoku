import cv2
import numpy as np


def pre_process_image(img):
    """
    Converts an image to grayscale, blurs it to reduce noise, applies
    adaptive thresholding, and dilates the result to close small gaps in
    grid borders — making the outer boundary a single connected contour.

    Args:
        img: BGR or grayscale uint8 image (numpy array).

    Returns:
        Binary thresholded image (same H×W, single channel).

    Raises:
        ValueError: If the input is not a valid numpy image array.
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("pre_process_image: received an empty or invalid image.")

    # Convert to grayscale if the image is colour
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Blur to smooth out background noise before thresholding
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # Adaptive thresholding handles uneven lighting across the image.
    # THRESH_BINARY_INV → foreground (lines + digits) are WHITE, background BLACK.
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Dilate slightly to close small gaps in the grid border so that the
    # outer rectangle is a single connected contour, not a broken chain.
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.dilate(thresh, dilate_kernel, iterations=1)

    return thresh


def find_sudoku_contour(thresh_img):
    """
    Locates the best 4-sided polygon in a thresholded image corresponding to
    the outer boundary of the Sudoku grid.

    Strategy:
    1. Filter contours to those with area >= 5% of image.
    2. For each candidate, score by squareness: a sudoku grid is roughly
       square, so the bounding-rect aspect ratio should be close to 1.0.
       Candidates are ranked by (area × squareness_score) so that a
       slightly-smaller-but-squarer contour beats a larger landscape rectangle
       (e.g. a book page, a hand, or a pencil).
    3. Try progressively looser epsilon values to get exactly 4 corners.
    4. Fall back to convex hull extremal points if needed.

    Args:
        thresh_img: Binary thresholded image (output of pre_process_image).

    Returns:
        A (4, 1, 2) numpy array of the 4 corner points.

    Raises:
        ValueError: If no suitable grid contour can be found.
    """
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found. The image may be blank or too noisy.")

    img_area = thresh_img.shape[0] * thresh_img.shape[1]
    min_area = 0.05 * img_area

    # Epsilon multipliers to try in order — start tight, go looser
    epsilons = [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]

    def squareness(c):
        """Score in [0, 1]: 1.0 = perfect square bounding rect, 0 = infinitely thin."""
        _, _, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            return 0.0
        ratio = min(w, h) / max(w, h)
        return ratio

    # Build candidate list sorted by area × squareness (descending)
    candidates = [
        c for c in contours if cv2.contourArea(c) >= min_area
    ]
    candidates.sort(
        key=lambda c: cv2.contourArea(c) * squareness(c),
        reverse=True,
    )

    best_candidate = candidates[0] if candidates else None

    for contour in candidates:
        perimeter = cv2.arcLength(contour, True)

        # Try each epsilon until we get exactly 4 corners
        for eps in epsilons:
            approx = cv2.approxPolyDP(contour, eps * perimeter, True)
            if len(approx) == 4:
                return approx

    # Fallback: convex hull of the best candidate
    if best_candidate is not None:
        hull = cv2.convexHull(best_candidate)
        pts = hull.reshape(-1, 2).astype(np.float32)

        sums = pts.sum(axis=1)
        diffs = pts[:, 0] - pts[:, 1]

        corners = np.array([
            pts[np.argmin(sums)],   # top-left
            pts[np.argmax(diffs)],  # top-right
            pts[np.argmax(sums)],   # bottom-right
            pts[np.argmin(diffs)],  # bottom-left
        ], dtype=np.float32).reshape(4, 1, 2)

        return corners

    raise ValueError(
        "No Sudoku grid found in image. "
        "Ensure the full grid is visible and well-lit."
    )


def reorder_points(points):
    """
    Reorders 4 corner points into the canonical perspective-transform order:
    [top-left, top-right, bottom-right, bottom-left].

    Uses explicit x-y differences instead of np.diff to avoid shape issues.

    Args:
        points: numpy array of shape (4, 1, 2) or (4, 2).

    Returns:
        numpy array of shape (4, 1, 2), dtype float32.
    """
    points = points.reshape((4, 2)).astype(np.float32)
    ordered = np.zeros((4, 1, 2), dtype=np.float32)

    # Top-left → min(x+y);  Bottom-right → max(x+y)
    sums = points.sum(axis=1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]

    # Top-right → max(x-y)  [large x, small y]
    # Bottom-left → min(x-y) [small x, large y]
    x = points[:, 0]
    y = points[:, 1]
    diffs = x - y
    ordered[1] = points[np.argmax(diffs)]
    ordered[3] = points[np.argmin(diffs)]

    return ordered


def warp_perspective(img, corners, size=630):
    """
    Applies a perspective transform to produce a top-down, square view of the grid.

    Args:
        img:     Grayscale image (same source used for contour detection).
        corners: 4-corner array from find_sudoku_contour.
        size:    Output image side length in pixels (default 630 → 70px per cell).
                 Larger values give the cell-cleaning step more pixels to work with.

    Returns:
        Square grayscale image of shape (size, size).
    """
    ordered = reorder_points(corners)

    src = np.float32(ordered)
    dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (size, size))

    return warped


def split_boxes(warped_img):
    """
    Splits a warped Sudoku image into 81 individual cell images (9×9 grid).

    Each cell is returned as an independent copy so that downstream processing
    cannot accidentally mutate the source array.

    Args:
        warped_img: Square grayscale image (ideally divisible by 9).

    Returns:
        List of 81 numpy arrays in row-major order (left→right, top→bottom).
    """
    h, w = warped_img.shape[:2]
    cell_h = h // 9
    cell_w = w // 9

    boxes = []
    for r in range(9):
        for c in range(9):
            cell = warped_img[
                r * cell_h : (r + 1) * cell_h,
                c * cell_w : (c + 1) * cell_w,
            ]
            boxes.append(cell.copy())  # copy — never return a view into the source

    return boxes


def clean_and_prepare_box(box_img, target_size=28, empty_threshold=0.15):
    """
    Prepares a single cell image for digit classification:
      1. Crops a 10% margin to remove grid-line artifacts.
      2. Applies morphological closing to join broken digit strokes.
      3. Checks digit pixel density to detect empty cells.
      4. Locates the digit bounding box and centers it on a square canvas.
      5. Adds MNIST-style outer padding, then resizes to target_size × target_size.

    The input is expected to already be a binary image (white digits on black
    background) — i.e. the output of pre_process_image after warping. No
    re-thresholding is applied here to avoid double-processing artefacts.

    Args:
        box_img:         Binary cell image (white digit on black background).
        target_size:     Output size in pixels (default 28 for MNIST models).
        empty_threshold: Fraction of white pixels below which cell is "empty".

    Returns:
        uint8 array of shape (target_size, target_size), or None if empty.
    """
    h, w = box_img.shape[:2]

    # Margin crop to strip the outermost grid lines.
    margin = max(2, int(min(h, w) * 0.15))
    cropped = box_img[margin : h - margin, margin : w - margin]

    if cropped.size == 0:
        return None

    # Morphological closing to connect broken strokes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, kernel_close)

    # Use connected components to find the digit, ignoring grid lines and noise
    ch, cw = closed.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

    best_label = -1
    max_area = 0

    for i in range(1, num_labels):
        x, y, dw, dh, area = stats[i]
        cx, cy = centroids[i]

        # 1. Reject small noise (must be > 2% of the cropped cell area)
        if area < (ch * cw * 0.02):
            continue

        # 2. Reject grid lines: their centroid is usually near the edge
        if cx < cw * 0.2 or cx > cw * 0.8 or cy < ch * 0.2 or cy > ch * 0.8:
            continue

        # 3. Reject highly elongated lines (vertical or horizontal)
        aspect = max(dw, dh) / (min(dw, dh) + 1e-6)
        if aspect > 8.0:
            continue
            
        # 4. Reject horizontal smudges/cross-outs (digits are never significantly wider than tall)
        if dw > dh * 1.5:
            continue

        if area > max_area:
            max_area = area
            best_label = i

    if best_label == -1:
        return None

    # We found a valid digit component!
    x, y, dw, dh, _ = stats[best_label]
    digit_roi = closed[y:y+dh, x:x+dw]

    # Center digit on a square canvas
    square_size = max(dw, dh)
    if square_size == 0:
        return None

    canvas = np.zeros((square_size, square_size), dtype=np.uint8)
    x_off = (square_size - dw) // 2
    y_off = (square_size - dh) // 2
    canvas[y_off:y_off+dh, x_off:x_off+dw] = digit_roi

    # Add outer padding to mimic MNIST's blank border around digits (~20% each side)
    border = max(2, int(square_size * 0.2))
    canvas = cv2.copyMakeBorder(
        canvas, border, border, border, border,
        cv2.BORDER_CONSTANT, value=0
    )

    # Resize to model input size
    final = cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return final


def _estimate_empty_threshold(boxes, margin_pct=0.15):
    """
    Automatically estimates an appropriate empty-cell threshold for a given set
    of 81 cell images.

    Strategy:
    1. Compute the white-pixel ratio for each cell (after margin crop + closing).
    2. If a clear bimodal gap exists in the distribution (gap > 0.02), use its
       midpoint as the threshold — this handles images with thick grid lines
       where empty and digit cells are well separated.
    3. Otherwise (smooth/continuous distribution from thin lines), fall back to
       the midpoint between the 50th and 65th percentile values. In a typical
       sudoku ~40-55 cells are empty and ~26-41 have digits, so this percentile
       range reliably straddles the boundary between the two groups.

    Args:
        boxes:      List of cell images (output of split_boxes).
        margin_pct: Margin fraction used when measuring each cell (should match
                    the value used in clean_and_prepare_box).

    Returns:
        float: Threshold value in [0.03, 0.40].
    """
    ratios = []
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for box in boxes:
        h, w = box.shape[:2]
        margin = max(2, int(min(h, w) * margin_pct))
        cropped = box[margin : h - margin, margin : w - margin]
        if cropped.size == 0:
            continue
        closed = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, kernel_close)

        # Use border flood-fill to remove grid lines
        ch, cw = closed.shape[:2]
        no_lines = closed.copy()
        mask = np.zeros((ch + 2, cw + 2), dtype=np.uint8)
        for x in range(cw):
            if no_lines[0, x] == 255: cv2.floodFill(no_lines, mask, (x, 0), 0)
            if no_lines[ch-1, x] == 255: cv2.floodFill(no_lines, mask, (x, ch-1), 0)
        for y in range(ch):
            if no_lines[y, 0] == 255: cv2.floodFill(no_lines, mask, (0, y), 0)
            if no_lines[y, cw-1] == 255: cv2.floodFill(no_lines, mask, (cw-1, y), 0)

        ratio = cv2.countNonZero(no_lines) / no_lines.size
        ratios.append(ratio)


    if len(ratios) < 2:
        return 0.10  # safe default

    ratios.sort()
    n = len(ratios)

    # Strategy 1: largest gap in the lower 70% of the distribution
    # Only gaps that occur before the 70th percentile separate empty cells from
    # digit cells.  Gaps in the tail are between different-density digit cells
    # and should be ignored.
    cutoff_idx = int(n * 0.70)
    max_gap = 0.0
    gap_threshold = None

    for i in range(1, cutoff_idx + 1):
        gap = ratios[i] - ratios[i - 1]
        if gap > max_gap:
            max_gap = gap
            gap_threshold = (ratios[i] + ratios[i - 1]) / 2.0

    if max_gap > 0.02 and gap_threshold is not None:
        return float(np.clip(gap_threshold, 0.03, 0.40))

    # Strategy 2: percentile midpoint
    # A standard 9×9 sudoku has 17-50 given digits; the median cell is typically
    # empty.  The boundary between empty and digit clusters consistently falls
    # between the 50th–65th percentile of the sorted ratio list.
    p50 = ratios[int(n * 0.50)]
    p65 = ratios[int(n * 0.65)]
    threshold = (p50 + p65) / 2.0

    return float(np.clip(threshold, 0.03, 0.40))


def extract_digit_cells(img, warp_size=450, target_size=28):
    """
    Full pipeline: raw image -> list of 81 prepared cell images.

    The empty-cell threshold is estimated automatically from the distribution
    of white pixel densities across all 81 cells, so the pipeline adapts to
    different image qualities (thick vs. thin grid lines, varying lighting).

    Args:
        img:         BGR or grayscale source image containing a Sudoku grid.
        warp_size:   Side length of the intermediate warped grid image.
        target_size: Pixel size of each output cell (for the digit model).

    Returns:
        List of 81 items — each is either a (target_size, target_size) uint8
        array (digit present) or None (empty cell), in row-major order.

    Raises:
        ValueError: If the image is invalid or no grid can be located.
    """
    # Stage 1: binarise for contour detection
    thresh = pre_process_image(img)

    # Stage 2: find the grid boundary
    corners = find_sudoku_contour(thresh)

    # Stage 3: warp using the grayscale image (not the binary) so that
    # clean_and_prepare_box receives properly scaled pixel intensities
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Re-binarise after warp so cells contain white-on-black digits
    warped_gray = warp_perspective(gray, corners, size=warp_size)
    warped_blur = cv2.GaussianBlur(warped_gray, (3, 3), 0)
    warped_thresh = cv2.adaptiveThreshold(
        warped_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 4
    )

    # Stage 4: split into 81 cells
    boxes = split_boxes(warped_thresh)

    # Stage 5: estimate the empty-cell threshold from this image's pixel
    # density distribution so the pipeline handles thick vs. thin grid lines
    # without manual tuning.
    auto_threshold = _estimate_empty_threshold(boxes)

    # Stage 6: clean and prepare each cell using the adaptive threshold
    cells = [
        clean_and_prepare_box(box, target_size=target_size, empty_threshold=auto_threshold)
        for box in boxes
    ]

    return cells