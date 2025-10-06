import cv2
import numpy as np
import argparse
from typing import List, Tuple, Optional

CascadePath = (
    cv2.data.haarcascades  # type: ignore[attr-defined]
    + "haarcascade_frontalface_default.xml"
)


def detect_faces_from_image(
    img: np.ndarray,
    scaleFactor: float = 1.1,
    minNeighbors: int = 5,
    minSize: Tuple[int, int] = (30, 30),
) -> List[Tuple[int, int, int, int]]:
    """Detect faces in an image (numpy BGR) and return list of rectangles (x, y, w, h)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(CascadePath)
    rects = detector.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize,
    )
    # Ensure we return tuples of ints (mypy-friendly)
    rects_list: List[Tuple[int, int, int, int]] = [
        (int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in rects
    ]
    return rects_list


def draw_boxes(
    img: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    color=(0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    out = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def detect_in_image(path: str, output: Optional[str] = None, show: bool = False):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {path}")
    boxes = detect_faces_from_image(img)
    out = draw_boxes(img, boxes)
    if output:
        cv2.imwrite(output, out)
    if show:
        cv2.imshow("faces", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return boxes


def detect_in_video(source: int | str = 0, output: Optional[str] = None):
    """Capture from webcam or video file and run face detection live.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    writer = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(  # type: ignore[attr-defined]
            output,
            fourcc,
            fps,
            (w, h),
        )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            boxes = detect_faces_from_image(frame)
            out = draw_boxes(frame, boxes)
            cv2.imshow("face-detection", out)
            if writer:
                writer.write(out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Simple face detection demo using OpenCV Haar cascades"
        )
    )
    sub = p.add_subparsers(dest="cmd")

    img = sub.add_parser("image", help="Detect faces in a single image")
    img.add_argument("path", help="Path to input image")
    img.add_argument("--output", help="Optional output path to save annotated image")
    img.add_argument("--show", action="store_true", help="Show annotated image")

    cam = sub.add_parser(
        "webcam",
        help="Run face detection on webcam (or video file)",
    )
    cam.add_argument(
        "--source",
        default=0,
        help=(
            "Video source (0 for default webcam or path to file)"
        ),
    )
    cam.add_argument("--output", help="Optional output video path (mp4)")

    args = p.parse_args()
    if args.cmd == "image":
        boxes = detect_in_image(
            args.path, output=args.output, show=args.show
        )
        print("Detected {} faces".format(len(boxes)))
    elif args.cmd == "webcam":
        src = int(args.source) if str(args.source).isdigit() else args.source
        detect_in_video(src, output=args.output)
    else:
        p.print_help()
