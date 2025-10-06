import numpy as np
from src.detect import detect_faces_from_image


def test_blank_image_has_no_faces():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = detect_faces_from_image(img)
    assert isinstance(boxes, list)
    assert len(boxes) == 0
