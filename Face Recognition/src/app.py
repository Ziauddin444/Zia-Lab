import streamlit as st
import numpy as np
import cv2
from src.detect import detect_faces_from_image, draw_boxes


st.set_page_config(page_title="Face Detection", layout="centered")

st.title("Face Detection demo")
st.write("Upload an image and the app will detect faces using OpenCV Haar cascades.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not decode image")
    else:
        boxes = detect_faces_from_image(img)
        out = draw_boxes(img, boxes)
        # Convert BGR->RGB for display
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        st.image(out_rgb, caption=f"Detected {len(boxes)} face(s)", use_column_width=True)

        # Prepare annotated image for download
        ok, buf = cv2.imencode('.png', out)
        if ok:
            b = buf.tobytes()
            st.download_button(
                "Download annotated image",
                data=b,
                file_name="annotated.png",
                mime="image/png",
            )

        st.write("Bounding boxes:")
        for i, (x, y, w, h) in enumerate(boxes, start=1):
            st.write(f"{i}: x={x}, y={y}, w={w}, h={h}")
