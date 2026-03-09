import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from cnn.Tensor import Tensor
from cnn.CNN import CNN

MODEL_OPTIONS = ["2conv (Simpler)", "3conv (Stronger)"]

with st.sidebar:
    st.header("Config")

    model_choice = st.selectbox("Select Model", options=MODEL_OPTIONS, index=0, key="model_choice")

    show_image = st.checkbox("Show Image", value=False, key="show_image")

    show_characters = st.checkbox("Character Hints", value=True, key="show_characters")

c1, c2 = st.columns([13, 7])


def characters():
    if not show_characters:
        return
    st.markdown(":gray[क,  ख,  ग,  घ,  ङ,  च]")
    st.markdown(":gray[छ,  ज,  झ,  ञ,  ट,  ठ]")
    st.markdown(":gray[ड,  ढ,  ण,  त,  थ,  द]")
    st.markdown(":gray[ध,  न,  प,  फ,  ब,  भ]")
    st.markdown(":gray[म,  य,  र,  ल,  व,  श]")
    st.markdown(":gray[ष,  स,  ह,  क्ष,  त्र,  ज्ञ]")


with c1:
    st.subheader("Draw")
    result = st_canvas(stroke_width=10, stroke_color="#1f6feb", background_color="#000000", height=400, width=400,
                       drawing_mode="freedraw", key="canvas", update_streamlit=True, )
    st.markdown("*:gray[See sidebar for config options.]*")

with c2:
    st.subheader("Prediction")

img = None
if result is not None:
    if hasattr(result, "image_data"):
        img = result.image_data
    elif isinstance(result, dict):
        img = result.get("image_data")
else:
    img = np.asarray(img)

image_gray = None
if img is not None:
    weights = [0.333, 0.333, 0.333]
    image_gray = np.dot(img[..., :3], weights)
    image_gray = image_gray.astype(np.uint8)
    image_gray = image_gray / 255
    image_gray = image_gray ** 0.05

input_img = None
IMAGE_SIZE = 32

if image_gray is not None and all([len(d) > 0 for d in np.where(image_gray > 0.10)]):
    y_nonzero, x_nonzero = np.where(image_gray > 0.10)
    ymin, ymax = y_nonzero.min(), y_nonzero.max()
    xmin, xmax = x_nonzero.min(), x_nonzero.max()
    if ymax - ymin < IMAGE_SIZE * 4 or xmax - xmin < IMAGE_SIZE * 4:
        with c2:
            st.write("That's too small, draw bigger!")
            characters()
    else:
        cropped = image_gray[ymin:ymax + 1, xmin:xmax + 1]

        target_inner = 28
        h, w = cropped.shape

        scale = target_inner / max(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

        y_offset = (IMAGE_SIZE - new_h) // 2
        x_offset = (IMAGE_SIZE - new_w) // 2

        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        input_img = canvas
else:
    with c2:
        if show_characters:
            st.markdown("Try drawing these characters:")
        else:
            st.markdown("Draw something!")
        characters()

if model_choice == MODEL_OPTIONS[0]:
    model = CNN.from_mw("models/2conv.mw")
else:
    model = CNN.from_mw("models/3conv.mw")

labels = pd.read_csv("streamlit/labels.csv")
consonants = sorted(labels['Label'].tolist())
symbols = {k: v for k, v in zip(labels['Label'], labels['Devanagari Label'])}

with c2:
    if input_img is not None:
        if type(input_img) is np.ndarray and show_image:
            st.image(input_img, width=input_img.shape[0] * 4)
        x = Tensor(input_img.tolist(), (1, IMAGE_SIZE, IMAGE_SIZE))

        if sum(x.data) != 0:
            probs = model.predict(x)
            preds = {k: round(v * 100, 2) for k, v in zip(consonants, probs)}
            preds = {k: v for k, v in sorted(preds.items(), key=lambda item: item[1], reverse=True)}

            for i, item in enumerate(preds.items()):
                k, v = item
                st.write(f"{symbols[k]} ({k}): {v}%")
                if i == 5:
                    break
        else:
            if show_characters:
                st.markdown("Try drawing these characters:")
            else:
                st.markdown("Draw something!")

        characters()