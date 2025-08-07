"""Logo Detection App using Streamlit, YOLOv8, and OpenCV.

Allows upload/search for images/videos, displays the original
media, runs YOLOv8 detection, and displays per-frame and
total video logo counts.
"""

import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from typing import List, Dict, Tuple, Optional

def get_logo_counts(
    results, class_names: List[str]
) -> Dict[str, int]:
    """Counts logos detected per frame."""
    if not results:
        return {}
    r = results[0]
    if not hasattr(r, "boxes") or r.boxes is None:
        return {}
    class_ids = r.boxes.cls.cpu().numpy().astype(int)
    unique, counts = np.unique(class_ids, return_counts=True)
    logo_counts = {class_names[i]: c for i, c in zip(unique, counts)}
    return logo_counts

def get_fps_and_length(
    cap: cv2.VideoCapture
) -> Tuple[float, int]:
    """Extracts FPS and total frame count from a video."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps if fps > 0 else 25, length

def process_video_per_second_and_total(
    video_path: str,
    model: YOLO,
    class_names: List[str],
    stframe
) -> Tuple[List[Tuple[int, Dict[str, int]]], Dict[str, int]]:
    """Processes a video, runs detection every second, aggregates results."""
    cap = cv2.VideoCapture(video_path)
    fps, total_frames = get_fps_and_length(cap)
    results_by_second = []
    all_class_ids = []
    sec = 0
    while cap.isOpened():
        frame_number = int(sec * fps)
        if frame_number >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        result_img = results[0].plot()
        stframe.image(result_img, channels="BGR", caption=f"Second: {sec+1}", use_container_width=True)

        logo_counts = get_logo_counts(results, class_names)
        results_by_second.append((sec + 1, logo_counts))
        # Accumulate class IDs for total count.
        if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            all_class_ids.extend(class_ids)
        sec += 1

    cap.release()
    total_logo_counts = {}
    if all_class_ids:
        unique, counts = np.unique(all_class_ids, return_counts=True)
        total_logo_counts = {class_names[i]: c for i, c in zip(unique, counts)}
    return results_by_second, total_logo_counts

def main():
    """Main entry point for the Streamlit logo detection application."""
    # --- Load trained model ---
    model = YOLO(r'content\runs\detect\train\weights\best.pt')  # UPDATE AS NEEDED

    # --- Folder paths for search mode ---
    PHOTO_FOLDER = r"sample_photos"    # UPDATE AS NEEDED
    VIDEO_FOLDER = r"sample_videos"    # UPDATE AS NEEDED

    # --- Streamlit UI setup ---
    st.set_page_config(page_title="Logo Detection App", layout="centered")
    st.title("🔍 Logo Detection with YOLOv8")
    st.markdown("Choose upload or search existing files to detect logos.")

    input_mode = st.radio("Choose Input Method", ["Upload", "Search"])
    file_type = st.radio("Select File Type", ["Image", "Video"])
    selected_file_path: Optional[str] = None

    # --------------------- UPLOAD MODE ---------------------
    if input_mode == "Upload":
        if file_type == "Image":
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", use_container_width=True)
                if st.button("Detect Logos"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                        img.save(temp.name)
                        results = model(temp.name)
                    for r in results:
                        result_img = r.plot()
                        st.image(result_img, caption="Detected Logos", use_container_width=True)
                    logo_counts = get_logo_counts(results, model.names)
                    if logo_counts:
                        st.subheader("Detected Logo Counts")
                        for name, count in logo_counts.items():
                            st.write(f"{name}: {count}")
                    else:
                        st.info("No logos detected.")

        elif file_type == "Video":
            uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
            if uploaded_video:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                if st.button("Detect Logos in Video (1 FPS)"):
                    stframe = st.empty()
                    results_by_second, total_logo_counts = process_video_per_second_and_total(
                        tfile.name, model, model.names, stframe)
                    st.subheader("Logos Detected Per Second")
                    for sec, logo_counts in results_by_second:
                        if logo_counts:
                            st.write(f"Second {sec}:")
                            for name, count in logo_counts.items():
                                st.write(f"- {name}: {count}")
                        else:
                            st.write(f"Second {sec}: No logos detected.")
                    st.divider()
                    st.subheader("Total Logos Detected in Video")
                    if total_logo_counts:
                        for name, count in total_logo_counts.items():
                            st.write(f"{name}: {count}")
                    else:
                        st.info("No logos detected in entire video.")

    # --------------------- SEARCH MODE ---------------------
    elif input_mode == "Search":
        if file_type == "Image":
            try:
                image_files = [f for f in os.listdir(PHOTO_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            except FileNotFoundError:
                image_files = []
            selected_filename = st.selectbox("Select an image", image_files)
            if selected_filename:
                selected_file_path = os.path.join(PHOTO_FOLDER, selected_filename)
                img = Image.open(selected_file_path)
                st.image(img, caption="Selected Image", use_container_width=True)
                if st.button("Detect Logos"):
                    results = model(selected_file_path)
                    for r in results:
                        result_img = r.plot()
                        st.image(result_img, caption="Detected Logos", use_container_width=True)
                    logo_counts = get_logo_counts(results, model.names)
                    if logo_counts:
                        st.subheader("Detected Logo Counts")
                        for name, count in logo_counts.items():
                            st.write(f"{name}: {count}")
                    else:
                        st.info("No logos detected.")

        elif file_type == "Video":
            try:
                video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith((".mp4", ".avi", ".mov"))]
            except FileNotFoundError:
                video_files = []
            selected_filename = st.selectbox("Select a video", video_files)
            if selected_filename:
                selected_file_path = os.path.join(VIDEO_FOLDER, selected_filename)
                # --- Display the original video, before detection ---
                with open(selected_file_path, "rb") as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes, format="video/mp4", start_time=0)

                if st.button("Detect Logos in Video (1 FPS)"):
                    stframe = st.empty()
                    results_by_second, total_logo_counts = process_video_per_second_and_total(
                        selected_file_path, model, model.names, stframe)
                    st.subheader("Logos Detected Per Second")
                    for sec, logo_counts in results_by_second:
                        if logo_counts:
                            st.write(f"Second {sec}:")
                            for name, count in logo_counts.items():
                                st.write(f"- {name}: {count}")
                        else:
                            st.write(f"Second {sec}: No logos detected.")
                    st.divider()
                    st.subheader("Total Logos Detected in Video")
                    if total_logo_counts:
                        for name, count in total_logo_counts.items():
                            st.write(f"{name}: {count}")
                    else:
                        st.info("No logos detected in entire video.")

if _name_ == "_main_":
    main()
