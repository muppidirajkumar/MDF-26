# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 12:38:48 2026

@author: User
"""

import os
import shutil
import cv2
import pandas as pd
from pathlib import Path
from moviepy import VideoFileClip

# ==========================================================
# INPUT DATASET
# ==========================================================
DATASET_ROOT = r"Datasets"

# ==========================================================
# OUTPUT
# ==========================================================
OUTPUT_ROOT = r"MultiDF-23"

# ==========================================================
# VIDEO EXTENSIONS
# ==========================================================
VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"
}

# ==========================================================
# OUTPUT FOLDERS
# ==========================================================
folders = [
    "videos/real",
    "videos/fake",
    "frames/real",
    "frames/fake",
    "audio/real",
    "audio/fake",
    "transcripts/real",
    "transcripts/fake",
    "metadata/real",
    "metadata/fake"
]

for folder in folders:
    os.makedirs(os.path.join(OUTPUT_ROOT, folder), exist_ok=True)

# ==========================================================
# DATASET PREFIX
# ==========================================================
dataset_prefix = {
    "dfdc": "DFDC",
    "FaceForensics++": "FFPP",
    "Celeb-DF-v2": "CDF",
    "FakeAVCeleb_v1.2": "FAC",
    "ForgeryNet": "FGN",
    "DeepFakeTIMIT": "DTM"
}

# ==========================================================
# LABEL KEYWORDS
# ==========================================================
REAL_KEYWORDS = [
    "real",
    "original",
    "authentic",
    "genuine"
]

FAKE_KEYWORDS = [
    "fake",
    "deepfake",
    "manipulated",
    "forged",
    "tampered",
    "synthesis"
]

# ==========================================================
# LABEL DETECTION
# ==========================================================
def determine_label(path):

    p = path.lower()

    for k in REAL_KEYWORDS:
        if k in p:
            return "real", 0

    for k in FAKE_KEYWORDS:
        if k in p:
            return "fake", 1

    return None, None


# ==========================================================
# FRAME EXTRACTION
# ==========================================================
def extract_frames(video_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_no = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_name = os.path.join(
            output_folder,
            f"frame_{frame_no:05d}.jpg"
        )

        cv2.imwrite(frame_name, frame)

        frame_no += 1

    cap.release()

    return frame_no


# ==========================================================
# AUDIO EXTRACTION
# ==========================================================
def extract_audio(video_path, audio_path):

    try:

        clip = VideoFileClip(video_path)

        if clip.audio is not None:
            clip.audio.write_audiofile(
                audio_path,
                logger=None
            )

        clip.close()

    except Exception as e:
        print("Audio extraction failed:", e)


# ==========================================================
# PROCESS
# ==========================================================
records = []

for dataset_name, prefix in dataset_prefix.items():

    dataset_path = os.path.join(DATASET_ROOT, dataset_name)

    if not os.path.exists(dataset_path):
        print(dataset_name, "not found")
        continue

    print("\nScanning", dataset_name)

    counter = 1

    for root, dirs, files in os.walk(dataset_path):

        for file in files:

            ext = Path(file).suffix.lower()

            if ext not in VIDEO_EXTENSIONS:
                continue

            fullpath = os.path.abspath(
                os.path.join(root, file)
            )

            cls, label = determine_label(fullpath)

            if cls is None:
                continue

            video_id = f"{prefix}_{counter:06d}"

            video_destination = os.path.join(
                OUTPUT_ROOT,
                "videos",
                cls,
                video_id + ext
            )

            frame_folder = os.path.join(
                OUTPUT_ROOT,
                "frames",
                cls,
                video_id
            )

            audio_path = os.path.join(
                OUTPUT_ROOT,
                "audio",
                cls,
                video_id + ".wav"
            )

            transcript_path = os.path.join(
                OUTPUT_ROOT,
                "transcripts",
                cls,
                video_id + ".txt"
            )

            metadata_path = os.path.join(
                OUTPUT_ROOT,
                "metadata",
                cls,
                video_id + ".json"
            )

            try:

                # Copy Video
                if not os.path.exists(video_destination):
                    shutil.copy2(fullpath, video_destination)

                # Extract Frames
                if not os.path.exists(frame_folder):
                    extract_frames(
                        video_destination,
                        frame_folder
                    )

                # Extract Audio
                if not os.path.exists(audio_path):
                    extract_audio(
                        video_destination,
                        audio_path
                    )

                records.append({

                    "video_id": video_id,

                    "dataset": dataset_name,

                    "video_path": video_destination,

                    "frame_folder": frame_folder,

                    "audio_path": audio_path,

                    "transcript_path": transcript_path,

                    "metadata_path": metadata_path,

                    "label": label

                })

                print(video_id, "Processed")

                counter += 1

            except Exception as e:

                print("Failed:", fullpath)

                print(e)

# ==========================================================
# SAVE CSV
# ==========================================================
df = pd.DataFrame(records)

csv_path = os.path.join(
    OUTPUT_ROOT,
    "labels.csv"
)

df.to_csv(csv_path, index=False)

print("\n======================================")
print("Dataset Creation Completed")
print("Total Videos :", len(df))
print("CSV Saved :", csv_path)
print("======================================")