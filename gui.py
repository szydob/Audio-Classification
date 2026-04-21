from __future__ import annotations

import warnings
import random
from pathlib import Path
import os.path as osp

import kagglehub
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.audio_classification.data import (
    scan_labeled_audio,
    split_files_train_val_test,
    audio_to_logmel,
    load_audio,
)
from src.audio_classification.sota import run_ast_logreg_baseline

warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

st.set_page_config(page_title="Audio Classification", layout="wide")
st.title("Audio Classification")
st.write("Load a dataset, split it, and show AST predictions on the test set.")

DATASET_ID = "andradaolteanu/gtzan-dataset-music-genre-classification"


# Show summary of dataset split sizes and class balance in a simple table
def show_split_summary() -> None:
    """Show split sizes and class balance in a simple table."""
    if not all(key in st.session_state for key in ["train_files", "val_files", "test_files", "class_names"]):
        return

    class_names = st.session_state["class_names"]
    summary = pd.DataFrame(
        {
            "train": np.bincount(st.session_state["y_train"], minlength=len(class_names)),
            "val": np.bincount(st.session_state["y_val"], minlength=len(class_names)),
            "test": np.bincount(st.session_state["y_test"], minlength=len(class_names)),
        },
        index=class_names,
    )

    st.subheader("Dataset split")
    cols = st.columns(4)
    cols[0].metric("Train", len(st.session_state["train_files"]))
    cols[1].metric("Val", len(st.session_state["val_files"]))
    cols[2].metric("Test", len(st.session_state["test_files"]))
    cols[3].metric("Classes", len(class_names))
    st.dataframe(summary, width="stretch")


# Show test results table and a simple accuracy metric
def show_test_results(results: pd.DataFrame) -> None:
    """Show test predictions and a simple accuracy summary."""
    if results.empty:
        st.warning("No test results to show.")
        return

    accuracy = float(results["correct"].mean())
    st.metric("Test accuracy", f"{accuracy:.4f}")
    st.dataframe(
        results[["true_label", "predicted_label", "correct", "confidence"]],
        width="stretch",
        height=420,
    )
    st.caption(f"Correct predictions: {int(results['correct'].sum())} / {len(results)}")


def _display_spectrogram(path: str | Path) -> None:
    """Helper: load audio, compute log-mel and render matplotlib spectrogram in Streamlit."""
    y = load_audio(path)
    logmel = audio_to_logmel(y)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(logmel, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(osp.basename(str(path)))
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bins")
    fig.colorbar(im, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)
    plt.close(fig)


# Keep track of dataset root and split in session state to share between actions
if "dataset_root" not in st.session_state:
    st.session_state["dataset_root"] = None

# Sidebar: dataset source and split parameters
source = st.sidebar.selectbox("Dataset source", ["GTZAN (KaggleHub)", "Local folder"])
train_val_fraction = st.sidebar.slider("Use for train+val (%)", 50, 95, 80, 5)
val_fraction = st.sidebar.slider("Validation fraction inside used data", 0.05, 0.4, 0.2, 0.05)

if source == "GTZAN (KaggleHub)":
    if st.sidebar.button("Download / use GTZAN"):
        st.session_state["dataset_root"] = kagglehub.dataset_download(DATASET_ID)
else:
    local_root = st.sidebar.text_input("Local dataset path")
    if local_root:
        st.session_state["dataset_root"] = local_root

st.info(f"Dataset root: {st.session_state.get('dataset_root')}")

# Layout: left column shows split summary, right column shows random example
cols = st.columns([1, 2])
with cols[0]:
    if all(key in st.session_state for key in ["train_files", "val_files", "test_files", "class_names"]):
        show_split_summary()

# Prepare stratified split on button click
if st.button("Prepare split"):
    root_value = st.session_state.get("dataset_root")
    if not root_value:
        st.error("Choose dataset first.")
    else:
        files, y, class_names = scan_labeled_audio(root_value)
        train_files, val_files, test_files, y_train, y_val, y_test = split_files_train_val_test(
            files,
            y,
            used_fraction=train_val_fraction / 100,
            val_fraction_of_used=val_fraction,
        )

        st.session_state.update(
            {
                "class_names": class_names,
                "train_files": train_files,
                "val_files": val_files,
                "test_files": test_files,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
            }
        )

        st.success(
            f"Split ready: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
        )
        show_split_summary()

with cols[1]:
    st.subheader("Random example")
    if st.button("Show random example spectrogram"):
        # collect candidate files from prepared split (prefer test)
        candidates = []
        for key in ("test_files", "val_files", "train_files"):
            if key in st.session_state:
                candidates.extend(st.session_state[key])

        # fallback: scan dataset root if available
        if not candidates and st.session_state.get("dataset_root"):
            files_all, _, _ = scan_labeled_audio(st.session_state["dataset_root"])
            candidates = files_all

        if not candidates:
            st.warning("No audio files available — prepare split or set dataset root first.")
        else:
            file_path = random.choice(candidates)
            try:
                _display_spectrogram(file_path)
            except Exception as exc:
                st.error(f"Failed to render spectrogram: {exc!r}")

# SotA - running AST baseline and displaying results
st.subheader("AST baseline")

# Button to run SoTA baseline on the test set, with error handling and progress indication
if st.button("Run SoTA baseline"):
    required = ["train_files", "test_files", "y_train", "y_test", "class_names"]
    if not all(key in st.session_state for key in required):
        st.error("Prepare split first.")
    else:
        with st.spinner("Classifying test set with AST..."):
            try:
                # No action needed for last example — we don't persistently display it
                results = run_ast_logreg_baseline(
                    train_files=st.session_state["train_files"],
                    y_train=st.session_state["y_train"],
                    test_files=st.session_state["test_files"],
                    y_test=st.session_state["y_test"],
                    class_names=st.session_state["class_names"],
                )
                st.session_state["ast_test_results"] = results
                st.success("Test set classified.")
            except Exception as exc:
                st.error(f"SoTA failed: {exc!r}")

# Show test results if available
if "ast_test_results" in st.session_state:
    st.subheader("Test predictions")
    show_test_results(st.session_state["ast_test_results"])
