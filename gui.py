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


from core.data.io import (
    scan_labeled_audio,
    split_files_train_val_test,
    audio_to_logmel,
    load_audio,
)
from core.models.cnn import run_cnn_baseline, train_and_save_cnn_baseline, load_and_eval_cnn_baseline
from core.models.sota import run_ast_logreg_baseline
from core.models.feature_model import run_feature_baseline, train_and_save_feature_baseline, load_and_eval_feature_baseline
from core.models.transformer import train_and_save_transformer_baseline, run_saved_transformer_baseline, run_transformer_baseline

warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

st.set_page_config(page_title="Audio Classification", layout="wide")

DATASET_ID = "andradaolteanu/gtzan-dataset-music-genre-classification"

# Initialize navigation state
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"


# ============================================================================
# Helper Functions
# ============================================================================

def show_split_summary() -> None:
    """Show split sizes and class balance in a simple table."""
    if not all(
        key in st.session_state
        for key in ["train_files", "val_files", "test_files", "class_names"]
    ):
        return

    class_names = st.session_state["class_names"]
    summary = pd.DataFrame(
        {
            "train": np.bincount(
                st.session_state["y_train"], minlength=len(class_names)
            ),
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


def show_test_results(results: pd.DataFrame) -> None:
    """Show test predictions and a simple accuracy summary."""
    if results.empty:
        st.warning("No test results to show.")
        return

    accuracy = float(results["correct"].mean())
    st.metric("Test accuracy", f"{accuracy:.4f}")
    st.dataframe(
        results[["true_label", "predicted_label", "correct", "confidence"]],
        use_container_width=True,
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


def navigate_to(page: str) -> None:
    """Update navigation state."""
    st.session_state["current_page"] = page


# ============================================================================
# Sidebar: Dataset Management (always visible)
# ============================================================================

st.sidebar.title("Dataset Management")

source = st.sidebar.selectbox("Dataset source", ["GTZAN (KaggleHub)", "Local folder"])
train_val_fraction = st.sidebar.slider("Use for train+val (%)", 50, 95, 80, 5)
val_fraction = st.sidebar.slider(
    "Validation fraction inside used data", 0.05, 0.4, 0.2, 0.05
)

if source == "GTZAN (KaggleHub)":
    if st.sidebar.button("Download / use GTZAN"):
        st.session_state["dataset_root"] = kagglehub.dataset_download(DATASET_ID)
else:
    local_root = st.sidebar.text_input("Local dataset path")
    if local_root:
        st.session_state["dataset_root"] = local_root

st.sidebar.info(f"Dataset root: {st.session_state.get('dataset_root', 'Not set')}")

# Prepare stratified split on button click
if st.sidebar.button("🔄 Prepare split"):
    root_value = st.session_state.get("dataset_root")
    if not root_value:
        st.sidebar.error("Choose dataset first.")
    else:
        with st.spinner("Scanning and splitting dataset..."):
            files, y, class_names = scan_labeled_audio(root_value)
            train_files, val_files, test_files, y_train, y_val, y_test = (
                split_files_train_val_test(
                    files,
                    y,
                    used_fraction=train_val_fraction / 100,
                    val_fraction_of_used=val_fraction,
                )
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
            st.sidebar.success(
                f"Split ready: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
            )

# Show dataset summary in sidebar
if all(
    key in st.session_state
    for key in ["train_files", "val_files", "test_files", "class_names"]
):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Summary")
    st.sidebar.metric("Train files", len(st.session_state["train_files"]))
    st.sidebar.metric("Val files", len(st.session_state["val_files"]))
    st.sidebar.metric("Test files", len(st.session_state["test_files"]))
    st.sidebar.metric("Classes", len(st.session_state["class_names"]))

# ============================================================================
# Page: Home - Model Selection Dashboard
# ============================================================================

def page_home():
    """Display the home page with 4 interactive model cards."""
    st.title("🎵 Audio Classification Dashboard")
    st.write("Select a model to configure and run classification")

    if not all(
        key in st.session_state
        for key in ["train_files", "val_files", "test_files", "class_names"]
    ):
        st.warning("⚠️ Please prepare a split first using the sidebar.")
        return

    st.markdown("---")
    st.subheader("Available Models")

    # Create 2x2 grid of model cards
    col1, col2 = st.columns(2)

    # Card 1: AST Baseline
    with col1:
        st.markdown(
            """
        <div style='padding: 20px; border: 2px solid #1f77b4; border-radius: 10px; background-color: #000206;'>
            <h3>AST Baseline</h3>
            <p>SotA Baseline</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("Open AST Baseline", key="btn_ast", use_container_width=True):
            navigate_to("ast")
            st.rerun()

    # Card 2: Feature-based Baseline
    with col2:
        st.markdown(
            """
        <div style='padding: 20px; border: 2px solid #ff7f0e; border-radius: 10px; background-color: #000206;'>
            <h3>Feature-based Baseline</h3>
            <p>Hand-crafted Audio Features + Logistic Regression</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("Open Feature Baseline", key="btn_feature", use_container_width=True):
            navigate_to("feature")
            st.rerun()

    col3, col4 = st.columns(2)

    # Card 3: Mini AudioTransformer
    with col3:
        st.markdown(
            """
        <div style='padding: 20px; border: 2px solid #2ca02c; border-radius: 10px; background-color: #000206;'>
            <h3>Mini AudioTransformer</h3>
            <p>Lightweight Custom Transformer Architecture</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("Open Mini AudioTransformer", key="btn_transformer", use_container_width=True):
            navigate_to("transformer")
            st.rerun()

    # Card 4: CNN Baseline
    with col4:
        st.markdown(
            """
        <div style='padding: 20px; border: 2px solid #d62728; border-radius: 10px; background-color: #000206;'>
            <h3>CNN Baseline</h3>
            <p>Custom Convolutional Neural Network</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("Open CNN Baseline", key="btn_cnn", use_container_width=True):
            navigate_to("cnn")
            st.rerun()

    st.markdown("---")
    st.subheader("Dataset Example")
    if st.button("Show random example spectrogram"):
        candidates = []
        for key in ("test_files", "val_files", "train_files"):
            if key in st.session_state:
                candidates.extend(st.session_state[key])

        if not candidates and st.session_state.get("dataset_root"):
            files_all, _, _ = scan_labeled_audio(st.session_state["dataset_root"])
            candidates = files_all

        if not candidates:
            st.warning(
                "No audio files available — prepare split or set dataset root first."
            )
        else:
            file_path = random.choice(candidates)
            try:
                _display_spectrogram(file_path)
            except Exception as exc:
                st.error(f"Failed to render spectrogram: {exc!r}")


# ============================================================================
# Page: AST Baseline
# ============================================================================

def page_ast():
    """Page for AST baseline model."""
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Back to Home"):
            navigate_to("home")
            st.rerun()
    with col_title:
        st.title("AST Baseline")

    st.markdown("---")

    required = ["train_files", "test_files", "y_train", "y_test", "class_names"]
    if not all(key in st.session_state for key in required):
        st.error("Prepare split first using the sidebar.")
        return

    st.subheader("Configuration")
    st.info("AST Baseline uses an AudioSet pre-trained transformer with logistic regression on top.")

    if st.button("Run SoTA (AST) Baseline", use_container_width=True):
        with st.spinner("Classifying test set with AST..."):
            try:
                results = run_ast_logreg_baseline(
                    train_files=st.session_state["train_files"],
                    y_train=st.session_state["y_train"],
                    test_files=st.session_state["test_files"],
                    y_test=st.session_state["y_test"],
                    class_names=st.session_state["class_names"],
                )
                st.session_state["ast_test_results"] = results
                st.success("✅ Test set classified successfully!")
            except Exception as exc:
                st.error(f"❌ AST failed: {exc!r}")

    st.markdown("---")
    st.subheader("Results")

    if "ast_test_results" in st.session_state:
        show_test_results(st.session_state["ast_test_results"])
    else:
        st.info("Run the model to see results.")


# ============================================================================
# Page: Feature-based Baseline
# ============================================================================

def page_feature():
    """Page for feature-based baseline model."""
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Back to Home"):
            navigate_to("home")
            st.rerun()
    with col_title:
        st.title("Feature-based Baseline")

    st.markdown("---")

    required = ["train_files", "test_files", "y_train", "y_test", "class_names"]
    if not all(key in st.session_state for key in required):
        st.error("Prepare split first using the sidebar.")
        return

    st.subheader("Configuration")
    # st.info(
    #     "Feature-based baseline extracts hand-crafted audio features "
    #     "(MFCC, spectral features, chroma, etc.) and trains a logistic regression classifier."
    # )

    mode = st.radio("Mode", ["Quick Eval", "Train & Save"], horizontal=True)

    if mode == "Train & Save":
        required_full = [
            "train_files", "val_files", "test_files",
            "y_train", "y_val", "y_test", "class_names"
        ]
        if not all(key in st.session_state for key in required_full):
            st.error("Prepare split first using the sidebar.")
            return

        st.subheader("Training Configuration")
        col1, col2 = st.columns(2)
        with col1:
            save_dir = st.text_input("Save directory", value="artifacts/feature_baseline")

        if st.button("Train & Save Feature Baseline", use_container_width=True):
            with st.spinner("Training feature-based classifier..."):
                try:
                    results, history = train_and_save_feature_baseline(
                        train_files=st.session_state["train_files"],
                        y_train=st.session_state["y_train"],
                        val_files=st.session_state["val_files"],
                        y_val=st.session_state["y_val"],
                        test_files=st.session_state["test_files"],
                        y_test=st.session_state["y_test"],
                        class_names=st.session_state["class_names"],
                        save_dir=save_dir,
                    )
                    st.session_state["feature_test_results"] = results
                    st.session_state["feature_history"] = history
                    st.success("✅ Feature-based model trained and saved!")
                except Exception as exc:
                    st.error(f"❌ Training failed: {exc!r}")

        if "feature_history" in st.session_state:
            st.subheader("Training Metrics")
            hist_df = pd.DataFrame(st.session_state["feature_history"])
            st.line_chart(hist_df, use_container_width=True)

    else:  # Quick Eval
        # if st.button("Run Feature Baseline (Quick)", use_container_width=True):
        #     with st.spinner("Training feature-based classifier..."):
        #         try:
        #             results = run_feature_baseline(
        #                 train_files=st.session_state["train_files"],
        #                 y_train=st.session_state["y_train"],
        #                 test_files=st.session_state["test_files"],
        #                 y_test=st.session_state["y_test"],
        #                 class_names=st.session_state["class_names"],
        #             )
        #             st.session_state["feature_test_results"] = results
        #             st.success("✅ Feature-based classification finished!")
        #         except Exception as exc:
        #             st.error(f"❌ Feature baseline failed: {exc!r}")

        # # Option to load saved model
        # st.markdown("---")
        st.subheader("Load Saved Model")
        load_dir = st.text_input("Load directory", value="artifacts/feature_baseline_custom")
        if st.button("Load & Evaluate Saved Model", use_container_width=True):
            with st.spinner("Loading saved model and evaluating..."):
                try:
                    results = load_and_eval_feature_baseline(
                        test_files=st.session_state["test_files"],
                        y_test=st.session_state["y_test"],
                        class_names=st.session_state["class_names"],
                        load_dir=load_dir,
                    )
                    st.session_state["feature_test_results"] = results
                    st.success("✅ Loaded model evaluated successfully!")
                except Exception as exc:
                    st.error(f"❌ Loading failed: {exc!r}")

    st.markdown("---")
    st.subheader("Results")

    if "feature_test_results" in st.session_state:
        show_test_results(st.session_state["feature_test_results"])
    else:
        st.info("Run the model to see results.")


# ============================================================================
# Page: Mini AudioTransformer
# ============================================================================

def page_transformer():
    """Page for Mini AudioTransformer model."""
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Back to Home"):
            navigate_to("home")
            st.rerun()
    with col_title:
        st.title("Mini AudioTransformer")

    st.markdown("---")

    required_base = ["test_files", "y_test", "class_names"]
    if not all(key in st.session_state for key in required_base):
        st.error("Prepare split first using the sidebar.")
        return

    st.subheader("Configuration")
    # st.info(
    #     "Mini AudioTransformer is a lightweight custom transformer architecture that processes log-mel spectrograms using patch embedding and transformer encoder blocks. Quick Eval trains a fresh model each time, Train & Save mode persists the model weights."
    # )

    mode = st.radio("Mode", ["Quick Eval", "Train & Save"], horizontal=True)

    if mode == "Train & Save":
        required_full = [
            "train_files", "val_files", "test_files",
            "y_train", "y_val", "y_test", "class_names"
        ]
        if not all(key in st.session_state for key in required_full):
            st.error("Prepare split first using the sidebar.")
            return

        st.subheader("Training Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            tf_epochs = st.slider("Epochs", min_value=1, max_value=30, value=10, step=1)
        with col2:
            tf_batch_size = st.selectbox("Batch size", options=[8, 16, 32, 64], index=2)
        with col3:
            tf_lr = st.selectbox(
                "Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3], index=2
            )

        save_dir = st.text_input("Save directory", value="artifacts/mini_audio_transformer_custom")

        if st.button("Train & Save Mini AudioTransformer", use_container_width=True):
            with st.spinner("Training Mini AudioTransformer..."):
                try:
                    results, history = train_and_save_transformer_baseline(
                        train_files=st.session_state["train_files"],
                        y_train=st.session_state["y_train"],
                        val_files=st.session_state["val_files"],
                        y_val=st.session_state["y_val"],
                        test_files=st.session_state["test_files"],
                        y_test=st.session_state["y_test"],
                        class_names=st.session_state["class_names"],
                        epochs=tf_epochs,
                        batch_size=int(tf_batch_size),
                        lr=float(tf_lr),
                        save_dir=save_dir,
                    )
                    st.session_state["transformer_test_results"] = results
                    st.session_state["transformer_history"] = history
                    st.success("✅ Mini AudioTransformer trained and saved!")
                except Exception as exc:
                    st.error(f"❌ Training failed: {exc!r}")

        if "transformer_history" in st.session_state:
            st.subheader("Training History")
            hist_df = pd.DataFrame(st.session_state["transformer_history"])
            st.line_chart(hist_df, use_container_width=True)

    else:  # Quick Eval
        required_full = [
            "train_files", "val_files", "test_files",
            "y_train", "y_val", "y_test", "class_names"
        ]
        if not all(key in st.session_state for key in required_full):
            st.error("Prepare split first using the sidebar.")
            return

        # if st.button("Run Transformer (Quick Eval)", use_container_width=True):
        #     with st.spinner("Training Mini AudioTransformer (quick eval)..."):
        #         try:
        #             results, history = run_transformer_baseline(
        #                 train_files=st.session_state["train_files"],
        #                 y_train=st.session_state["y_train"],
        #                 val_files=st.session_state["val_files"],
        #                 y_val=st.session_state["y_val"],
        #                 test_files=st.session_state["test_files"],
        #                 y_test=st.session_state["y_test"],
        #                 class_names=st.session_state["class_names"],
        #             )
        #             st.session_state["transformer_test_results"] = results
        #             st.session_state["transformer_history"] = history
        #             st.success("✅ Test set classified with Mini AudioTransformer!")
        #         except Exception as exc:
        #             st.error(f"❌ Mini AudioTransformer failed: {exc!r}")

        # st.markdown("---")
        st.subheader("Load Saved Model")
        artifacts_dir = st.text_input(
            "Artifacts directory",
            value="artifacts/mini_audio_transformer"
        )

        if st.button("Load Saved Mini AudioTransformer", use_container_width=True):
            with st.spinner("Classifying test set with saved Mini AudioTransformer..."):
                try:
                    results = run_saved_transformer_baseline(
                        test_files=st.session_state["test_files"],
                        y_test=st.session_state["y_test"],
                        class_names=st.session_state["class_names"],
                        artifacts_dir=artifacts_dir,
                    )
                    st.session_state["transformer_test_results"] = results
                    st.success("✅ Test set classified with saved Mini AudioTransformer!")
                except Exception as exc:
                    st.error(f"❌ Loading failed: {exc!r}")

    st.markdown("---")
    st.subheader("Results")

    if "transformer_test_results" in st.session_state:
        show_test_results(st.session_state["transformer_test_results"])
    else:
        st.info("Run the model to see results.")


# ============================================================================
# Page: CNN Baseline
# ============================================================================

def page_cnn():
    """Page for CNN baseline model."""
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Back to Home"):
            navigate_to("home")
            st.rerun()
    with col_title:
        st.title("CNN Baseline")

    st.markdown("---")

    required = [
        "train_files",
        "val_files",
        "test_files",
        "y_train",
        "y_val",
        "y_test",
        "class_names",
    ]
    if not all(key in st.session_state for key in required):
        st.error("Prepare split first using the sidebar.")
        return

    st.subheader("Configuration")
    # st.info(
    #     "CNN baseline trains a custom convolutional neural network on log-mel spectrograms. "
    #     "Quick Eval trains a fresh model each time, Train & Save mode persists the model weights."
    # )

    mode = st.radio("Mode", ["Quick Eval", "Train & Save"], horizontal=True)

    if mode == "Train & Save":
        st.subheader("Training Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            cnn_epochs = st.slider("Epochs", min_value=1, max_value=30, value=5, step=1)
        with col2:
            cnn_batch_size = st.selectbox("Batch size", options=[8, 16, 32, 64], index=2)
        with col3:
            cnn_lr = st.selectbox(
                "Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3], index=2
            )

        save_dir = st.text_input("Save directory", value="artifacts/cnn_baseline_custom")

        if st.button("Train & Save CNN", use_container_width=True):
            with st.spinner("Training CNN and saving weights..."):
                try:
                    cnn_results, cnn_history = train_and_save_cnn_baseline(
                        train_files=st.session_state["train_files"],
                        y_train=st.session_state["y_train"],
                        val_files=st.session_state["val_files"],
                        y_val=st.session_state["y_val"],
                        test_files=st.session_state["test_files"],
                        y_test=st.session_state["y_test"],
                        class_names=st.session_state["class_names"],
                        epochs=cnn_epochs,
                        batch_size=int(cnn_batch_size),
                        lr=float(cnn_lr),
                        save_dir=save_dir,
                    )
                    st.session_state["cnn_test_results"] = cnn_results
                    st.session_state["cnn_history"] = cnn_history
                    st.success("✅ CNN trained and weights saved!")
                except Exception as exc:
                    st.error(f"❌ CNN training failed: {exc!r}")

        if "cnn_history" in st.session_state:
            st.markdown("---")
            st.subheader("Training History")
            hist_df = pd.DataFrame(st.session_state["cnn_history"])
            st.line_chart(hist_df, use_container_width=True)

    else:  # Quick Eval
    #     if st.button("Run CNN (Quick Eval)", use_container_width=True):
    #         with st.spinner("Training CNN and evaluating test set..."):
    #             try:
    #                 cnn_results, cnn_history = run_cnn_baseline(
    #                     train_files=st.session_state["train_files"],
    #                     y_train=st.session_state["y_train"],
    #                     val_files=st.session_state["val_files"],
    #                     y_val=st.session_state["y_val"],
    #                     test_files=st.session_state["test_files"],
    #                     y_test=st.session_state["y_test"],
    #                     class_names=st.session_state["class_names"],
    #                     epochs=5,
    #                     batch_size=32,
    #                     lr=1e-3,
    #                 )
    #                 st.session_state["cnn_test_results"] = cnn_results
    #                 st.session_state["cnn_history"] = cnn_history
    #                 st.success("✅ CNN evaluation finished!")
    #             except Exception as exc:
    #                 st.error(f"❌ CNN run failed: {exc!r}")

        # Option to load saved model
        # st.markdown("---")
        st.subheader("Load Saved Model")
        load_dir = st.text_input("Load directory", value="artifacts/cnn")
        if st.button("Load & Evaluate Saved CNN", use_container_width=True):
            with st.spinner("Loading saved CNN and evaluating..."):
                try:
                    cnn_results = load_and_eval_cnn_baseline(
                        test_files=st.session_state["test_files"],
                        y_test=st.session_state["y_test"],
                        class_names=st.session_state["class_names"],
                        load_dir=load_dir,
                    )
                    st.session_state["cnn_test_results"] = cnn_results
                    st.success("✅ Loaded CNN evaluated successfully!")
                except Exception as exc:
                    st.error(f"❌ Loading failed: {exc!r}")

    st.markdown("---")
    st.subheader("Results")
    if "cnn_test_results" in st.session_state:
        show_test_results(st.session_state["cnn_test_results"])
    else:
        st.info("Run the model to see results.")


# ============================================================================
# Main Router
# ============================================================================

page = st.session_state.get("current_page", "home")

if page == "home":
    page_home()
elif page == "ast":
    page_ast()
elif page == "feature":
    page_feature()
elif page == "transformer":
    page_transformer()
elif page == "cnn":
    page_cnn()
else:
    st.error("Unknown page")

