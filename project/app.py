"""Offline Streamlit UI for Customer Message Routing (BANKING77).

Starts with one command from repository root:
    streamlit run project/app.py

Does not send any messages or telemetry to any online service.
"""

from __future__ import annotations

import html
import os
import sys
import time

import pandas as pd
import streamlit as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from project.routing_model import RoutingBundle, URGENT_SET
from project.train_model import BUNDLE_PATH

st.set_page_config(
    page_title="Alfred Bank — Customer Message Router",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource
def load_bundle() -> RoutingBundle:
    """Load the local CPU RoutingBundle once per Streamlit session."""
    if not os.path.exists(BUNDLE_PATH):
        raise FileNotFoundError(
            f"Trained bundle not found at {BUNDLE_PATH}. Run `python -m project.train_model` first."
        )
    return RoutingBundle.load(BUNDLE_PATH)


def render_highlighted_words(word_scores: list[tuple[str, float]]) -> str:
    """Render inline HTML chips colored by leave-one-word-out logit contribution."""
    if not word_scores:
        return ""
    max_abs = max((abs(s) for _, s in word_scores), default=1.0) or 1.0
    chips: list[str] = []
    for token, score in word_scores:
        esc = html.escape(token)
        intensity = min(1.0, abs(score) / max_abs)
        if score > 0.05:
            alpha = 0.15 + 0.55 * intensity
            bg = f"rgba(34, 197, 94, {alpha:.2f})"
            border = "rgba(22, 163, 74, 0.7)"
        elif score < -0.05:
            alpha = 0.15 + 0.45 * intensity
            bg = f"rgba(239, 68, 68, {alpha:.2f})"
            border = "rgba(220, 38, 38, 0.6)"
        else:
            bg = "rgba(148, 163, 184, 0.15)"
            border = "rgba(148, 163, 184, 0.35)"
        chips.append(
            f"<span title='Logit delta: {score:+.3f}' style='"
            f"display:inline-block;padding:4px 8px;margin:3px;border-radius:6px;"
            f"background:{bg};border:1px solid {border};font-family:monospace;font-size:0.95rem;'>"
            f"{esc} <small style='opacity:0.75'>({score:+.2f})</small></span>"
        )
    return "".join(chips)


def main() -> None:
    st.title("Customer Message Router (Offline CPU)")
    st.caption(
        "Calibrated Word+Char TF-IDF Logistic Regression with cost-sensitive human review escalation. "
        "All inference and explanations run 100% locally."
    )

    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Model & Routing Controls")
        meta = bundle.metadata
        st.metric("5-Fold Group CV Accuracy", f"{meta.get('oof_accuracy', 0.0):.1%}")
        st.metric("Temperature (T)", f"{bundle.temperature:.3f}")
        st.metric("Calibrated OOF ECE", f"{meta.get('oof_ece_calibrated', 0.0):.4f}")

        st.divider()
        risk_thr = st.slider(
            "Human Review Risk Threshold",
            min_value=0.1,
            max_value=10.0,
            value=float(round(bundle.risk_threshold, 2)),
            step=0.1,
            help="Messages with expected misrouting risk >= threshold go to a person.",
        )
        ood_thr = st.slider(
            "OOD Cosine Similarity Floor",
            min_value=0.05,
            max_value=0.50,
            value=float(round(bundle.config.ood_sim_threshold, 2)),
            step=0.01,
            help="Messages below this similarity to all 77 training centroids are flagged OOD and sent to a person.",
        )

        st.divider()
        st.subheader("Quick Test Examples")
        examples = {
            "Normal Request": "I still have not received the new card you mailed two weeks ago.",
            "Urgent Safety Report": "Someone stole my wallet with my bank card inside at the station!",
            "Unfamiliar Wording (Urgent)": "There is a charge from a merchant I never visited on my statement.",
            "Ambiguous / Overlapping": "How long does a card delivery usually take?",
            "Out-of-Scope (OOD)": "The ceiling tiles in the office are leaking water onto the floor.",
        }
        for label, ex_text in examples.items():
            if st.button(label, use_container_width=True):
                st.session_state["msg_input"] = ex_text

    default_text = st.session_state.get(
        "msg_input",
        "I noticed a card payment from a store I never went to, please check!",
    )
    user_text = st.text_area("Customer message", value=default_text, height=110)

    if st.button("Route Message", type="primary") or user_text.strip():
        clean = user_text.strip()
        if not clean:
            st.warning("Please enter a customer message.")
            st.stop()

        t0 = time.perf_counter()
        result = bundle.predict_one(
            clean,
            threshold_override=risk_thr,
            ood_threshold_override=ood_thr,
            top_k=5,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Primary Decision Banner
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if result["send_to_person"]:
                st.error("**Routing:** SEND TO PERSON (REVIEW)")
            else:
                st.success(f"**Routing:** AUTO-QUEUE -> `{result['chosen_queue']}`")
        with col2:
            st.metric("Chosen Queue (Cost-Optimal)", result["chosen_queue"])
            if result["chosen_queue"] != result["top_prob_category"]:
                st.caption(f"Highest prob class: `{result['top_prob_category']}` (overridden by urgent safety cost)")
        with col3:
            st.metric(
                "Expected Misrouting Risk",
                f"{result['risk']:.2f}",
                delta=f"thr {risk_thr:.2f}",
                delta_color="inverse",
            )
        with col4:
            st.metric("CPU Latency (incl. Explain)", f"{elapsed_ms:.1f} ms")

        # Status Warnings
        reasons: list[str] = []
        if result["is_urgent"]:
            reasons.append(
                f"Urgent safety signal detected (urgent probability sum = {result['urgent_prob']:.1%})."
            )
        if result["risk"] >= risk_thr:
            reasons.append(
                f"Expected cost risk ({result['risk']:.2f}) meets or exceeds review threshold ({risk_thr:.2f})."
            )
        if result["is_ood"]:
            reasons.append(
                f"Out-of-distribution wording warning: max centroid similarity {result['max_centroid_sim']:.3f} < {ood_thr:.2f}."
            )
        if reasons:
            st.info("Escalation / Safety Notes:\n- " + "\n- ".join(reasons))

        left, right = st.columns([1.1, 1.0])

        with left:
            st.subheader("Top-5 Class Probabilities (Calibrated)")
            prob_df = pd.DataFrame(
                [
                    {
                        "Queue": cat,
                        "Probability": prob,
                        "Urgent": "YES" if cat in URGENT_SET else "",
                    }
                    for cat, prob in result["top_probs"]
                ]
            )
            st.dataframe(
                prob_df.style.format({"Probability": "{:.2%}"}),
                use_container_width=True,
                hide_index=True,
            )
            chart_df = prob_df.set_index("Queue")[["Probability"]]
            st.bar_chart(chart_df)

        with right:
            st.subheader(f"Words Driving `{result['chosen_queue']}`")
            st.caption("Green chips push the model toward this queue; red chips push away (exact leave-one-word-out logit delta).")
            st.markdown(render_highlighted_words(result["word_scores"]), unsafe_allow_html=True)

            sorted_words = sorted(result["word_scores"], key=lambda x: abs(x[1]), reverse=True)
            w_df = pd.DataFrame(sorted_words[:10], columns=["Word", "Logit Contribution"])
            st.dataframe(
                w_df.style.format({"Logit Contribution": "{:+.3f}"}),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
