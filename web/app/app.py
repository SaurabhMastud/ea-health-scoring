"""streamlit_app.py - the demo surface for the Early-Access Health Score.

reads the same JSON bundle the static site reads, so the two cannot disagree
about a score, a band cutoff or a colour. nothing here is recomputed: if a
number is on this screen it came out of the pipeline.

    streamlit run app/streamlit_app.py

no hex code is written in this file. every colour is read from the settings
file, which is the same file the notebooks and the site read (Design.md 6).
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import yaml

matplotlib.use("Agg")

# locating the bundle. this file runs from two layouts - the working repo and
# the submission package - so i look for the bundle rather than assume it.

HERE = Path(__file__).resolve().parent
CANDIDATE_BUNDLES = [
    HERE.parent / "web" / "public" / "data" / "v1",       # working repo: app/ sits beside web/
    HERE.parent / "website" / "dist" / "data" / "v1",     # package: web/app -> web/website/dist
    HERE.parent / "website" / "public" / "data" / "v1",   # package, before the site is built
]
CANDIDATE_SETTINGS = [
    HERE.parent / "config" / "settings.yaml",             # working repo
    HERE.parent.parent / "notebooks" / "config.yaml",     # package: web/app -> notebooks/
]


def first_existing(paths, what):
    for p in paths:
        if p.exists():
            return p
    st.error(f"could not find {what}. looked in:\n" + "\n".join(f"  {p}" for p in paths))
    st.stop()


BUNDLE = first_existing(CANDIDATE_BUNDLES, "the JSON data bundle")
SETTINGS_PATH = first_existing(CANDIDATE_SETTINGS, "the settings file")

# review-level tables. these hold the actual review text and exist ONLY locally -
# nothing here is ever written into the published bundle, and the site build has
# a guard that fails if review text reaches it. this app is the local draft
# surface, so it may read them; the website may not.
def find_dir(*rels):
    for base in (HERE.parent, HERE.parent.parent):
        for rel in rels:
            p = base.joinpath(*rel)
            if p.exists():
                return p
    return None


REVIEW_DIR = find_dir(("data", "interim", "reviews"),
                      ("data", "processed", "interim", "reviews"))
TOX_DIR = find_dir(("data", "interim", "toxicity"),
                   ("data", "processed", "interim", "toxicity"))


@st.cache_data
def load_bundle():
    read = lambda name: json.loads((BUNDLE / name).read_text(encoding="utf-8"))
    settings = yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8"))
    return read("meta.json"), read("model.json"), read("games.json"), settings


@st.cache_data
def load_game(appid):
    return json.loads((BUNDLE / "games" / f"{appid}.json").read_text(encoding="utf-8"))


@st.cache_data
def load_reviews(appid):
    """Review text joined to its scores, for one game. Local surface only."""
    import pandas as pd
    if REVIEW_DIR is None or TOX_DIR is None:
        return None
    rp, tp = REVIEW_DIR / f"{appid}.parquet", TOX_DIR / f"{appid}.parquet"
    if not (rp.exists() and tp.exists()):
        return None
    rev = pd.read_parquet(rp, columns=[
        "review_id", "review_text_clean", "review_date", "voted_up", "votes_up",
        "written_during_early_access", "playtime_at_review", "post_review_persistence"])
    tox = pd.read_parquet(tp, columns=[
        "review_id", "toxicity", "severe_toxicity", "obscene", "threat", "insult",
        "identity_attack", "sentiment_compound", "masked_profanity_runs"])
    df = rev.merge(tox, on="review_id", how="inner")
    df = df[df.review_text_clean.notna() & (df.review_text_clean.str.strip() != "")]
    return df


meta, model, games, settings = load_bundle()
palette = settings["palette"]

# dark mode is a selected set of steps, not a flipped one - settings.yaml carries
# a light and a dark value for every ink and series colour, so read the pair that
# matches the surface streamlit is actually painting on.
MODE = "dark" if (st.get_option("theme.base") or "light").lower() == "dark" else "light"
step = lambda d: d[MODE] if isinstance(d, dict) and MODE in d else d

# a band is colour + icon + word, never a bare colour. the amber sits below 3:1
# against the page, so the icon and the word are what carry it - not decoration.
BAND_STYLE = {
    "At Risk":         (palette["status"]["at_risk"],         "▲"),
    "Needs Attention": (palette["status"]["needs_attention"], "●"),
    "Healthy":         (palette["status"]["healthy"],         "✓"),
}
BLUE = step(palette["series"]["health_score"])
DIVERGING_LOW = step(palette["diverging"]["low"])     # helps
DIVERGING_HIGH = step(palette["diverging"]["high"])   # hurts
INK_SECONDARY = step(palette["ink"]["secondary"])
GRIDLINE = step(palette["ink"]["gridline"])
NOT_ENOUGH = "Not enough data"


def esc(s):
    """Review text is rendered inside HTML, so it has to be escaped first."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def score_pct(v, digits=3):
    """A 0-1 proportion with its percentage: 0.752 (75.2%). Not for Brier."""
    if v is None:
        return NOT_ENOUGH
    return f"{v:.{digits}f} ({v * 100:.1f}%)"


def fmt(v, digits=1, suffix=""):
    """a missing measurement and a measurement of zero are different claims."""
    return NOT_ENOUGH if v is None else f"{v:,.{digits}f}{suffix}"


def figure(height=2.4):
    fig, ax = plt.subplots(figsize=(7, height), dpi=140)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRIDLINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    return fig, ax


st.set_page_config(page_title="Early-Access Health Score", layout="wide")

# header

st.title("Early-Access Health Score")
st.caption(
    f"{meta['corpus']['n_games_scored']:,} Steam games scored from "
    f"{meta['corpus']['n_reviews_collected']:,} public early-access reviews. "
    f"Corpus collected {meta['collection_date']}."
)

with st.expander("What this score is, and what it is not", expanded=False):
    st.markdown(
        f"""
The score is a **rank among {meta['corpus']['n_games_scored']:,} games, not a calibrated
measurement**. It carries no confidence interval; interval estimation is named as future work
rather than quietly omitted. A score of 62 differs from 30 and does not differ from 65.

{meta['proxy_statement']}

Band cutoffs are **{meta['band_cutoffs']['at_risk_below']}** and
**{meta['band_cutoffs']['healthy_at_or_above']}**, derived on the training partition only and
never fitted on the games they are used to judge.

This demo reads the same JSON bundle as the static site, so the two cannot disagree.
        """
    )

tab_game, tab_model, tab_corpus = st.tabs(["A game", "Model performance", "The corpus"])

# per-game surface

with tab_game:
    by_name = {g["name"]: g for g in sorted(games, key=lambda g: g["name"].lower())}
    default = "Stormgate" if "Stormgate" in by_name else next(iter(by_name))
    chosen = st.selectbox("Game", list(by_name), index=list(by_name).index(default))
    g = load_game(by_name[chosen]["appid"])

    colour, icon = BAND_STYLE[g["band"]]
    left, right = st.columns([1, 2])

    with left:
        st.metric("Health score", f"{g['score']:.0f}", help="0-100. A rank, not a measurement.")
        st.markdown(
            f"<span style='color:{colour};font-weight:600'>{icon} {g['band']}</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Left early access" if g["cohort"] == "launched" else "Never left early access"
        )

    with right:
        st.markdown("**Components** — the two halves of the score, each 0-100.")
        if g["engagement"] is None or g["community"] is None:
            st.info(
                "This game never left early access, so it has no post-launch outcome to fit the "
                "component split against. It still receives an overall score, which is what the "
                "survivorship analysis compares across cohorts."
            )
        else:
            # magnitude on a common 0-100 scale -> one axis, one hue
            fig, ax = figure(1.5)
            names = ["Community", "Engagement"]
            vals = [g["community"], g["engagement"]]
            ax.barh(names, vals, color=BLUE, height=0.5)
            for y, v in enumerate(vals):
                ax.text(v + 1.5, y, f"{v:.0f}", va="center", fontsize=8, color=INK_SECONDARY)
            ax.set_xlim(0, 100)
            ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.6)
            ax.set_axisbelow(True)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.divider()

    m = g["measures"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Persistence, median", fmt(m.get("prp_median_hours"), 1, " h"),
              help="Median hours played AFTER writing a review. The retention proxy.")
    c2.metric("Never played again", fmt(
        None if m.get("pct_never_played_again") is None else m["pct_never_played_again"] * 100, 0, "%"))
    c3.metric("Review toxicity, mean", fmt(m.get("ea_toxicity_mean"), 3))
    c4.metric("Positive-review share", fmt(
        None if m.get("ea_positive_share") is None else m["ea_positive_share"] * 100, 0, "%"))

    # what is driving the score
    st.subheader("What is driving this score")
    if g["shap"]:
        # polarity, so the diverging pair: one hue each side of a neutral zero
        entries = list(reversed(g["shap"][:8]))
        fig, ax = figure(max(2.2, 0.34 * len(entries)))
        labels = [e["label"] for e in entries]
        vals = [e["value"] for e in entries]
        colours = [DIVERGING_HIGH if e["health_effect"] == "hurts" else DIVERGING_LOW
                   for e in entries]
        ax.barh(labels, vals, color=colours, height=0.6)
        ax.axvline(0, color=INK_SECONDARY, linewidth=1)
        ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("contribution to retention risk (log-odds)", fontsize=8,
                      color=INK_SECONDARY)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            f"Bars right of zero (<span style='color:{DIVERGING_HIGH}'>■</span> hurts) push the "
            f"game towards at-risk; left of zero "
            f"(<span style='color:{DIVERGING_LOW}'>■</span> helps) pushes it away.",
            unsafe_allow_html=True,
        )
    else:
        st.info(g["shap_unavailable_reason"] or "No per-game explanation for this game.")

    # trajectory
    st.subheader("Trajectory through early access")
    if g["trajectory"] and g["trajectory"]["points"]:
        pts = g["trajectory"]["points"]
        xs = [p["period_start"] for p in pts]

        # toxicity is 0-1 and persistence is hours. two scales never share an
        # axis, so these are two charts rather than one with a second y-axis.
        for key, label, digits in [
            ("toxicity_mean", "Review toxicity (mean, 0-1)", 3),
            ("persistence_median_hours", "Post-Review Persistence (median hours)", 1),
        ]:
            ys = [p[key] for p in pts]
            if all(v is None for v in ys):
                st.caption(f"{label}: {NOT_ENOUGH}")
                continue
            fig, ax = figure(1.9)
            # gaps stay gaps - a period with too few reviews is not a zero
            ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=3)
            ax.set_ylabel(label, fontsize=8, color=INK_SECONDARY)
            ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.6)
            ax.set_axisbelow(True)
            step = max(1, len(xs) // 6)
            ax.set_xticks(xs[::step])
            ax.set_xticklabels([x[:7] for x in xs[::step]], fontsize=7)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        st.caption(
            f"Each point is roughly {g['trajectory']['bucket_days']} days of reviews. "
            "Gaps are periods with too few reviews to average, not zeros."
        )
    else:
        st.info(g["trajectory_unavailable_reason"] or "No trajectory available for this game.")

    # caveats travel with the number
    if g["data_quality"]:
        st.subheader("Read this score with care")
        for f in g["data_quality"]:
            st.warning(f"**{f['label']}** — {f['detail']}")

    # what the community actually wrote
    # LOCAL ONLY. review text never enters the published bundle or the website;
    # the site build fails if it does. this app reads the review tables straight
    # off disk, which is why a studio can read the evidence here and not there.
    st.divider()
    st.subheader("What the community actually wrote")

    reviews = load_reviews(g["appid"])
    if reviews is None or reviews.empty:
        st.info(
            "Review text is not available for this game in this copy of the data. "
            "The scored tables are derived from it, but the text itself lives only in the "
            "local review layer."
        )
    else:
        st.caption(
            f"{len(reviews):,} scored reviews for this game. Toxicity is a model score from 0 to 1 "
            "over the review text. It is a proxy for how the community writes, not a judgement "
            "of any individual."
        )

        lo, hi = st.slider(
            "Select the toxicity score", 0.0, 1.0, (0.5, 1.0), 0.05,
            help="Filter reviews by their toxicity score. Drag the left handle down to see "
                 "milder criticism, or up to isolate the most hostile.",
        )
        window = reviews[(reviews.toxicity >= lo) & (reviews.toxicity <= hi)]

        a, b, c = st.columns(3)
        a.metric("Reviews in range", f"{len(window):,}")
        b.metric("Share of all reviews", f"{len(window) / len(reviews) * 100:.1f}%")
        c.metric("Mean toxicity in range", fmt(window.toxicity.mean() if len(window) else None, 3))

        # which categories fire in this range - this is what tells a studio the
        # difference between blunt language and something aimed at a person
        cats = ["insult", "obscene", "threat", "identity_attack", "severe_toxicity"]
        if len(window):
            shares = [(c_, float((window[c_] > 0.5).mean())) for c_ in cats]
            shares = [s for s in shares if s[1] > 0]
            if shares:
                fig, ax = figure(max(1.4, 0.42 * len(shares)))
                labels = [s[0].replace("_", " ") for s in shares][::-1]
                vals = [s[1] * 100 for s in shares][::-1]
                ax.barh(labels, vals, color=DIVERGING_HIGH, height=0.55)
                for y, v in enumerate(vals):
                    ax.text(v + 0.6, y, f"{v:.0f}%", va="center", fontsize=8, color=INK_SECONDARY)
                ax.set_xlim(0, max(max(vals) * 1.25, 5))
                ax.set_xlabel("share of reviews in range flagged above 0.5", fontsize=8,
                              color=INK_SECONDARY)
                ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.6)
                ax.set_axisbelow(True)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        left, right = st.columns(2)

        with left:
            st.markdown("**Driving the toxicity score**")
            worst = window.sort_values("toxicity", ascending=False).head(15)
            if worst.empty:
                st.caption("No reviews in the selected range.")
            for _, r in worst.iterrows():
                flags = [c_ for c_ in cats if r[c_] > 0.5]
                st.markdown(
                    f"<div style='border-left:3px solid {DIVERGING_HIGH};padding:0.15rem 0 0.15rem 0.7rem;"
                    f"margin-bottom:0.9rem'>"
                    f"<div style='font-size:0.78rem;color:{INK_SECONDARY}'>"
                    f"toxicity {r.toxicity:.3f}"
                    + (f" · {', '.join(flags)}" if flags else "")
                    + f" · {'early access' if r.written_during_early_access else 'post launch'}"
                    f" · {str(r.review_date)[:10]}</div>"
                    f"<div style='font-size:0.9rem'>{esc(str(r.review_text_clean))[:600]}</div></div>",
                    unsafe_allow_html=True,
                )

        with right:
            st.markdown("**What the game is praised for**")
            good = reviews[(reviews.voted_up) & (reviews.toxicity < 0.1)]
            good = good.sort_values(["votes_up", "sentiment_compound"], ascending=False).head(15)
            if good.empty:
                st.caption("No positive low-toxicity reviews for this game.")
            for _, r in good.iterrows():
                st.markdown(
                    f"<div style='border-left:3px solid {palette['status']['healthy']};"
                    f"padding:0.15rem 0 0.15rem 0.7rem;margin-bottom:0.9rem'>"
                    f"<div style='font-size:0.78rem;color:{INK_SECONDARY}'>"
                    f"sentiment {r.sentiment_compound:+.2f} · {int(r.votes_up)} found this helpful"
                    f" · {str(r.review_date)[:10]}</div>"
                    f"<div style='font-size:0.9rem'>{esc(str(r.review_text_clean))[:600]}</div></div>",
                    unsafe_allow_html=True,
                )

        st.caption(
            "Text is shown cleaned: Steam markup and links are stripped, and profanity Steam had "
            "already censored appears as its token. Author identifiers were hashed before this "
            "point and are not shown."
        )

# model performance

with tab_model:
    h = model["headline"]
    st.subheader("The headline result")
    st.success(h["claim"])
    a, b, c = st.columns(3)
    a.metric("Test AUC", score_pct(h["auc"]))
    b.metric("Free baseline", score_pct(h["baseline_auc"]))
    c.metric("Gain", "+" + score_pct(h["gain"]),
             help=f"95% CI [{h['gain_ci'][0]:.3f}, {h['gain_ci'][1]:.3f}]")
    st.caption(h["why_it_matters"])

    st.subheader("Every model run")
    st.markdown(
        "Rows marked **Association** share a measurement instrument between predictor and "
        "outcome. They are not predictive performance and are not reported as such."
    )
    st.dataframe(
        [
            {
                "Model": r["run"],
                "Outcome": r["outcome"],
                "Features": r["n_features"],
                "AUC-ROC": score_pct(r["test_auc_roc"], 4),
                "AUC-PR": score_pct(r["test_auc_pr"], 4),
                "Brier": round(r["test_brier"], 4),
                "Reading": "Prediction" if r["is_prediction"] else "Association",
            }
            for r in model["runs"]
        ],
        use_container_width=True,
        hide_index=True,
    )
    for r in model["runs"]:
        if r.get("caveat"):
            st.caption(f"**{r['run']}** — {r['caveat']}")

    neg = model["negative_result"]
    st.subheader("The negative result, reported as-is")
    st.error(
        f"{neg['claim']} Free baseline {score_pct(neg['baseline_auc'], 4)} beats the full model "
        f"{score_pct(neg['full_model_auc'], 4)}. {neg['detail']}"
    )

    st.subheader("Limitations")
    for lim in model["limitations"]:
        st.markdown(f"- **{lim['limitation']}** — {lim['detail']}")

# corpus

with tab_corpus:
    c = meta["corpus"]
    a, b, d, e = st.columns(4)
    a.metric("Reviews collected", f"{c['n_reviews_collected']:,}")
    b.metric("Games collected", f"{c['n_games_collected']:,}")
    d.metric("Games scored", f"{c['n_games_scored']:,}")
    e.metric("English share", f"{c['english_share_overall'] * 100:.1f}%")

    st.subheader("Survivorship")
    st.caption(model["survivorship"]["detail"])
    st.dataframe(
        [
            {"Cohort": r["cohort"], "Games": r["n"], "Median health": round(r["median_health"], 1)}
            for r in model["survivorship"]["rows"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Which signals move the model")
    top = list(reversed(model["shap_importance"][:10]))
    fig, ax = figure(max(2.4, 0.34 * len(top)))
    ax.barh([t["label"] for t in top], [t["mean_abs_shap"] for t in top],
            color=BLUE, height=0.6)
    ax.set_xlabel("mean |SHAP|", fontsize=8, color=INK_SECONDARY)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "Ranked by how much each signal moves the model overall, not by how much it moves any "
        "one game's score — that is what the per-game panel is for."
    )

st.divider()
st.caption(
    "Public data only. No human participants. No raw review text and no author identifier "
    "appears in this application or in the bundle it reads. Review toxicity is a proxy for the "
    "climate of a review community, never a measure of any individual's conduct."
)
