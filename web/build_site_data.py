"""build_site_data.py - the serving layer.

reads the pipeline's processed outputs and emits the versioned JSON bundle the
static site reads. deterministic, idempotent, one command, fails loudly.

"""

import json
import glob
import shutil
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

SCHEMA_VERSION = "1.0"

# this file lives in web/, so the project root is one level up
root = Path(__file__).parent.parent
out_dir = root/"web"/"website"/"public"/"data"/"v1"

# a trajectory bucket with fewer reviews than this reports null rather than a
# number. a mean toxicity over 3 reviews is noise wearing the costume of a signal
MIN_REVIEWS_PER_BUCKET = 10
MAX_TRAJECTORY_BUCKETS = 52
N_COMPARABLES = 5


def jsonable(v):
    """numpy and pandas types are not json serialisable, and NaN is not valid json.
    every null in this bundle is an explicit None so the site can render
    'not enough data' instead of a confident 0."""
    if v is None:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if pd.isna(v) else round(float(v), 6)
    if isinstance(v, (pd.Timestamp,)):
        return None if pd.isna(v) else v.strftime("%Y-%m-%d")
    if pd.isna(v):
        return None
    return v


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    # sort_keys so a rebuild on unchanged data produces a byte-identical file
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path.stat().st_size


print("BUILD SITE DATA")

settings = yaml.safe_load((root/"notebooks"/"config.yaml").read_text(encoding="utf-8"))
bands = settings["bands"]
print(f"band cutoffs from settings.yaml: at_risk<{bands['at_risk_below']} "
      f"healthy>={bands['healthy_at_or_above']} calibrated={bands['calibrated']}")

scores = pd.read_parquet(root/settings["paths"]["game_scores"])
features = pd.read_parquet(root/settings["paths"]["game_features"])
print(f"scores {scores.shape}  features {features.shape}")

# names and genres. appdetails is the good source; the screening csv is the
# fallback for anything appdetails missed
meta_by_appid = {}
for p in glob.glob(str(root/"data"/"raw"/"meta"/"*.json")):
    try:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        continue
    meta_by_appid[int(d["appid"])] = d

candidates = pd.read_csv(root/"data"/"processed"/"catalogue"/"candidate_games.csv")
name_fallback = dict(zip(candidates.appid, candidates.name))
total_reviews_reported = dict(zip(candidates.appid, candidates.total_reviews))
print(f"appdetails {len(meta_by_appid)}  screening rows {len(candidates)}")

missing_name = sum(1 for a in scores.appid if a not in meta_by_appid and a not in name_fallback)
if missing_name:
    print(f"  WARNING {missing_name} scored games have no name from either source")

feat = features.set_index("appid")

# every shap_ column maps back to the predictor it explains
shap_cols = [c for c in scores.columns if c.startswith("shap_")]
predictor_of = {c: c[len("shap_"):] for c in shap_cols}

# plain-language names, so the studio panel reads as English rather than as
# column names. anything unmapped falls back to the raw column
FEATURE_LABELS = {
    "pct_never_played_again": "Share who never played again after reviewing",
    "prp_median_hours": "Post-Review Persistence (median)",
    "prp_p75_hours": "Post-Review Persistence (top quartile)",
    "playtime_at_review_median_hours": "Playtime before reviewing (median)",
    "ea_duration_days": "Length of early access",
    "n_ea_reviews_collected": "Early-access review count",
    "ea_reviews_per_day_collected": "Review velocity during early access",
    "ea_positive_share": "Positive-review share",
    "ea_positive_share_trend": "Trend in positive-review share",
    "ea_toxicity_mean": "Average review toxicity",
    "ea_toxicity_share_above_half": "Share of reviews scored strongly toxic",
    "ea_sentiment_mean": "Average review sentiment",
    "ea_censored_rate": "Rate of Steam-censored profanity",
    "ea_toxicity_trend": "Trend in review toxicity",
    "ea_author_games_owned_median": "Reviewer library size (median)",
    "ea_author_review_count_median": "Reviewer review count (median)",
    "ea_share_edited": "Share of reviews later edited",
    "has_ea_positive_share_trend": "Positive-share trend measurable",
    "has_ea_toxicity_trend": "Toxicity trend measurable",
}


def data_quality(appid, row):
    """the flags that tell a reader how much to trust this game's score.
    a shaky score shown as a confident number is the failure mode here."""
    flags = []
    f = feat.loc[appid] if appid in feat.index else None

    if f is not None:
        if bool(f.get("cap_was_hit", False)):
            flags.append({
                "code": "cap_was_hit",
                "label": "Review collection hit its per-window cap",
                "detail": "Not every review was collected for this game. Volume figures come "
                          "from Steam's reported total, not from the collected rows.",
            })
        # english_share is stored 0-100 in game_features.parquet, NOT 0-1 as the
        # data dictionary states (measured: min 0.54, max 99.54, mean 54.42).
        # normalised to a fraction here so the bundle's contract is 0-1 throughout
        share = f.get("english_share")
        share = share/100 if pd.notna(share) else share
        if pd.notna(share) and share < 0.40:
            flags.append({
                "code": "low_english_share",
                "label": f"Only {share:.0%} of reviews are in English",
                "detail": "Toxicity and sentiment are measured on English text only, so this "
                          "score reflects a minority of this game's community.",
            })
        n_ea = f.get("n_ea_reviews_collected")
        if pd.notna(n_ea) and n_ea < 400:
            flags.append({
                "code": "small_sample",
                "label": f"Scored on {int(n_ea):,} early-access reviews",
                "detail": "Near the 200-review inclusion floor. Aggregates over a small sample "
                          "move a lot with a handful of reviews.",
            })
        if bool(f.get("flag_timestamp_disagreement", False)):
            flags.append({
                "code": "boundary_uncertainty",
                "label": "Early-access boundary is uncertain for this game",
                "detail": "Steam's own early-access flag and the review timestamps disagree more "
                          "than usual here, so the split between periods is less sharp.",
            })
    return flags


def trajectory(appid, launch_date):
    """toxicity, sentiment and persistence across the early-access period.

    aggregates only - this reads review-level files and emits bucket means, never
    text. buckets are equal-width over the game's own EA span, capped so a
    ten-year early access does not produce a 500-point line.
    """
    tox_path = root/settings["paths"]["toxicity"]/f"{appid}.parquet"
    rev_path = root/settings["paths"]["clean_reviews"]/f"{appid}.parquet"
    if not tox_path.exists() or not rev_path.exists():
        return None

    tox = pd.read_parquet(tox_path, columns=["review_id", "review_date", "toxicity", "sentiment_compound"])
    rev = pd.read_parquet(rev_path, columns=["review_id", "review_date", "post_review_persistence",
                                             "voted_up", "has_playtime"])

    # cohort split is on the timestamp against the bisected launch date, never on
    # steam's own flag - settings.yaml is explicit about this
    if launch_date is not None and pd.notna(launch_date):
        rev = rev[rev.review_date < launch_date]
        tox = tox[tox.review_date < launch_date]
    if len(rev) < MIN_REVIEWS_PER_BUCKET:
        return None

    df = rev.merge(tox.drop(columns=["review_date"]), on="review_id", how="left")
    start, end = df.review_date.min(), df.review_date.max()
    span_days = max((end - start).days, 1)

    n_buckets = min(MAX_TRAJECTORY_BUCKETS, max(1, span_days//7))
    bucket_days = span_days/n_buckets
    idx = ((df.review_date - start).dt.days//bucket_days).astype(int).clip(0, n_buckets - 1)

    points = []
    for b, g in df.groupby(idx):
        n = len(g)
        played = g[g.has_playtime]
        # persistence is stored in minutes, as steam returns it
        persistence = played.post_review_persistence.median()/60 if len(played) else np.nan
        thin = n < MIN_REVIEWS_PER_BUCKET
        points.append({
            "bucket": int(b),
            "period_start": (start + pd.Timedelta(days=bucket_days*b)).strftime("%Y-%m-%d"),
            "n_reviews": int(n),
            # a bucket too thin to mean anything reports null, not a number
            "toxicity_mean": None if thin else jsonable(g.toxicity.mean()),
            "sentiment_mean": None if thin else jsonable(g.sentiment_compound.mean()),
            "persistence_median_hours": None if thin else jsonable(persistence),
            "positive_share": None if thin else jsonable(g.voted_up.mean()),
        })

    return {
        "bucket_days": round(bucket_days, 1),
        "ea_start": start.strftime("%Y-%m-%d"),
        "ea_end": end.strftime("%Y-%m-%d"),
        "points": sorted(points, key=lambda p: p["bucket"]),
    }


def toxicity_evidence(appid, launch_date):
    """what is behind a game's toxicity score, as counts rather than text.

    the website may not carry review text (steam's terms, and the guard at the
    bottom of this file enforces it), so a studio gets the shape of the evidence
    instead: how many reviews sit in each toxicity band, which detoxify
    categories fire, and how much profanity steam had already censored. the
    streamlit draft reads the same tables and can show the text itself.
    """
    tox_path = root/settings["paths"]["toxicity"]/f"{appid}.parquet"
    if not tox_path.exists():
        return None
    cats = ["insult", "obscene", "threat", "identity_attack", "severe_toxicity"]
    tox = pd.read_parquet(tox_path, columns=["review_id", "review_date", "toxicity",
                                             "masked_profanity_runs"] + cats)
    if launch_date is not None and pd.notna(launch_date):
        tox = tox[tox.review_date < launch_date]
    n = len(tox)
    if n < MIN_REVIEWS_PER_BUCKET:
        return None

    # bands a reader can act on, not arbitrary deciles
    edges = [(0.0, 0.1, "negligible"), (0.1, 0.3, "mild"),
             (0.3, 0.5, "moderate"), (0.5, 0.8, "high"), (0.8, 1.01, "severe")]
    bands = [{"label": lbl, "from": lo, "to": min(hi, 1.0),
              "n": int(((tox.toxicity >= lo) & (tox.toxicity < hi)).sum())}
             for lo, hi, lbl in edges]

    # a category only counts when it clears the same 0.5 the bands use
    categories = []
    for c in cats:
        k = int((tox[c] > 0.5).sum())
        if k:
            categories.append({"category": c.replace("_", " "), "n": k, "share": k/n})
    categories.sort(key=lambda d: -d["n"])

    censored = tox.masked_profanity_runs.fillna(0)
    return {
        "n_scored": n,
        "bands": bands,
        "categories": categories,
        "n_above_half": int((tox.toxicity > 0.5).sum()),
        "censored_share": float((censored > 0).mean()),
        "censored_runs_total": int(censored.sum()),
    }


def shap_panel(row):
    """what is driving this score, biggest push first.

    only test-split games carry SHAP, so most games return an explicit reason
    rather than an empty panel that looks like a rendering bug.
    """
    values = {predictor_of[c]: row[c] for c in shap_cols if pd.notna(row[c])}
    if not values:
        return None
    ranked = sorted(values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [{
        "feature": k,
        "label": FEATURE_LABELS.get(k, k),
        "value": jsonable(v),
        # POLARITY, and do not flip this without re-running the check below it.
        # the model predicts retention_at_risk, where 1 means at risk, so a
        # POSITIVE shap value pushes a game TOWARDS at-risk and is bad for health.
        # measured on the 156 test games: corr(sum of shap, health_score) = -0.787,
        # corr(sum of shap, retention_at_risk) = +0.680, and games that actually
        # turned out at-risk average +1.65 against -1.45 for those that did not.
        # Design.md 2.3 pins blue = pushes score up, red = pushes score down, so a
        # positive shap value renders RED.
        "health_effect": "hurts" if v > 0 else "helps",
    } for k, v in ranked]


# the index the browse, search and leaderboard surfaces read
print("\nbuilding game index...")
index = []
for _, row in scores.iterrows():
    appid = int(row.appid)
    m = meta_by_appid.get(appid, {})
    f = feat.loc[appid] if appid in feat.index else None
    index.append({
        "appid": appid,
        "name": m.get("name") or name_fallback.get(appid) or f"App {appid}",
        "score": jsonable(row.health_score_equal),
        "score_data_driven": jsonable(row.health_score_data_driven),
        "band": row.health_band,
        "engagement": jsonable(row.engagement_component),
        "community": jsonable(row.community_component),
        "cohort": row.cohort,
        "genres": (m.get("genres") or [])[:3],
        "launch_date": jsonable(row.launch_date),
        "n_ea_reviews": jsonable(f.get("n_ea_reviews_collected") if f is not None else None),
        "total_reviews": jsonable(total_reviews_reported.get(appid)),
        # normalised 0-100 -> 0-1, see the note in data_quality()
        "english_share": jsonable((f.get("english_share")/100) if f is not None
                                  and pd.notna(f.get("english_share")) else None),
        "in_test_set": jsonable(row.in_test_set),
        "has_explanation": bool(shap_panel(row)),
        "n_quality_flags": len(data_quality(appid, row)),
    })

index.sort(key=lambda g: g["score"] if g["score"] is not None else -1, reverse=True)
size = write_json(out_dir/"games.json", index)
print(f"  games.json  {len(index)} games  {size/1024:.0f} KB")
if size > 1_000_000:
    print("  WARNING index over 1 MB - trim fields or split (brief 7.3)")

# per-game detail, lazy-loaded one request at a time
print("\nbuilding per-game detail (reads review-level files, this is the slow part)...")
by_appid = {g["appid"]: g for g in index}
detail_dir = out_dir/"games"
if detail_dir.exists():
    shutil.rmtree(detail_dir)

# counted off the index rather than written in, because the literal that used to
# sit in the reason string below said "156 of 920" long after the corpus grew to
# 164 of 969 - and it was on 805 pages before anyone read one
n_explained = sum(g["has_explanation"] for g in index)

n_traj = n_shap = n_evid = 0
for i, (_, row) in enumerate(scores.iterrows(), 1):
    appid = int(row.appid)
    entry = by_appid[appid]
    m = meta_by_appid.get(appid, {})
    f = feat.loc[appid] if appid in feat.index else None

    shap = shap_panel(row)
    traj = trajectory(appid, row.launch_date)
    evid = toxicity_evidence(appid, row.launch_date)
    n_shap += shap is not None
    n_traj += traj is not None
    n_evid += evid is not None

    # nearest games by size, sharing a genre. benchmarking is the first thing a
    # founder asks for and it is free from the index we already built
    genres = set(entry["genres"])
    mine = entry["n_ea_reviews"] or 0
    peers = [g for g in index
             if g["appid"] != appid and genres & set(g["genres"]) and g["score"] is not None]
    peers.sort(key=lambda g: abs((g["n_ea_reviews"] or 0) - mine))
    comparables = [{k: g[k] for k in ("appid", "name", "score", "band", "n_ea_reviews")}
                   for g in peers[:N_COMPARABLES]]

    write_json(detail_dir/f"{appid}.json", {
        "schema_version": SCHEMA_VERSION,
        "appid": appid,
        "name": entry["name"],
        # `or []` not `.get(k, [])` - appdetails sometimes carries an explicit
        # null (appid 509580 has developers: null), and a default only applies
        # when the key is absent. the contract promises a list, so coerce here
        # once rather than defending against null in every template
        "developers": m.get("developers") or [],
        "genres": m.get("genres") or [],
        "score": entry["score"],
        "score_data_driven": entry["score_data_driven"],
        "band": entry["band"],
        "engagement": entry["engagement"],
        "community": entry["community"],
        "cohort": entry["cohort"],
        "launch_date": entry["launch_date"],
        "n_ea_reviews": entry["n_ea_reviews"],
        "total_reviews": entry["total_reviews"],
        "english_share": entry["english_share"],
        "in_test_set": entry["in_test_set"],
        "shap": shap,
        # a null panel needs to say why, or it reads as a broken page
        "shap_unavailable_reason": None if shap else (
            f"Per-game explanations are produced for the temporal test split only "
            f"({n_explained} of {len(index)} games). This game was used to fit the "
            f"model, so attributing its own score back to it would be circular."),
        "trajectory": traj,
        "toxicity_evidence": evid,
        "trajectory_unavailable_reason": None if traj else (
            "Not enough early-access reviews with dates to plot a trend."),
        "comparables": comparables,
        "data_quality": data_quality(appid, row),
        "measures": {
            "prp_median_hours": jsonable(row.prp_median_hours),
            "ea_toxicity_mean": jsonable(row.ea_toxicity_mean),
            "ea_positive_share": jsonable(row.ea_positive_share),
            "pct_never_played_again": jsonable(f.get("pct_never_played_again") if f is not None else None),
        },
    })
    if i % 200 == 0:
        print(f"  {i}/{len(scores)}")

print(f"  {len(scores)} detail files | trajectory on {n_traj} | explanation on {n_shap} | evidence on {n_evid}")

# model.json - the honesty layer. this is the file that stops the site reading
# as marketing
print("\nbuilding model.json...")


def read_table(name):
    return pd.read_csv(root/"data"/"processed"/"tables"/f"{name}.csv", encoding="utf-8")


comparison = read_table("model_comparison")
runs = []
for _, r in comparison.iterrows():
    label = str(r.iloc[0])
    outcome = label.split()[0]
    # the two same-instrument outcomes are association, not forecasting, and the
    # site is required to label them wherever they appear
    cross = "cross-instrument" in label
    runs.append({
        "run": label,
        "outcome": outcome,
        "n_features": jsonable(r.n_features),
        "test_auc_roc": jsonable(r.test_auc_roc),
        "test_auc_pr": jsonable(r.test_auc_pr),
        "test_brier": jsonable(r.test_brier),
        "gain_over_baseline": jsonable(r.gain_over_baseline),
        "gain_ci": [jsonable(r.gain_ci_low), jsonable(r.gain_ci_high)],
        "is_prediction": bool(cross),
        "caveat": None if cross else (
            "Predictor and outcome share a measurement instrument, so this figure "
            "measures trait persistence rather than forecasting."),
    })

survivorship = read_table("survivorship_cohort")
band_perf = read_table("band_performance")
limits = read_table("limitations")


def at_risk_share(cohort):
    sub = scores[scores.cohort == cohort]
    return float((sub.health_band == "At Risk").mean())

# the headline and the negative result were literals here, and the runs table beside them
# was generated - so a re-run left the hero quoting one corpus and the table below it
# another. both now come off the same csvs the rest of this file reads.
xi = read_table("cross_instrument_model").set_index("model").test_auc_roc
comp = comparison.set_index(comparison.columns[0]).test_auc_roc


def run_auc(*parts):
    """model_comparison labels carry a middle dot, so match on substrings instead of
    reproducing it. Insisting on exactly one hit is the point - a label that starts
    matching two rows is a silently wrong number otherwise."""
    hits = [i for i in comp.index if all(p in i for p in parts)]
    if len(hits) != 1:
        raise SystemExit(f"model_comparison: {len(hits)} rows match {parts}, expected 1")
    return float(comp[hits[0]])

model = {
    "schema_version": SCHEMA_VERSION,
    "headline": {
        "claim": "Early-access community climate - toxicity and sentiment measured from review "
                 "text - predicts post-launch player retention measured from playtime.",
        "auc": float(xi["cross-instrument (community text)"]),
        "baseline_auc": float(xi["free baseline"]),
        "gain": float(xi["GAIN over baseline (bootstrap mean)"]),
        "gain_ci": [float(xi["gain 95% CI low"]), float(xi["gain 95% CI high"])],
        "why_it_matters": "Predictor and outcome come from different instruments - text in, "
                          "playtime out - so this result cannot be explained by the two sharing "
                          "a measurement.",
    },
    "runs": runs,
    # the methodology page quotes this when it explains why persistence is association and
    # not prediction. it was a literal on that page.
    "persistence_rho": float(read_table("predictor_outcome_correlations")
                             .set_index("pair").spearman_rho["prp_median_hours vs post_prp_median_hours"]),
    "negative_result": {
        "claim": "On post-launch satisfaction the free baseline beats the full model.",
        "baseline_auc": run_auc("satisfaction", "review-% baseline"),
        "full_model_auc": run_auc("satisfaction", "full features"),
        "detail": "Positive-review share alone predicts future positive-review share better than "
                  "19 engineered features do. Reported because it is true, not because it helps.",
    },
    "bands": {
        "at_risk_below": bands["at_risk_below"],
        "healthy_at_or_above": bands["healthy_at_or_above"],
        "cutoff_source": str(band_perf.cutoff_source.iloc[0]),
        "test_precision": jsonable(band_perf.test_precision.iloc[0]),
        "test_recall": jsonable(band_perf.test_recall.iloc[0]),
        "test_base_rate": jsonable(band_perf.test_base_rate.iloc[0]),
    },
    "survivorship": {
        "rows": [{"cohort": r.cohort, "n": jsonable(r.n),
                  "median_health": jsonable(r.median_health)} for _, r in survivorship.iterrows()],
        # the two rates were written into this sentence by hand and went stale while the
        # cohort counts printed directly above them stayed correct - the page contradicted
        # itself. they are counted off the scored games now.
        "detail": "Games that died in early access score lower than games that launched. The band "
                  "cutoffs were fitted only on launched training games and never saw this cohort, "
                  f"yet they flag {at_risk_share('died_in_early_access'):.1%} of died-in-EA games "
                  f"as At Risk against {at_risk_share('launched'):.1%} of launched games.",
    },
    "calibration": [{"predicted": jsonable(r.predicted_risk), "observed": jsonable(r.observed_rate)}
                    for _, r in read_table("calibration_cross_instrument").iterrows()],
    "threshold_curve": [{"health_below": jsonable(r.health_below), "flagged": jsonable(r.games_flagged),
                         "precision": jsonable(r.precision), "recall": jsonable(r.recall)}
                        for _, r in read_table("threshold_precision_recall").iterrows()],
    "shap_importance": [{"feature": str(r.iloc[0]), "label": FEATURE_LABELS.get(str(r.iloc[0]), str(r.iloc[0])),
                         "mean_abs_shap": jsonable(r.mean_abs_shap)}
                        for _, r in read_table("shap_importance").iterrows()],
    "limitations": [{"limitation": r.limitation, "detail": r.detail} for _, r in limits.iterrows()],
}
write_json(out_dir/"model.json", model)
print(f"  {len(runs)} runs, {len(model['limitations'])} limitations")

# meta.json - provenance, and the cutoffs the UI reads instead of hardcoding
n_scored = len(scores)
launched = int((scores.cohort == "launched").sum())
died = int((scores.cohort == "died_in_early_access").sum())

# corpus totals come off disk, not out of a literal. they were hardcoded here once and
# a later collection round left the site quoting a corpus that no longer existed.
# n_reviews_collected counts rows as collected, so it reads the checkpoints rather than
# the cleaned features table - cleaning drops a couple of thousand rows and the two are
# not the same quantity.
n_collected = len(features)

checkpoints = []
for p in glob.glob(str(root/settings["paths"]["checkpoints"]/"*.json")):
    try:
        checkpoints.append(json.loads(Path(p).read_text(encoding="utf-8")))
    except (json.JSONDecodeError, UnicodeDecodeError):
        continue
# only the games that actually produced a review file - a checkpoint also exists for
# every screened-out title, and those carry no rows
cp = pd.DataFrame(checkpoints)
cp = cp[cp.appid.isin(set(features.appid))]
n_reviews_collected = int(cp.n_reviews.sum())

all_rows = features["n_reviews_all_languages"].sum()
english_share = float((features["english_share"]/100 * features["n_reviews_all_languages"]).sum()
                      / all_rows)

# the data-quality findings the "known gaps" list quotes. these were four literals typed
# into data.astro and methodology.astro, measured on an earlier corpus, and nothing about
# a re-run could correct them.
capped = cp[cp.cap_was_hit & (cp.n_reviews > 0)]
delta = cp.listing_vs_launch_delta_days.abs().dropna()   # defined only where a launch was derived
wrong = delta > 7
mendeley = json.loads((root/"data"/"raw"/"games.json").read_text(encoding="utf-8"))
joined = len(set(features.appid) & {int(k) for k in mendeley})

data_quality = {
    "cap_incidence": float(cp.cap_was_hit.mean()),
    "cap_understatement_median": float((capped.total_reviews/capped.n_reviews).median()),
    "mendeley_join_share": joined/len(features),
    "release_date_wrong_share": float(wrong.mean()),
    "release_date_median_error_days": float(delta[wrong].median()),
    "release_date_n_comparable": int(len(delta)),
}

meta = {
    "schema_version": SCHEMA_VERSION,
    "generated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "collection_date": str(settings["collection_date"]),
    "corpus": {
        "n_reviews_collected": n_reviews_collected,
        "n_games_collected": n_collected,
        "n_games_scored": n_scored,
        "n_launched": launched,
        "n_died_in_early_access": died,
        "n_with_explanation": n_shap,
        "english_share_overall": round(english_share, 3),
    },
    "data_quality": {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in data_quality.items()},
    "band_cutoffs": {
        "at_risk_below": bands["at_risk_below"],
        "healthy_at_or_above": bands["healthy_at_or_above"],
        "calibrated": bands["calibrated"],
    },
    "frame": {
        "min_ea_reviews": settings["frame"]["min_ea_reviews"],
        "min_post_launch_reviews": settings["frame"]["min_post_launch_reviews"],
        "post_launch_window_days": settings["frame"]["post_launch_window_days"],
        "split": settings["split"]["method"],
    },
    "proxy_statement": "The Health Score is built from public Steam review text and playtime. "
                       "Toxicity is a community-climate proxy measured from what reviewers wrote - "
                       "it is not a measure of in-game player behaviour, and it is not a judgement "
                       "of the developers.",
}
write_json(out_dir/"meta.json", meta)

# design tokens. generated from the same settings.yaml the notebooks read, so a
# hex literal in the site's css is not just discouraged, it is unnecessary -
# Design.md 6 calls one a bug and this is what makes that enforceable
print("\ngenerating design tokens...")
p = settings["palette"]


def both_modes(entry):
    """settings.yaml carries either a plain hex or a {light, dark} pair."""
    if isinstance(entry, dict):
        return entry["light"], entry["dark"]
    return entry, entry


light_vars, dark_vars = {}, {}
for name, val in p["series"].items():
    lo, dk = both_modes(val)
    light_vars[f"--series-{name.replace('_','-')}"] = lo
    dark_vars[f"--series-{name.replace('_','-')}"] = dk
for name, val in p["status"].items():
    light_vars[f"--status-{name.replace('_','-')}"] = val
    dark_vars[f"--status-{name.replace('_','-')}"] = val
for name, val in p["ink"].items():
    lo, dk = both_modes(val)
    light_vars[f"--ink-{name.replace('_','-')}"] = lo
    dark_vars[f"--ink-{name.replace('_','-')}"] = dk
for i, step in enumerate(p["sequential_blue"]):
    light_vars[f"--seq-{i}"] = step
    dark_vars[f"--seq-{i}"] = step
for k in ("low", "high"):
    light_vars[f"--diverging-{k}"] = p["diverging"][k]
    dark_vars[f"--diverging-{k}"] = p["diverging"][k]
lo, dk = both_modes(p["diverging"]["mid"])
light_vars["--diverging-mid"], dark_vars["--diverging-mid"] = lo, dk


def block(vars_):
    return "\n".join(f"  {k}: {v};" for k, v in vars_.items())


tokens_css = f"""/* GENERATED by build_site_data.py from config/settings.yaml - do not edit.
   Every colour in this site traces to that one file (Design.md 6). A hex literal
   anywhere else in the site is a bug. Re-run the build to regenerate. */

:root {{
{block(light_vars)}

  /* type scale - Design.md 3 */
  --font-sans: system-ui, -apple-system, "Segoe UI", sans-serif;
  --size-hero: 64px;
  --size-stat: 32px;
  --size-title: 16px;
  --size-subtitle: 13px;
  --size-tick: 12px;
  --size-body: 15px;
  --size-footnote: 11px;
}}

/* dark is a selected set of steps validated against the dark surface, never an
   inversion. prefers-color-scheme is the default signal; the explicit toggle wins */
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
{block(dark_vars)}
  }}
}}

:root[data-theme="dark"] {{
{block(dark_vars)}
}}

:root[data-theme="light"] {{
{block(light_vars)}
}}
"""
tokens_path = root/"web"/"website"/"src"/"styles"/"tokens.css"
tokens_path.parent.mkdir(parents=True, exist_ok=True)
tokens_path.write_text(tokens_css, encoding="utf-8")
print(f"  tokens.css  {len(light_vars)} tokens x 2 modes")

print("VERIFICATION")

# fail loudly rather than shipping a bundle the site renders as NaN
errors = []
if len(index) != n_scored:
    errors.append(f"index has {len(index)} rows, expected {n_scored}")
if any(g["score"] is None for g in index):
    errors.append("a game in the index has no score")
if any(g["band"] not in ("At Risk", "Needs Attention", "Healthy") for g in index):
    errors.append("unknown band value in the index")

# range checks. english_share arrived as 0-100 while the data dictionary called it
# 0-1, which rendered as "8815% of reviews in English" before it was caught. a
# unit change upstream is silent unless something asserts the range
for field, lo, hi in [("score", 0, 100), ("engagement", 0, 100),
                      ("community", 0, 100), ("english_share", 0, 1)]:
    vals = [g[field] for g in index if g.get(field) is not None]
    if vals and not (lo <= min(vals) and max(vals) <= hi):
        errors.append(f"{field} out of range [{lo},{hi}]: "
                      f"observed [{min(vals):.4g}, {max(vals):.4g}]")
if len(list(detail_dir.glob("*.json"))) != n_scored:
    errors.append("detail file count does not match the index")

# the one that would breach steam's terms if it ever failed. this scans EVERY
# file including the 920 per-game details - the details are where a leak would
# actually live, since they are the only files built from review-level input
# these four never occur in legitimate prose, so a hit is a real leak. field
# names like playtime_forever are deliberately NOT here - they appear in the
# limitations text as methodology description and would only ever false-positive
text_cols = ("review_text", "review_id", "author_hash", "steamid")
for p in out_dir.rglob("*.json"):
    blob = p.read_text(encoding="utf-8")
    for c in text_cols:
        if c in blob:
            errors.append(f"{p.name} contains '{c}' - raw review data must never ship")

# shap polarity regression check. the explanation panel colours every driver off
# this sign, so an inversion here silently tells a studio the opposite of the
# truth. positive shap must mean worse health
check = scores[scores[shap_cols[0]].notna()]
if len(check) > 30:
    corr = check[shap_cols].sum(axis=1).corr(check.health_score_equal)
    if corr > -0.3:
        errors.append(f"shap polarity check failed: corr(sum shap, health) = {corr:.3f}, "
                      f"expected strongly negative. the 'hurts'/'helps' mapping is wrong")
    else:
        print(f"shap polarity   corr(sum shap, health) = {corr:.3f} - positive shap = worse health, as expected")

bundle_bytes = sum(p.stat().st_size for p in out_dir.rglob("*.json"))
print(f"bundle            {bundle_bytes/1024/1024:.1f} MB across {len(list(out_dir.rglob('*.json')))} files")
print(f"index            {len(index)} games ({launched} launched, {died} died in EA)")
print(f"trajectory        {n_traj}/{n_scored} games")
print(f"toxicity evidence {n_evid}/{n_scored} games")
print(f"explanation       {n_shap}/{n_scored} games (test split only)")
print(f"bands             At Risk {sum(g['band']=='At Risk' for g in index)} | "
      f"Needs Attention {sum(g['band']=='Needs Attention' for g in index)} | "
      f"Healthy {sum(g['band']=='Healthy' for g in index)}")

if errors:
    print("\nFAILED:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
print("\nall checks passed")
