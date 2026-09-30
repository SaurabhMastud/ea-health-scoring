// Build-time access to the JSON bundle emitted by build_site_data.py.
//
// The site reads files, never a database or an API. Everything here runs during
// `astro build` and disappears from the shipped output.

import fs from 'node:fs';
import path from 'node:path';

const BUNDLE = path.resolve(process.cwd(), 'public/data/v1');
const SCHEMA_VERSION = '1.0';

export type Band = 'At Risk' | 'Needs Attention' | 'Healthy';

export interface GameIndexRow {
  appid: number;
  name: string;
  score: number;
  score_data_driven: number | null;
  band: Band;
  engagement: number | null;
  community: number | null;
  cohort: 'launched' | 'died_in_early_access';
  genres: string[];
  launch_date: string | null;
  n_ea_reviews: number | null;
  total_reviews: number | null;
  english_share: number | null;
  in_test_set: boolean | null;
  has_explanation: boolean;
  n_quality_flags: number;
}

export interface ShapEntry {
  feature: string;
  label: string;
  value: number;
  /** "hurts" = pushes the game towards at-risk. See the polarity note in build_site_data.py. */
  health_effect: 'hurts' | 'helps';
}

export interface TrajectoryPoint {
  bucket: number;
  period_start: string;
  n_reviews: number;
  toxicity_mean: number | null;
  sentiment_mean: number | null;
  persistence_median_hours: number | null;
  positive_share: number | null;
}

/**
 * What sits behind a game's toxicity score, as counts.
 *
 * The site may not carry review text — Steam's terms, and the build fails if it
 * does — so this is the shape of the evidence rather than the evidence itself.
 * The Streamlit draft reads the same tables and can show the text.
 */
export interface ToxicityEvidence {
  n_scored: number;
  bands: Array<{ label: string; from: number; to: number; n: number }>;
  categories: Array<{ category: string; n: number; share: number }>;
  n_above_half: number;
  censored_share: number;
  censored_runs_total: number;
}

export interface QualityFlag {
  code: string;
  label: string;
  detail: string;
}

export interface GameDetail extends Omit<GameIndexRow, 'has_explanation' | 'n_quality_flags'> {
  developers: string[];
  shap: ShapEntry[] | null;
  shap_unavailable_reason: string | null;
  trajectory: {
    bucket_days: number;
    ea_start: string;
    ea_end: string;
    points: TrajectoryPoint[];
  } | null;
  trajectory_unavailable_reason: string | null;
  comparables: Array<Pick<GameIndexRow, 'appid' | 'name' | 'score' | 'band' | 'n_ea_reviews'>>;
  toxicity_evidence: ToxicityEvidence | null;
  data_quality: QualityFlag[];
  measures: Record<string, number | null>;
}

export interface Meta {
  schema_version: string;
  generated_at: string;
  collection_date: string;
  corpus: {
    n_reviews_collected: number;
    n_games_collected: number;
    n_games_scored: number;
    n_launched: number;
    n_died_in_early_access: number;
    n_with_explanation: number;
    english_share_overall: number;
  };
  /** Measured properties of the collection itself, quoted by the "known gaps" list. */
  data_quality: {
    cap_incidence: number;
    cap_understatement_median: number;
    mendeley_join_share: number;
    release_date_wrong_share: number;
    release_date_median_error_days: number;
    release_date_n_comparable: number;
  };
  band_cutoffs: { at_risk_below: number; healthy_at_or_above: number; calibrated: boolean };
  frame: Record<string, string | number>;
  proxy_statement: string;
}

export interface ModelRun {
  run: string;
  outcome: string;
  n_features: number;
  test_auc_roc: number;
  test_auc_pr: number;
  test_brier: number;
  gain_over_baseline: number;
  gain_ci: [number | null, number | null];
  /** false = predictor and outcome share an instrument. Must be labelled wherever shown. */
  is_prediction: boolean;
  caveat: string | null;
}

export interface Model {
  schema_version: string;
  headline: {
    claim: string; auc: number; baseline_auc: number; gain: number;
    gain_ci: [number, number]; why_it_matters: string;
  };
  runs: ModelRun[];
  /** Spearman rho, early-access persistence against the post-launch outcome. */
  persistence_rho: number;

  negative_result: { claim: string; baseline_auc: number; full_model_auc: number; detail: string };
  bands: {
    at_risk_below: number; healthy_at_or_above: number; cutoff_source: string;
    test_precision: number; test_recall: number; test_base_rate: number;
  };
  survivorship: { rows: Array<{ cohort: string; n: number; median_health: number }>; detail: string };
  calibration: Array<{ predicted: number; observed: number }>;
  threshold_curve: Array<{ health_below: number; flagged: number; precision: number; recall: number }>;
  shap_importance: Array<{ feature: string; label: string; mean_abs_shap: number }>;
  limitations: Array<{ limitation: string; detail: string }>;
}

function read<T>(rel: string): T {
  return JSON.parse(fs.readFileSync(path.join(BUNDLE, rel), 'utf-8')) as T;
}

/** A major schema mismatch must stop the build, not render wrong numbers in a browser. */
function assertSchema(got: string, file: string) {
  if (got.split('.')[0] !== SCHEMA_VERSION.split('.')[0]) {
    throw new Error(
      `${file} is schema ${got}, this site expects ${SCHEMA_VERSION}. ` +
      `Re-run build_site_data.py or update the site — refusing to render mismatched data.`
    );
  }
}

export function getMeta(): Meta {
  const m = read<Meta>('meta.json');
  assertSchema(m.schema_version, 'meta.json');
  return m;
}

export function getModel(): Model {
  const m = read<Model>('model.json');
  assertSchema(m.schema_version, 'model.json');
  return m;
}

export function getGames(): GameIndexRow[] {
  return read<GameIndexRow[]>('games.json');
}

export function getGame(appid: number): GameDetail {
  const g = read<GameDetail & { schema_version: string }>(`games/${appid}.json`);
  assertSchema(g.schema_version, `games/${appid}.json`);
  return g;
}

// ---------------------------------------------------------------------------
// formatting. nulls are rendered as an explicit phrase, never as 0 — a missing
// measurement and a measurement of zero are different claims (brief §4)
// ---------------------------------------------------------------------------

export const NOT_ENOUGH_DATA = 'Not enough data';

export function num(v: number | null | undefined, digits = 1, suffix = ''): string {
  if (v === null || v === undefined || Number.isNaN(v)) return NOT_ENOUGH_DATA;
  return v.toLocaleString('en-GB', { minimumFractionDigits: digits, maximumFractionDigits: digits }) + suffix;
}

export function int(v: number | null | undefined): string {
  if (v === null || v === undefined) return NOT_ENOUGH_DATA;
  return Math.round(v).toLocaleString('en-GB');
}

export function pct(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined) return NOT_ENOUGH_DATA;
  return (v * 100).toFixed(digits) + '%';
}

/**
 * A 0–1 score with its percentage alongside: "0.752 (75.2%)".
 *
 * Only for measures that ARE proportions — AUC-ROC, AUC-PR, precision, recall,
 * a gain between two of them. Deliberately not used on the Brier score, which
 * is a mean squared error: rendering 0.159 as "15.9%" invites reading it as an
 * accuracy figure when a lower value is better.
 */
export function score(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return NOT_ENOUGH_DATA;
  return `${v.toFixed(digits)} (${(v * 100).toFixed(1)}%)`;
}

/** The percentage half alone, for places too tight for both. */
export function scorePct(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return NOT_ENOUGH_DATA;
  return `${(v * 100).toFixed(1)}%`;
}

export function hours(v: number | null | undefined): string {
  if (v === null || v === undefined) return NOT_ENOUGH_DATA;
  return v >= 100 ? `${Math.round(v).toLocaleString('en-GB')} h` : `${v.toFixed(1)} h`;
}

/** Band presentation is colour + icon + word, always. Never a bare coloured dot (Design.md §2.4). */
export const BAND_META: Record<Band, { icon: string; token: string; slug: string }> = {
  'At Risk':         { icon: '▲', token: 'var(--status-at-risk)',         slug: 'at-risk' },
  'Needs Attention': { icon: '●', token: 'var(--status-needs-attention)', slug: 'needs-attention' },
  'Healthy':         { icon: '✓', token: 'var(--status-healthy)',         slug: 'healthy' },
};

export function bandOf(score: number, meta: Meta): Band {
  if (score < meta.band_cutoffs.at_risk_below) return 'At Risk';
  if (score >= meta.band_cutoffs.healthy_at_or_above) return 'Healthy';
  return 'Needs Attention';
}
