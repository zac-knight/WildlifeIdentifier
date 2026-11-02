import { logSoftmax, softmax, topK } from "./math";

export interface FusionInput {
  /** Raw image logits, if a photo was supplied. */
  imageLogits?: ArrayLike<number>;
  imageTemperature?: number;
  /** Raw audio logits, if a recording was supplied. */
  audioLogits?: ArrayLike<number>;
  audioTemperature?: number;
  /** Presence score per species from the location model, if a place is known. */
  prior?: ArrayLike<number>;
  priorWeight?: number;
  priorFloor?: number;
  /** Fitted after fusion, because combining evidence changes confidence. */
  fusedTemperature?: number;
}

/**
 * score(y) = log p(y|image) + log p(y|audio) + weight * log max(prior(y), floor)
 *
 * Mirrors `fuse` in ml/wildlens/metrics.py. A shared fixture checks that the two agree.
 */
export function fuseScores(input: FusionInput): Float64Array {
  const parts: ArrayLike<number>[] = [];
  if (input.imageLogits) parts.push(logSoftmax(input.imageLogits, input.imageTemperature ?? 1));
  if (input.audioLogits) parts.push(logSoftmax(input.audioLogits, input.audioTemperature ?? 1));
  if (parts.length === 0) throw new Error("Fusion needs a photo or a recording.");
  const scores = new Float64Array(parts[0].length);
  for (const part of parts) {
    if (part.length !== scores.length) throw new Error("Model outputs have different lengths.");
    for (let i = 0; i < scores.length; i += 1) scores[i] += part[i];
  }
  if (input.prior) {
    const weight = input.priorWeight ?? 1;
    const floor = input.priorFloor ?? 1e-3;
    for (let i = 0; i < scores.length; i += 1) {
      scores[i] += weight * Math.log(Math.max(input.prior[i], floor));
    }
  }
  return scores;
}

export function fusedProbabilities(input: FusionInput): Float64Array {
  return softmax(fuseScores(input), input.fusedTemperature ?? 1);
}

export interface Thresholds {
  species: number;
  genus: number;
  family: number;
}

export type Decision =
  | { level: "species"; index: number; confidence: number }
  | { level: "genus" | "family"; name: string; confidence: number }
  | { level: "unknown"; confidence: number };

function groupTop(
  probabilities: ArrayLike<number>,
  groups: readonly string[],
): { name: string; total: number } {
  const totals = new Map<string, number>();
  for (let i = 0; i < probabilities.length; i += 1) {
    totals.set(groups[i], (totals.get(groups[i]) ?? 0) + probabilities[i]);
  }
  let best = { name: "", total: -1 };
  for (const [name, total] of totals) if (total > best.total) best = { name, total };
  return best;
}

/** Report the most specific level that clears its threshold, otherwise admit uncertainty. */
export function decide(
  probabilities: ArrayLike<number>,
  genera: readonly string[],
  families: readonly string[],
  thresholds: Thresholds,
): Decision {
  const [best] = topK(probabilities, 1);
  if (probabilities[best] >= thresholds.species) {
    return { level: "species", index: best, confidence: probabilities[best] };
  }
  const genus = groupTop(probabilities, genera);
  if (genus.total >= thresholds.genus) {
    return { level: "genus", name: genus.name, confidence: genus.total };
  }
  const family = groupTop(probabilities, families);
  if (family.total >= thresholds.family) {
    return { level: "family", name: family.name, confidence: family.total };
  }
  return { level: "unknown", confidence: probabilities[best] };
}

export interface RankChange {
  index: number;
  rankWithout: number;
  rankWith: number;
  /** Presence score at this place and season. */
  prior: number;
  effect: "boosted" | "lowered" | "unchanged";
}

/** Explains how location moved each shown species, for the "why this ranking" line in the UI. */
export function rankChanges(
  withoutPrior: ArrayLike<number>,
  withPrior: ArrayLike<number>,
  prior: ArrayLike<number>,
  shown: readonly number[],
): RankChange[] {
  const ranks = (scores: ArrayLike<number>) => {
    const order = topK(scores, scores.length);
    const lookup = new Map<number, number>();
    order.forEach((species, position) => lookup.set(species, position + 1));
    return lookup;
  };
  const before = ranks(withoutPrior);
  const after = ranks(withPrior);
  return shown.map((index) => {
    const rankWithout = before.get(index) ?? 0;
    const rankWith = after.get(index) ?? 0;
    const effect =
      rankWith < rankWithout ? "boosted" : rankWith > rankWithout ? "lowered" : "unchanged";
    return { index, rankWithout, rankWith, prior: prior[index], effect };
  });
}

/** Plain language, so a number is never mistaken for certainty. */
export function confidenceLabel(confidence: number): string {
  if (confidence >= 0.9) return "Very likely";
  if (confidence >= 0.7) return "Likely";
  if (confidence >= 0.4) return "Possibly";
  return "Unlikely";
}
