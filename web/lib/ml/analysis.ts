import {
  type Decision,
  decide,
  fusedProbabilities,
  type FusionInput,
  rankChanges,
  type RankChange,
} from "./fusion";
import { energyScore, softmax, topK } from "./math";
import type { AudioMetadata, ImageMetadata } from "./metadata";

export interface AnalysisInput {
  image: ImageMetadata;
  audio?: AudioMetadata;
  imageLogits?: ArrayLike<number>;
  audioLogits?: ArrayLike<number>;
  prior?: ArrayLike<number>;
}

export interface Candidate {
  index: number;
  name: string;
  probability: number;
  change?: RankChange;
}

export interface Analysis {
  decision: Decision;
  candidates: Candidate[];
  /** True when the photo looks unlike the species the model knows. */
  unfamiliar: boolean;
  usedLocation: boolean;
  probabilities: Float64Array;
}

export function analyse(input: AnalysisInput): Analysis {
  const { image, audio, imageLogits, audioLogits, prior } = input;
  const evidence: FusionInput = {
    imageLogits,
    imageTemperature: image.temperature,
    audioLogits,
    audioTemperature: audio?.temperature,
  };
  const usedLocation = Boolean(prior);
  const withPlace: FusionInput = prior
    ? {
        ...evidence,
        prior,
        priorWeight: image.priorWeight,
        priorFloor: image.priorFloor,
        fusedTemperature: image.fusedTemperature,
      }
    : evidence;
  const probabilities = fusedProbabilities(withPlace);
  const thresholds = usedLocation ? image.thresholds : image.noLocationThresholds;
  const decision = decide(probabilities, image.genera, image.families, thresholds);

  const shown = topK(probabilities, 5);
  const changes = prior
    ? rankChanges(fusedProbabilities(evidence), probabilities, prior, shown)
    : undefined;
  const candidates = shown.map((index, position) => ({
    index,
    name: image.classes[index],
    probability: probabilities[index],
    change: changes?.[position],
  }));
  // Energy applies to the photo model only. A recording of a known bird must not be rejected
  // because of a threshold fitted to photos.
  const unfamiliar = imageLogits
    ? energyScore(imageLogits, image.temperature) < image.energyThreshold
    : false;
  return { decision, candidates, unfamiliar, usedLocation, probabilities };
}

/** An uncertain result should prompt the user to look closer, never read as an identification. */
export function isConfident(analysis: Analysis): boolean {
  return analysis.decision.level === "species" && !analysis.unfamiliar;
}

export function explainChange(change: RankChange | undefined): string | null {
  if (!change) return null;
  if (change.effect === "boosted") {
    return `Moved up from #${change.rankWithout}: more expected here at this time of year.`;
  }
  if (change.effect === "lowered") {
    return `Moved down from #${change.rankWithout}: less expected here at this time of year.`;
  }
  return null;
}

export function imageOnlyProbabilities(
  logits: ArrayLike<number>,
  temperature: number,
): Float64Array {
  return softmax(logits, temperature);
}

export function dayOfYear(date: Date): number {
  const cumulative = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
  return cumulative[date.getMonth()] + date.getDate();
}

/** The models cover Australia only, so they should not be trusted for a place outside it. */
export function isInAustralia(latitude: number, longitude: number): boolean {
  return latitude >= -44.5 && latitude <= -9 && longitude >= 112 && longitude <= 155;
}

/** Feedback stores only an approximate place: finer for common species, coarser for threatened. */
export function coarsenForStorage(value: number, threatened: boolean): number {
  const step = threatened ? 0.5 : 0.1;
  return Math.round(value / step) * step;
}
