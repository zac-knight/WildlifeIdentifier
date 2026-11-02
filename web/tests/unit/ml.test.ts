import { describe, expect, test } from "vitest";
import { averageLogits, CLIP_SAMPLES, energyClips, toMono } from "@/lib/ml/audio";
import { classActivationMap, heatmapPixels, normaliseMap } from "@/lib/ml/cam";
import {
  confidenceLabel,
  decide,
  fusedProbabilities,
  fuseScores,
  rankChanges,
} from "@/lib/ml/fusion";
import { energyScore, entropy, logSoftmax, softmax, topK } from "@/lib/ml/math";
import { preprocess, resizedDimensions } from "@/lib/ml/preprocess";
import vectors from "../fixtures/python_vectors.json";

const { fusion, hierarchy, images } = vectors;

describe("agreement with the Python code", () => {
  test("fusion scores match", () => {
    const scores = fuseScores({
      imageLogits: fusion.logits,
      imageTemperature: fusion.temperature,
      audioLogits: fusion.audio,
      audioTemperature: fusion.audio_temperature,
      prior: fusion.prior,
      priorWeight: fusion.weight,
      priorFloor: fusion.floor,
    });
    fusion.fused.forEach((value, i) => expect(scores[i]).toBeCloseTo(value, 8));
  });

  test("fused probabilities match", () => {
    const probabilities = softmax(Float64Array.from(fusion.fused));
    fusion.probabilities.forEach((value, i) => expect(probabilities[i]).toBeCloseTo(value, 8));
  });

  test("energy score matches", () => {
    expect(energyScore(fusion.logits)).toBeCloseTo(fusion.energy, 8);
  });

  test.each(hierarchy.cases)("fallback decision: $level", (item) => {
    const thresholds = { species: 0.7, genus: 0.7, family: 0.7 };
    const decision = decide(item.probabilities, hierarchy.genera, hierarchy.families, thresholds);
    expect(decision.level).toBe(item.level);
    expect(decision.confidence).toBeCloseTo(item.confidence, 8);
    if (decision.level === "genus" || decision.level === "family") {
      expect(decision.name).toBe(item.group);
    }
    if (decision.level === "species") expect(decision.index).toBe(item.group);
  });

  test.each(images)("preprocessing matches for $width x $height", (item) => {
    const rgba = Uint8Array.from(atob(item.rgba_base64), (c) => c.charCodeAt(0));
    const output = preprocess(rgba, item.width, item.height);
    item.sampled_output.forEach((value, i) => {
      expect(Math.abs(output[i * item.sample_step] - value)).toBeLessThan(1e-3);
    });
  });
});

describe("math", () => {
  test("softmax sums to one and respects temperature", () => {
    const sharp = softmax([1, 2, 3]);
    const flat = softmax([1, 2, 3], 10);
    expect(sharp.reduce((a, b) => a + b)).toBeCloseTo(1, 10);
    expect(flat[2]).toBeLessThan(sharp[2]);
  });

  test("logSoftmax survives very large logits", () => {
    expect(logSoftmax([1000, 1000])[0]).toBeCloseTo(Math.log(0.5), 8);
  });

  test("topK returns highest first", () => {
    expect(topK([0.1, 0.5, 0.4], 2)).toEqual([1, 2]);
  });

  test("entropy is highest when uniform", () => {
    expect(entropy([0.5, 0.5])).toBeGreaterThan(entropy([0.99, 0.01]));
    expect(entropy([1, 0])).toBe(0);
  });
});

describe("fusion", () => {
  test("needs some evidence", () => {
    expect(() => fuseScores({ prior: [0.5] })).toThrow();
  });

  test("rejects mismatched model outputs", () => {
    expect(() => fuseScores({ imageLogits: [1, 2], audioLogits: [1, 2, 3] })).toThrow();
  });

  test("the floor stops a vagrant being ruled out", () => {
    const scores = fuseScores({
      imageLogits: [8, 0],
      prior: [1e-12, 0.9],
      priorWeight: 1,
      priorFloor: 1e-2,
    });
    expect(scores[0]).toBeGreaterThan(scores[1]);
  });

  test("fused temperature softens confidence", () => {
    const input = { imageLogits: [4, 0] };
    const sharp = fusedProbabilities(input);
    const soft = fusedProbabilities({ ...input, fusedTemperature: 3 });
    expect(soft[0]).toBeLessThan(sharp[0]);
  });

  test("reports how location moved a species", () => {
    const [first, second] = rankChanges([2, 1], [1, 2], [0.1, 0.9], [1, 0]);
    expect(first).toMatchObject({ index: 1, rankWithout: 2, rankWith: 1, effect: "boosted" });
    expect(second).toMatchObject({ index: 0, rankWithout: 1, rankWith: 2, effect: "lowered" });
  });

  test.each([
    [0.95, "Very likely"],
    [0.75, "Likely"],
    [0.5, "Possibly"],
    [0.1, "Unlikely"],
  ])("confidence %s reads as %s", (value, label) => {
    expect(confidenceLabel(value)).toBe(label);
  });
});

describe("preprocessing", () => {
  test("keeps the short side at the input size", () => {
    expect(resizedDimensions(500, 375)).toEqual([298, 224]);
    expect(resizedDimensions(375, 500)).toEqual([224, 298]);
  });

  test("a uniform image normalises to the expected value everywhere", () => {
    const rgba = new Uint8ClampedArray(64 * 64 * 4).fill(255);
    const output = preprocess(rgba, 64, 64);
    expect(output[0]).toBeCloseTo((1 - 0.485) / 0.229, 5);
    expect(output[output.length - 1]).toBeCloseTo((1 - 0.406) / 0.225, 5);
  });
});

describe("class activation maps", () => {
  test("combines channels using the class weights", () => {
    // Two channels, one row, two columns.
    const map = classActivationMap([1, 2, 3, 4], 2, 1, 2, [10, 1]);
    expect(Array.from(map)).toEqual([13, 24]);
  });

  test("rejects the wrong shape", () => {
    expect(() => classActivationMap([1, 2], 2, 2, 2, [1, 1])).toThrow();
  });

  test("normalising clips negative evidence", () => {
    expect(Array.from(normaliseMap([-2, 0, 2, 4]))).toEqual([0, 0, 0.5, 1]);
  });

  test("an all-negative map shows nothing", () => {
    expect(Array.from(normaliseMap([-2, -1]))).toEqual([0, 0]);
  });

  test("pixels are transparent where evidence is absent", () => {
    const pixels = heatmapPixels([0, 1]);
    expect(pixels[3]).toBe(0);
    expect(pixels[7]).toBe(255);
  });
});

describe("audio", () => {
  test("short recordings are padded", () => {
    const clips = energyClips(new Float32Array(100).fill(1), 3);
    expect(clips).toHaveLength(1);
    expect(clips[0]).toHaveLength(CLIP_SAMPLES);
    expect(clips[0][99]).toBe(1);
    expect(clips[0][100]).toBe(0);
  });

  test("the loudest part is chosen", () => {
    const wave = new Float32Array(CLIP_SAMPLES * 4);
    wave.fill(1, CLIP_SAMPLES * 2, CLIP_SAMPLES * 3);
    const [clip] = energyClips(wave, 1);
    expect(clip.reduce((a, b) => a + b) / clip.length).toBeGreaterThan(0.9);
  });

  test("chosen clips do not overlap", () => {
    const wave = new Float32Array(CLIP_SAMPLES * 6).map((_, i) => Math.sin(i));
    expect(energyClips(wave, 3)).toHaveLength(3);
  });

  test("logits average across clips", () => {
    expect(
      Array.from(
        averageLogits([
          [1, 3],
          [3, 5],
        ]),
      ),
    ).toEqual([2, 4]);
  });

  test("stereo mixes to mono", () => {
    expect(Array.from(toMono([Float32Array.of(1, 0), Float32Array.of(0, 1)]))).toEqual([0.5, 0.5]);
  });
});
