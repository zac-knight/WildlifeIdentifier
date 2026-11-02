/**
 * Class Activation Map: the classifier's weights for one species applied at each spatial position.
 *
 * The EfficientNetV2 head is a global average pool followed by one linear layer, so the map's mean
 * plus the class bias equals the class logit exactly (tested in ml/tests/test_export.py). It shows
 * which regions raised this species' score. It does not show how the model reasoned about them.
 */
export function classActivationMap(
  features: ArrayLike<number>,
  channels: number,
  height: number,
  width: number,
  weights: ArrayLike<number>,
): Float32Array {
  if (features.length !== channels * height * width) throw new Error("Unexpected feature shape.");
  if (weights.length !== channels) throw new Error("Weights do not match feature channels.");
  const map = new Float32Array(height * width);
  for (let channel = 0; channel < channels; channel += 1) {
    const weight = weights[channel];
    const offset = channel * height * width;
    for (let i = 0; i < map.length; i += 1) map[i] += weight * features[offset + i];
  }
  return map;
}

/** Scale to 0-1 for display. Negative evidence is clipped because the overlay shows support only. */
export function normaliseMap(map: ArrayLike<number>): Float32Array {
  let max = 0;
  for (let i = 0; i < map.length; i += 1) max = Math.max(max, map[i]);
  const output = new Float32Array(map.length);
  if (max <= 0) return output;
  for (let i = 0; i < map.length; i += 1) output[i] = Math.max(0, map[i]) / max;
  return output;
}

/** A perceptually ordered ramp from transparent through amber to red. Returns RGBA pixels. */
export function heatmapPixels(map: ArrayLike<number>): Uint8ClampedArray {
  const pixels = new Uint8ClampedArray(map.length * 4);
  for (let i = 0; i < map.length; i += 1) {
    const value = map[i];
    pixels[i * 4] = 255;
    pixels[i * 4 + 1] = Math.round(220 * (1 - value));
    pixels[i * 4 + 2] = Math.round(60 * (1 - value));
    pixels[i * 4 + 3] = Math.round(255 * Math.min(1, value * 1.2));
  }
  return pixels;
}
