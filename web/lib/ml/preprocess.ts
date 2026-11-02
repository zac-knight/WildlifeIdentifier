/**
 * Reproduces the evaluation transform in ml/wildlens/imagedata.py: resize the short side to 224
 * with PIL's antialiased bilinear filter, centre-crop to 224, then normalise.
 *
 * Canvas scaling is not used. Browsers each choose their own filter, so canvas-resized images would
 * differ from what the model saw in validation, and accuracy results would no longer apply.
 */

export const INPUT_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const FIXED_BITS = 22;
const FIXED_ONE = 2 ** FIXED_BITS;
const FIXED_HALF = 2 ** (FIXED_BITS - 1);

interface Coefficients {
  starts: Int32Array;
  counts: Int32Array;
  weights: Float64Array;
  width: number;
}

/** The triangle filter widens when shrinking, so every source pixel contributes (antialiasing). */
function coefficients(source: number, target: number): Coefficients {
  const scale = source / target;
  const filterScale = Math.max(scale, 1);
  const support = filterScale; // The bilinear filter has a support of 1.
  const width = Math.ceil(support) * 2 + 1;
  const starts = new Int32Array(target);
  const counts = new Int32Array(target);
  const weights = new Float64Array(target * width);
  for (let i = 0; i < target; i += 1) {
    const centre = (i + 0.5) * scale;
    const start = Math.max(0, Math.trunc(centre - support + 0.5));
    const end = Math.min(source, Math.trunc(centre + support + 0.5));
    let total = 0;
    for (let x = start; x < end; x += 1) {
      const distance = Math.abs((x + 0.5 - centre) / filterScale);
      const weight = distance < 1 ? 1 - distance : 0;
      weights[i * width + (x - start)] = weight;
      total += weight;
    }
    for (let k = 0; k < end - start; k += 1) {
      // PIL stores weights as fixed-point integers, so identical rounding gives identical pixels.
      const value = (weights[i * width + k] / total) * FIXED_ONE;
      weights[i * width + k] = value < 0 ? Math.trunc(value - 0.5) : Math.trunc(value + 0.5);
    }
    starts[i] = start;
    counts[i] = end - start;
  }
  return { starts, counts, weights, width };
}

/** Convert a fixed-point sum back to an 8-bit value, rounding as PIL does. */
function toByte(fixedSum: number): number {
  const value = Math.floor((fixedSum + FIXED_HALF) / FIXED_ONE);
  return value < 0 ? 0 : value > 255 ? 255 : value;
}

/** Two-pass resize of RGBA pixels, returning interleaved RGB rounded to 8 bits as PIL does. */
export function resizeRgb(
  rgba: ArrayLike<number>,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
): Float32Array {
  const horizontal = coefficients(sourceWidth, targetWidth);
  const vertical = coefficients(sourceHeight, targetHeight);
  const middle = new Float32Array(targetWidth * sourceHeight * 3);
  for (let y = 0; y < sourceHeight; y += 1) {
    for (let x = 0; x < targetWidth; x += 1) {
      const base = (y * sourceWidth + horizontal.starts[x]) * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        let sum = 0;
        for (let k = 0; k < horizontal.counts[x]; k += 1) {
          sum += rgba[base + k * 4 + channel] * horizontal.weights[x * horizontal.width + k];
        }
        // PIL stores each pass in 8 bits, so the rounding must happen here too.
        middle[(y * targetWidth + x) * 3 + channel] = toByte(sum);
      }
    }
  }
  const output = new Float32Array(targetWidth * targetHeight * 3);
  for (let y = 0; y < targetHeight; y += 1) {
    for (let x = 0; x < targetWidth; x += 1) {
      for (let channel = 0; channel < 3; channel += 1) {
        let sum = 0;
        for (let k = 0; k < vertical.counts[y]; k += 1) {
          const row = vertical.starts[y] + k;
          sum +=
            middle[(row * targetWidth + x) * 3 + channel] *
            vertical.weights[y * vertical.width + k];
        }
        output[(y * targetWidth + x) * 3 + channel] = toByte(sum);
      }
    }
  }
  return output;
}

export function resizedDimensions(width: number, height: number): [number, number] {
  // torchvision truncates the long side when scaling so that the short side equals the target.
  if (width <= height) return [INPUT_SIZE, Math.trunc((INPUT_SIZE * height) / width)];
  return [Math.trunc((INPUT_SIZE * width) / height), INPUT_SIZE];
}

/** Returns a planar (3, 224, 224) tensor. Same-size input skips resampling, as PIL does. */
export function preprocess(rgba: ArrayLike<number>, width: number, height: number): Float32Array {
  const [resizedWidth, resizedHeight] = resizedDimensions(width, height);
  const resized = resizeRgb(rgba, width, height, resizedWidth, resizedHeight);
  // torchvision rounds the centre-crop offset to the nearest integer.
  const left = Math.round((resizedWidth - INPUT_SIZE) / 2);
  const top = Math.round((resizedHeight - INPUT_SIZE) / 2);
  const output = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  for (let y = 0; y < INPUT_SIZE; y += 1) {
    for (let x = 0; x < INPUT_SIZE; x += 1) {
      const from = ((y + top) * resizedWidth + x + left) * 3;
      for (let channel = 0; channel < 3; channel += 1) {
        output[channel * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x] =
          (resized[from + channel] / 255 - MEAN[channel]) / STD[channel];
      }
    }
  }
  return output;
}
