export function logSumExp(values: ArrayLike<number>): number {
  let max = -Infinity;
  for (let i = 0; i < values.length; i += 1) max = Math.max(max, values[i]);
  if (!Number.isFinite(max)) return max;
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) sum += Math.exp(values[i] - max);
  return max + Math.log(sum);
}

export function logSoftmax(logits: ArrayLike<number>, temperature = 1): Float64Array {
  const scaled = Float64Array.from({ length: logits.length }, (_, i) => logits[i] / temperature);
  const total = logSumExp(scaled);
  return scaled.map((value) => value - total);
}

export function softmax(logits: ArrayLike<number>, temperature = 1): Float64Array {
  return logSoftmax(logits, temperature).map(Math.exp);
}

/** Higher means more like the training species. Uses every logit, not only the largest. */
export function energyScore(logits: ArrayLike<number>, temperature = 1): number {
  const scaled = Float64Array.from({ length: logits.length }, (_, i) => logits[i] / temperature);
  return temperature * logSumExp(scaled);
}

export function topK(scores: ArrayLike<number>, k: number): number[] {
  return Array.from({ length: scores.length }, (_, i) => i)
    .sort((a, b) => scores[b] - scores[a])
    .slice(0, k);
}

export function entropy(probabilities: ArrayLike<number>): number {
  let total = 0;
  for (let i = 0; i < probabilities.length; i += 1) {
    if (probabilities[i] > 0) total -= probabilities[i] * Math.log(probabilities[i]);
  }
  return total;
}
