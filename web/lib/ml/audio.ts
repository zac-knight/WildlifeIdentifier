export const AUDIO_SAMPLE_RATE = 32_000;
export const CLIP_SAMPLES = AUDIO_SAMPLE_RATE * 5;

/**
 * Choose the loudest non-overlapping five-second windows, matching `energy_clips` in
 * ml/wildlens/audio.py. Short recordings are zero-padded.
 */
export function energyClips(waveform: Float32Array, count: number): Float32Array[] {
  if (waveform.length <= CLIP_SAMPLES) {
    const padded = new Float32Array(CLIP_SAMPLES);
    padded.set(waveform);
    return [padded];
  }
  const step = Math.floor(CLIP_SAMPLES / 2);
  const cumulative = new Float64Array(waveform.length + 1);
  for (let i = 0; i < waveform.length; i += 1) {
    cumulative[i + 1] = cumulative[i] + waveform[i] * waveform[i];
  }
  const windows: { start: number; energy: number }[] = [];
  for (let start = 0; start + CLIP_SAMPLES <= waveform.length; start += step) {
    windows.push({ start, energy: cumulative[start + CLIP_SAMPLES] - cumulative[start] });
  }
  windows.sort((a, b) => b.energy - a.energy || a.start - b.start);
  const chosen: number[] = [];
  for (const { start } of windows) {
    if (chosen.every((other) => Math.abs(start - other) >= CLIP_SAMPLES)) chosen.push(start);
    if (chosen.length === count) break;
  }
  return chosen.map((start) => waveform.slice(start, start + CLIP_SAMPLES));
}

/** Averaging log probabilities across clips favours species heard consistently in the recording. */
export function averageLogits(clips: readonly ArrayLike<number>[]): Float64Array {
  if (clips.length === 0) throw new Error("No audio clips were scored.");
  const output = new Float64Array(clips[0].length);
  for (const clip of clips) for (let i = 0; i < output.length; i += 1) output[i] += clip[i];
  return output.map((value) => value / clips.length);
}

/** Mix to mono. Stereo microphones add no useful identity information. */
export function toMono(channels: readonly Float32Array[]): Float32Array {
  if (channels.length === 1) return channels[0];
  const mono = new Float32Array(channels[0].length);
  for (const channel of channels) {
    for (let i = 0; i < mono.length; i += 1) mono[i] += channel[i] / channels.length;
  }
  return mono;
}
