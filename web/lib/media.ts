import { AUDIO_SAMPLE_RATE, toMono } from "./ml/audio";

export interface LoadedImage {
  rgba: Uint8ClampedArray<ArrayBuffer>;
  width: number;
  height: number;
  previewUrl: string;
}

/** Very large photos are reduced first, as 2048 px is far above the 224 px the model uses. */
const MAX_DECODE_SIDE = 2048;

export async function loadImage(file: Blob): Promise<LoadedImage> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, MAX_DECODE_SIDE / Math.max(bitmap.width, bitmap.height));
  const width = Math.round(bitmap.width * scale);
  const height = Math.round(bitmap.height * scale);
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("This browser could not read the image.");
  context.imageSmoothingQuality = "high";
  context.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  const { data } = context.getImageData(0, 0, width, height);
  return {
    rgba: new Uint8ClampedArray(data),
    width,
    height,
    previewUrl: URL.createObjectURL(file),
  };
}

export interface PhotoDetails {
  latitude?: number;
  longitude?: number;
  date?: Date;
}

/** Location and date are read locally. Nothing about the photo leaves the device here. */
export async function readPhotoDetails(file: Blob): Promise<PhotoDetails> {
  try {
    const { parse } = await import("exifr");
    const tags = await parse(file, { gps: true, pick: ["DateTimeOriginal", "CreateDate"] });
    if (!tags) return {};
    const date = tags.DateTimeOriginal ?? tags.CreateDate;
    return {
      latitude: typeof tags.latitude === "number" ? tags.latitude : undefined,
      longitude: typeof tags.longitude === "number" ? tags.longitude : undefined,
      date: date instanceof Date ? date : undefined,
    };
  } catch {
    return {}; // Screenshots and edited photos often have no readable details.
  }
}

/** Decodes any browser-supported audio to the mono 32 kHz format the model was trained on. */
export async function decodeAudio(file: Blob): Promise<Float32Array<ArrayBuffer>> {
  const context = new OfflineAudioContext(1, 1, AUDIO_SAMPLE_RATE);
  const buffer = await context.decodeAudioData(await file.arrayBuffer());
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, index) =>
    buffer.getChannelData(index),
  );
  return Float32Array.from(toMono(channels));
}

/** Browser voice processing would remove bird calls as "noise", so all of it is turned off. */
export const RECORDING_CONSTRAINTS: MediaStreamConstraints = {
  audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
};
