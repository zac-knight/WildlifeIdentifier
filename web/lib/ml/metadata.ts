import { z } from "zod";

export const MODEL_BASE_URL = (process.env.NEXT_PUBLIC_MODEL_BASE_URL ?? "/models").replace(
  /\/$/,
  "",
);

const fileEntry = z.object({
  path: z.string(),
  sha256: z.string().optional(),
  bytes: z.number().optional(),
});

const files = z.object({ webgpu: fileEntry, wasm: fileEntry });

export const imageMetadata = z.object({
  version: z.string(),
  classes: z.array(z.string()),
  genera: z.array(z.string()),
  families: z.array(z.string()),
  temperature: z.number().positive(),
  fusedTemperature: z.number().positive(),
  noLocationThresholds: z.object({
    species: z.number(),
    genus: z.number(),
    family: z.number(),
  }),
  thresholds: z.object({ species: z.number(), genus: z.number(), family: z.number() }),
  priorWeight: z.number().nonnegative(),
  priorFloor: z.number().positive(),
  energyThreshold: z.number(),
  featureChannels: z.number().int(),
  featureSize: z.number().int(),
  weightsFile: fileEntry,
  files,
});
export type ImageMetadata = z.infer<typeof imageMetadata>;

export const priorMetadata = z.object({
  version: z.string(),
  classes: z.array(z.string()),
  files,
});
export type PriorMetadata = z.infer<typeof priorMetadata>;

export const audioMetadata = z.object({
  version: z.string(),
  classes: z.array(z.string()),
  temperature: z.number().positive(),
  clips: z.number().int().positive(),
  files,
});
export type AudioMetadata = z.infer<typeof audioMetadata>;

export const fewShotMetadata = z.object({
  version: z.string(),
  /** Classifier confidence below which prototypes are consulted. */
  uncertainBelow: z.number(),
  /** Blending weight for prototype evidence, tuned on validation data. */
  weight: z.number(),
  scale: z.number(),
  dimension: z.number().int(),
  classes: z.array(z.string()),
  prototypesFile: fileEntry,
  files,
});
export type FewShotMetadata = z.infer<typeof fewShotMetadata>;

export const detectorMetadata = z.object({
  version: z.string(),
  inputSize: z.number().int(),
  classes: z.array(z.string()),
  confidence: z.number(),
  iou: z.number(),
  files,
});
export type DetectorMetadata = z.infer<typeof detectorMetadata>;

const schemas = {
  image: imageMetadata,
  prior: priorMetadata,
  audio: audioMetadata,
  fewshot: fewShotMetadata,
  detector: detectorMetadata,
} as const;

export type ModelName = keyof typeof schemas;

export interface MetadataByName {
  image: ImageMetadata;
  prior: PriorMetadata;
  audio: AudioMetadata;
  fewshot: FewShotMetadata;
  detector: DetectorMetadata;
}

export async function fetchMetadata<N extends ModelName>(
  name: N,
  base = MODEL_BASE_URL,
): Promise<MetadataByName[N]> {
  const response = await fetch(`${base}/${name}/metadata.json`);
  if (!response.ok) throw new Error(`Model details for "${name}" could not be loaded.`);
  return schemas[name].parse(await response.json()) as MetadataByName[N];
}
