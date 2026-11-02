export interface RemoteFile {
  url: string;
  sha256?: string;
}

type Float = Float32Array<ArrayBuffer>;

export type WorkerRequest = { id: number } & (
  | { type: "load"; key: string; webgpu: RemoteFile; wasm: RemoteFile }
  | {
      type: "image";
      key: string;
      rgba: Uint8ClampedArray<ArrayBuffer>;
      width: number;
      height: number;
    }
  | { type: "prior"; key: string; latitude: number; longitude: number; day: number }
  | { type: "audio"; key: string; clips: Float[] }
  | {
      type: "embed";
      key: string;
      rgba: Uint8ClampedArray<ArrayBuffer>;
      width: number;
      height: number;
    }
  | { type: "detect"; key: string; pixels: Float; size: number }
);

export type WorkerResponse = { id: number } & (
  | { type: "progress"; loaded: number; total: number | null }
  | { type: "loaded"; provider: string }
  | { type: "image"; logits: Float; features?: Float; milliseconds: number }
  | { type: "prior"; presence: Float }
  | { type: "audio"; logits: Float[]; milliseconds: number }
  | { type: "embed"; embedding: Float }
  | { type: "detect"; boxes: Float; scores: Float }
  | { type: "error"; message: string }
);
