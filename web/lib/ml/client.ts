import type { ProgressHandler } from "./model-cache";
import { MODEL_BASE_URL } from "./metadata";
import type { WorkerRequest, WorkerResponse } from "./protocol";

type Float = Float32Array<ArrayBuffer>;
type Pending = {
  resolve: (value: WorkerResponse) => void;
  reject: (reason: Error) => void;
  onProgress?: ProgressHandler;
};
type WithoutId<T> = T extends unknown ? Omit<T, "id"> : never;

/** Promise-based access to the inference worker. Created lazily, so pages load no ML until used. */
export class InferenceClient {
  private worker: Worker | null = null;
  private pending = new Map<number, Pending>();
  private nextId = 1;

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.addEventListener("message", (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      const entry = this.pending.get(message.id);
      if (!entry) return;
      if (message.type === "progress") {
        entry.onProgress?.(message.loaded, message.total);
        return;
      }
      this.pending.delete(message.id);
      if (message.type === "error") entry.reject(new Error(message.message));
      else entry.resolve(message);
    });
    worker.addEventListener("error", (event) => {
      const failure = new Error(event.message || "The inference worker stopped unexpectedly.");
      for (const entry of this.pending.values()) entry.reject(failure);
      this.pending.clear();
      worker.terminate();
      this.worker = null;
    });
    this.worker = worker;
    return worker;
  }

  private call(
    request: WithoutId<WorkerRequest>,
    transfer: Transferable[] = [],
    onProgress?: ProgressHandler,
  ): Promise<WorkerResponse> {
    const worker = this.ensureWorker();
    const id = this.nextId;
    this.nextId += 1;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      worker.postMessage({ ...request, id } as WorkerRequest, transfer);
    });
  }

  async load(
    key: string,
    files: {
      webgpu: { path: string; sha256?: string };
      wasm: { path: string; sha256?: string };
    },
    onProgress?: ProgressHandler,
    base = MODEL_BASE_URL,
  ): Promise<string> {
    // Worker URLs need an absolute origin, because the worker's own base URL is its script.
    const absolute = (path: string) =>
      new URL(`${base}/${key}/${path}`, globalThis.location.href).href;
    const response = await this.call(
      {
        type: "load",
        key,
        webgpu: { url: absolute(files.webgpu.path), sha256: files.webgpu.sha256 },
        wasm: { url: absolute(files.wasm.path), sha256: files.wasm.sha256 },
      },
      [],
      onProgress,
    );
    if (response.type !== "loaded") throw new Error("Unexpected worker response.");
    return response.provider;
  }

  async image(rgba: Uint8ClampedArray<ArrayBuffer>, width: number, height: number) {
    const response = await this.call({ type: "image", key: "image", rgba, width, height }, [
      rgba.buffer,
    ]);
    if (response.type !== "image") throw new Error("Unexpected worker response.");
    return response;
  }

  async prior(latitude: number, longitude: number, day: number): Promise<Float> {
    const response = await this.call({ type: "prior", key: "prior", latitude, longitude, day });
    if (response.type !== "prior") throw new Error("Unexpected worker response.");
    return response.presence;
  }

  async audio(clips: Float[]) {
    const response = await this.call(
      { type: "audio", key: "audio", clips },
      clips.map((clip) => clip.buffer),
    );
    if (response.type !== "audio") throw new Error("Unexpected worker response.");
    return response;
  }

  async embed(rgba: Uint8ClampedArray<ArrayBuffer>, width: number, height: number): Promise<Float> {
    const response = await this.call({ type: "embed", key: "fewshot", rgba, width, height }, [
      rgba.buffer,
    ]);
    if (response.type !== "embed") throw new Error("Unexpected worker response.");
    return response.embedding;
  }

  async detect(pixels: Float, size: number): Promise<{ boxes: Float; scores: Float }> {
    const response = await this.call({ type: "detect", key: "detector", pixels, size }, [
      pixels.buffer,
    ]);
    if (response.type !== "detect") throw new Error("Unexpected worker response.");
    return response;
  }
}

let shared: InferenceClient | null = null;

export function getClient(): InferenceClient {
  shared ??= new InferenceClient();
  return shared;
}
