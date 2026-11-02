/// <reference lib="webworker" />
// Inference runs in a worker so that a slow model never freezes the page.
import * as ort from "onnxruntime-web/webgpu";
import { fetchCached } from "./model-cache";
import { INPUT_SIZE, preprocess } from "./preprocess";
import type { WorkerRequest, WorkerResponse } from "./protocol";

const ORT_VERSION = "1.31.1";
// The runtime's WebAssembly is 14-28 MB. A versioned CDN copy spares the app's hosting bandwidth.
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
ort.env.wasm.numThreads = self.crossOriginIsolated
  ? Math.min(4, self.navigator.hardwareConcurrency ?? 1)
  : 1;

const sessions = new Map<string, { session: ort.InferenceSession; provider: string }>();

function hasWebGpu(): boolean {
  return "gpu" in self.navigator;
}

function post(message: WorkerResponse, transfer: Transferable[] = []): void {
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(message, transfer);
}

async function load(request: Extract<WorkerRequest, { type: "load" }>): Promise<string> {
  const existing = sessions.get(request.key);
  if (existing) return existing.provider;
  const progress = (loaded: number, total: number | null) =>
    post({ id: request.id, type: "progress", loaded, total });

  // WebGPU is much faster on supported devices. WebAssembly is the fallback that works everywhere.
  if (hasWebGpu()) {
    try {
      const data = await fetchCached(request.webgpu.url, request.webgpu.sha256, progress);
      const session = await ort.InferenceSession.create(data, {
        executionProviders: ["webgpu"],
      });
      sessions.set(request.key, { session, provider: "webgpu" });
      return "webgpu";
    } catch (error) {
      console.warn("WebGPU was unavailable, so WildLens is falling back to WebAssembly.", error);
    }
  }
  const data = await fetchCached(request.wasm.url, request.wasm.sha256, progress);
  const session = await ort.InferenceSession.create(data, { executionProviders: ["wasm"] });
  sessions.set(request.key, { session, provider: "wasm" });
  return "wasm";
}

function requireSession(key: string): ort.InferenceSession {
  const entry = sessions.get(key);
  if (!entry) throw new Error(`Model "${key}" has not been loaded.`);
  return entry.session;
}

function copyFloat32(tensor: ort.Tensor): Float32Array<ArrayBuffer> {
  return Float32Array.from(tensor.data as Float32Array);
}

async function handle(request: WorkerRequest): Promise<void> {
  switch (request.type) {
    case "load": {
      const provider = await load(request);
      post({ id: request.id, type: "loaded", provider });
      return;
    }
    case "image": {
      const session = requireSession(request.key);
      const began = performance.now();
      const pixels = preprocess(request.rgba, request.width, request.height);
      const input = new ort.Tensor("float32", pixels, [1, 3, INPUT_SIZE, INPUT_SIZE]);
      const outputs = await session.run({ image: input });
      const logits = copyFloat32(outputs.logits);
      const features = outputs.features ? copyFloat32(outputs.features) : undefined;
      const milliseconds = performance.now() - began;
      post({ id: request.id, type: "image", logits, features, milliseconds }, [
        logits.buffer,
        ...(features ? [features.buffer] : []),
      ]);
      return;
    }
    case "prior": {
      const session = requireSession(request.key);
      const input = new ort.Tensor(
        "float32",
        Float32Array.of(request.latitude, request.longitude, request.day),
        [1, 3],
      );
      const outputs = await session.run({ place: input });
      const presence = copyFloat32(outputs.presence);
      post({ id: request.id, type: "prior", presence }, [presence.buffer]);
      return;
    }
    case "audio": {
      const session = requireSession(request.key);
      const results: Float32Array<ArrayBuffer>[] = [];
      const began = performance.now();
      for (const clip of request.clips) {
        const input = new ort.Tensor("float32", clip, [1, clip.length]);
        const outputs = await session.run({ waveform: input });
        results.push(copyFloat32(outputs.logits));
      }
      post(
        {
          id: request.id,
          type: "audio",
          logits: results,
          milliseconds: performance.now() - began,
        },
        results.map((result) => result.buffer),
      );
      return;
    }
    case "embed": {
      const session = requireSession(request.key);
      const pixels = preprocess(request.rgba, request.width, request.height);
      const input = new ort.Tensor("float32", pixels, [1, 3, INPUT_SIZE, INPUT_SIZE]);
      const outputs = await session.run({ image: input });
      const embedding = copyFloat32(outputs.embedding);
      post({ id: request.id, type: "embed", embedding }, [embedding.buffer]);
      return;
    }
    case "detect": {
      const session = requireSession(request.key);
      const size = request.size;
      const input = new ort.Tensor("float32", request.pixels, [1, 3, size, size]);
      const outputs = await session.run({ image: input });
      const boxes = copyFloat32(outputs.boxes);
      const scores = copyFloat32(outputs.scores);
      post({ id: request.id, type: "detect", boxes, scores }, [boxes.buffer, scores.buffer]);
      return;
    }
  }
}

self.addEventListener("message", (event: MessageEvent<WorkerRequest>) => {
  handle(event.data).catch((error: unknown) => {
    post({
      id: event.data.id,
      type: "error",
      message: error instanceof Error ? error.message : "The model could not run.",
    });
  });
});
