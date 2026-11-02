const CACHE_NAME = "wildlens-models-v1";

export type ProgressHandler = (loaded: number, total: number | null) => void;

async function openCache(): Promise<Cache | null> {
  try {
    return typeof caches === "undefined" ? null : await caches.open(CACHE_NAME);
  } catch {
    return null; // Private browsing can block the Cache API. Models then download each visit.
  }
}

/** Read a response body while reporting progress, so a large first download never looks stuck. */
export async function readWithProgress(
  response: Response,
  onProgress?: ProgressHandler,
): Promise<Uint8Array<ArrayBuffer>> {
  const header = response.headers.get("content-length");
  const total = header ? Number(header) : null;
  if (!response.body) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.(loaded, total);
  }
  const output = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
}

export async function sha256Hex(data: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

/**
 * Fetch a file once, then serve it from the Cache API. The checksum protects users from a corrupt or
 * replaced file, and a bad cache entry is discarded rather than kept.
 */
export async function fetchCached(
  url: string,
  expectedSha256?: string,
  onProgress?: ProgressHandler,
): Promise<Uint8Array<ArrayBuffer>> {
  const cache = await openCache();
  const cached = await cache?.match(url);
  if (cached) {
    const data = new Uint8Array(await cached.arrayBuffer());
    if (!expectedSha256 || (await sha256Hex(data)) === expectedSha256) {
      onProgress?.(data.length, data.length);
      return data;
    }
    await cache?.delete(url);
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Download failed with status ${response.status}.`);
  const data = await readWithProgress(response, onProgress);
  if (expectedSha256 && (await sha256Hex(data)) !== expectedSha256) {
    throw new Error("The downloaded model failed its integrity check.");
  }
  try {
    await cache?.put(url, new Response(data));
  } catch {
    // A full storage quota should not stop the app. The file just downloads again next time.
  }
  return data;
}
