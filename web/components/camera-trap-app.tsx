"use client";

import { AlertTriangle, Download, Loader2 } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { analyse, dayOfYear, isInAustralia } from "@/lib/ml/analysis";
import { getClient } from "@/lib/ml/client";
import {
  type BatchRow,
  decodeDetections,
  isBatchImage,
  letterbox,
  toCsv,
} from "@/lib/ml/detection";
import { fetchMetadata } from "@/lib/ml/metadata";
import { allSpecies, displayName, speciesByName } from "@/lib/species";

interface Row extends BatchRow {
  id: number;
  thumbnail: string | null;
}

type Filter = "all" | "animal" | "empty" | "person_or_vehicle";

const FILTERS: { value: Filter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "animal", label: "Animals" },
  { value: "empty", label: "Empty" },
  { value: "person_or_vehicle", label: "People and vehicles" },
];

async function expandFiles(files: File[]): Promise<{ name: string; blob: Blob }[]> {
  const output: { name: string; blob: Blob }[] = [];
  for (const file of files) {
    if (/\.zip$/i.test(file.name)) {
      const { unzipSync } = await import("fflate");
      const entries = unzipSync(new Uint8Array(await file.arrayBuffer()), {
        filter: (entry) => isBatchImage(entry.name),
      });
      for (const [name, data] of Object.entries(entries)) {
        output.push({ name, blob: new Blob([new Uint8Array(data)]) });
      }
    } else if (isBatchImage(file.name)) {
      output.push({ name: file.name, blob: file });
    }
  }
  return output;
}

function drawPixels(
  source: CanvasImageSource,
  x: number,
  y: number,
  width: number,
  height: number,
  outWidth: number,
  outHeight: number,
) {
  const canvas = new OffscreenCanvas(Math.max(1, outWidth), Math.max(1, outHeight));
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas unavailable");
  context.imageSmoothingQuality = "high";
  context.drawImage(source, x, y, width, height, 0, 0, canvas.width, canvas.height);
  return { canvas, context };
}

export function CameraTrapApp() {
  const [files, setFiles] = useState<File[]>([]);
  const [rows, setRows] = useState<Row[]>([]);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [speciesFilter, setSpeciesFilter] = useState("");
  const [minimum, setMinimum] = useState(0);
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [date, setDate] = useState("");
  const cancel = useRef(false);

  async function run() {
    cancel.current = false;
    setError(null);
    setRows([]);
    try {
      const client = getClient();
      setStatus("Preparing files");
      const images = await expandFiles(files);
      if (images.length === 0) {
        setError("No JPEG, PNG or WebP images were found in that selection.");
        return;
      }
      setStatus("Downloading the models (first time only)");
      const [detector, classifier] = await Promise.all([
        fetchMetadata("detector"),
        fetchMetadata("image"),
      ]);
      await client.load("detector", detector.files);
      await client.load("image", classifier.files);
      let prior: Float32Array | undefined;
      const lat = Number(latitude);
      const lon = Number(longitude);
      if (latitude && longitude && date && isInAustralia(lat, lon)) {
        const metadata = await fetchMetadata("prior");
        await client.load("prior", metadata.files);
        prior = await client.prior(lat, lon, dayOfYear(new Date(`${date}T12:00:00`)));
      }
      setStatus("Processing images");
      setProgress({ done: 0, total: images.length });

      const size = detector.inputSize;
      for (const [position, item] of images.entries()) {
        if (cancel.current) break;
        const id = position;
        let row: Row;
        try {
          const bitmap = await createImageBitmap(item.blob);
          const box = letterbox(bitmap.width, bitmap.height, size);
          const { context } = drawPixels(bitmap, 0, 0, bitmap.width, bitmap.height, size, size);
          context.fillStyle = "rgb(114,114,114)";
          context.fillRect(0, 0, size, size);
          context.drawImage(bitmap, box.padLeft, box.padTop, box.width, box.height);
          const pixels = context.getImageData(0, 0, size, size).data;
          const planar = new Float32Array(3 * size * size);
          for (let i = 0; i < size * size; i += 1) {
            planar[i] = pixels[i * 4] / 255;
            planar[size * size + i] = pixels[i * 4 + 1] / 255;
            planar[2 * size * size + i] = pixels[i * 4 + 2] / 255;
          }
          const { boxes, scores } = await client.detect(planar, size);
          const detections = decodeDetections(
            boxes,
            scores,
            detector.classes.length,
            box,
            bitmap.width,
            bitmap.height,
            detector.confidence,
            detector.iou,
          );
          const animals = detections.filter(
            (detection) => detector.classes[detection.classIndex] === "animal",
          );
          const others = detections.filter(
            (detection) => detector.classes[detection.classIndex] !== "animal",
          );
          const base = {
            id,
            file: item.name,
            species: null,
            commonName: null,
            confidence: null,
            corrected: false,
            thumbnail: null,
          };
          if (animals.length > 0) {
            const best = animals[0];
            const width = best.x2 - best.x1;
            const height = best.y2 - best.y1;
            const side = Math.max(width, height);
            // Square crops avoid distorting the bird, and the classifier expects square input.
            const x = Math.max(0, best.x1 - (side - width) / 2);
            const y = Math.max(0, best.y1 - (side - height) / 2);
            const crop = drawPixels(bitmap, x, y, side, side, 300, 300);
            const data = crop.context.getImageData(0, 0, 300, 300).data;
            const output = await client.image(new Uint8ClampedArray(data), 300, 300);
            const analysis = analyse({
              image: classifier,
              imageLogits: output.logits,
              prior,
            });
            const top = analysis.candidates[0];
            const species = speciesByName(top.name);
            const thumbnail = URL.createObjectURL(
              await crop.canvas.convertToBlob({ type: "image/jpeg", quality: 0.7 }),
            );
            row = {
              ...base,
              status: "animal",
              detectionScore: best.score,
              species: top.name,
              commonName: species ? displayName(species) : null,
              confidence: top.probability,
              thumbnail,
            };
          } else if (others.length > 0) {
            // People are never classified or shown. Only the fact of the detection is reported.
            row = { ...base, status: "person_or_vehicle", detectionScore: others[0].score };
          } else {
            row = { ...base, status: "empty", detectionScore: null };
          }
          bitmap.close();
        } catch {
          row = {
            id,
            file: item.name,
            status: "error",
            detectionScore: null,
            species: null,
            commonName: null,
            confidence: null,
            corrected: false,
            thumbnail: null,
          };
        }
        setRows((current) => [...current, row]);
        setProgress({ done: position + 1, total: images.length });
      }
    } catch (failure) {
      console.warn("Identification failed", failure);
      setError("The detection model could not be loaded. Camera trap mode is unavailable here.");
    } finally {
      setStatus(null);
    }
  }

  function correct(id: number, name: string) {
    setRows((current) =>
      current.map((row) => {
        if (row.id !== id) return row;
        const species = speciesByName(name);
        return {
          ...row,
          species: name || null,
          commonName: species ? displayName(species) : null,
          confidence: null,
          corrected: true,
        };
      }),
    );
  }

  function exportCsv() {
    const url = URL.createObjectURL(new Blob([toCsv(rows)], { type: "text/csv" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "wildlens-camera-trap.csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  const visible = useMemo(
    () =>
      rows.filter(
        (row) =>
          (filter === "all" || row.status === filter) &&
          (!speciesFilter || row.species === speciesFilter) &&
          (row.confidence === null || row.confidence * 100 >= minimum),
      ),
    [rows, filter, speciesFilter, minimum],
  );
  const present = useMemo(
    () => [...new Set(rows.map((row) => row.species).filter(Boolean))] as string[],
    [rows],
  );
  const busy = status !== null;

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="space-y-1.5 sm:col-span-2 lg:col-span-4">
          <Label htmlFor="batch-files">Images or a zip file</Label>
          <Input
            id="batch-files"
            type="file"
            multiple
            accept="image/*,.zip"
            onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
          />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="batch-lat">Latitude (optional)</Label>
          <Input
            id="batch-lat"
            inputMode="decimal"
            value={latitude}
            onChange={(event) => setLatitude(event.target.value)}
            placeholder="-33.87"
          />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="batch-lon">Longitude (optional)</Label>
          <Input
            id="batch-lon"
            inputMode="decimal"
            value={longitude}
            onChange={(event) => setLongitude(event.target.value)}
            placeholder="151.21"
          />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="batch-date">Date (optional)</Label>
          <Input
            id="batch-date"
            type="date"
            value={date}
            onChange={(event) => setDate(event.target.value)}
          />
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <Button disabled={files.length === 0 || busy} onClick={() => void run()}>
          {busy ? <Loader2 className="animate-spin" aria-hidden /> : null}
          Process {files.length > 0 ? `${files.length} file${files.length === 1 ? "" : "s"}` : ""}
        </Button>
        {busy && progress ? (
          <Button variant="outline" onClick={() => (cancel.current = true)}>
            Stop
          </Button>
        ) : null}
        {rows.length > 0 ? (
          <Button variant="outline" onClick={exportCsv}>
            <Download aria-hidden /> Export CSV
          </Button>
        ) : null}
      </div>

      <p className="text-xs text-muted-foreground">
        Everything runs on your device. Images are not uploaded.
      </p>

      {error ? (
        <Alert variant="destructive">
          <AlertTriangle aria-hidden />
          <AlertTitle>Could not process this batch</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}

      {status || progress ? (
        <div className="space-y-1.5" aria-live="polite">
          <p className="text-sm">
            {status ?? "Finished"}
            {progress ? `: ${progress.done} of ${progress.total}` : ""}
          </p>
          <Progress
            value={progress ? (progress.done / progress.total) * 100 : null}
            aria-label="Batch progress"
          />
        </div>
      ) : null}

      {rows.length > 0 ? (
        <section aria-label="Results" className="space-y-4">
          <div className="flex flex-wrap items-end gap-4">
            <div className="flex flex-wrap gap-1" role="group" aria-label="Filter by result">
              {FILTERS.map((item) => (
                <Button
                  key={item.value}
                  size="sm"
                  variant={filter === item.value ? "default" : "outline"}
                  aria-pressed={filter === item.value}
                  onClick={() => setFilter(item.value)}
                >
                  {item.label} (
                  {item.value === "all"
                    ? rows.length
                    : rows.filter((row) => row.status === item.value).length}
                  )
                </Button>
              ))}
            </div>
            <div className="space-y-1">
              <Label htmlFor="filter-species">Species</Label>
              <select
                id="filter-species"
                value={speciesFilter}
                onChange={(event) => setSpeciesFilter(event.target.value)}
                className="h-8 rounded-md border border-input bg-background px-2 text-sm"
              >
                <option value="">Any</option>
                {present.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1">
              <Label htmlFor="filter-confidence">Minimum confidence: {minimum}%</Label>
              <input
                id="filter-confidence"
                type="range"
                min={0}
                max={100}
                step={5}
                value={minimum}
                onChange={(event) => setMinimum(Number(event.target.value))}
                className="block w-40"
              />
            </div>
          </div>
          <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {visible.map((row) => (
              <li key={row.id} className="space-y-2 rounded-lg border border-border p-2 text-sm">
                {row.thumbnail ? (
                  // eslint-disable-next-line @next/next/no-img-element -- Local object URL
                  <img
                    src={row.thumbnail}
                    alt={`Detected animal in ${row.file}`}
                    className="aspect-square w-full rounded object-cover"
                  />
                ) : (
                  <div className="flex aspect-square w-full items-center justify-center rounded bg-muted p-2 text-center text-xs text-muted-foreground">
                    {row.status === "empty"
                      ? "No animal detected"
                      : row.status === "error"
                        ? "This image could not be read"
                        : "Person or vehicle detected. Image hidden."}
                  </div>
                )}
                <p className="truncate text-xs text-muted-foreground" title={row.file}>
                  {row.file}
                </p>
                {row.status === "animal" ? (
                  <>
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-medium">{row.commonName ?? row.species}</span>
                      {row.corrected ? (
                        <Badge variant="secondary">Corrected</Badge>
                      ) : row.confidence !== null ? (
                        <span className="text-xs tabular-nums">
                          {Math.round(row.confidence * 100)}%
                        </span>
                      ) : null}
                    </div>
                    <select
                      aria-label={`Correct the species for ${row.file}`}
                      value={row.species ?? ""}
                      onChange={(event) => correct(row.id, event.target.value)}
                      className="h-8 w-full rounded-md border border-input bg-background px-1 text-xs"
                    >
                      {allSpecies.map((species) => (
                        <option key={species.name} value={species.name}>
                          {displayName(species)}
                        </option>
                      ))}
                    </select>
                  </>
                ) : null}
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}
