export interface Detection {
  /** Corners in original image pixels. */
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  classIndex: number;
  score: number;
}

export interface Letterbox {
  scale: number;
  padLeft: number;
  padTop: number;
  width: number;
  height: number;
}

/** Fit an image inside the square model input, keeping its shape, as MegaDetector was trained. */
export function letterbox(width: number, height: number, size: number): Letterbox {
  const scale = Math.min(size / width, size / height);
  const newWidth = Math.trunc(width * scale);
  const newHeight = Math.trunc(height * scale);
  return {
    scale,
    padLeft: Math.floor((size - newWidth) / 2),
    padTop: Math.floor((size - newHeight) / 2),
    width: newWidth,
    height: newHeight,
  };
}

function intersectionOverUnion(a: Detection, b: Detection): number {
  const width = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const height = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const overlap = width * height;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  const union = areaA + areaB - overlap;
  return union > 0 ? overlap / union : 0;
}

/** Keep the strongest of any overlapping boxes, separately for each class. */
export function nonMaximumSuppression(detections: Detection[], iou: number): Detection[] {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const kept: Detection[] = [];
  for (const candidate of sorted) {
    const overlaps = kept.some(
      (other) =>
        other.classIndex === candidate.classIndex && intersectionOverUnion(other, candidate) > iou,
    );
    if (!overlaps) kept.push(candidate);
  }
  return kept;
}

/** Convert raw per-anchor model output into final detections in original image coordinates. */
export function decodeDetections(
  boxes: ArrayLike<number>,
  scores: ArrayLike<number>,
  classCount: number,
  box: Letterbox,
  imageWidth: number,
  imageHeight: number,
  confidence: number,
  iou: number,
): Detection[] {
  const anchors = boxes.length / 4;
  const found: Detection[] = [];
  for (let anchor = 0; anchor < anchors; anchor += 1) {
    for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
      const score = scores[anchor * classCount + classIndex];
      if (score < confidence) continue;
      const toX = (value: number) =>
        Math.min(imageWidth, Math.max(0, (value - box.padLeft) / box.scale));
      const toY = (value: number) =>
        Math.min(imageHeight, Math.max(0, (value - box.padTop) / box.scale));
      found.push({
        x1: toX(boxes[anchor * 4]),
        y1: toY(boxes[anchor * 4 + 1]),
        x2: toX(boxes[anchor * 4 + 2]),
        y2: toY(boxes[anchor * 4 + 3]),
        classIndex,
        score,
      });
    }
  }
  return nonMaximumSuppression(found, iou);
}

export interface BatchRow {
  file: string;
  status: "animal" | "empty" | "person_or_vehicle" | "error";
  detectionScore: number | null;
  species: string | null;
  commonName: string | null;
  confidence: number | null;
  corrected: boolean;
}

function csvCell(value: string | number | boolean | null): string {
  if (value === null) return "";
  const text = String(value);
  // A leading formula character would run as a formula if the file is opened in a spreadsheet.
  const safe = /^[=+\-@\t\r]/.test(text) ? `'${text}` : text;
  return /[",\n\r]/.test(safe) ? `"${safe.replaceAll('"', '""')}"` : safe;
}

export function toCsv(rows: readonly BatchRow[]): string {
  const header = [
    "file",
    "status",
    "detection_score",
    "species",
    "common_name",
    "confidence",
    "corrected",
  ];
  const lines = rows.map((row) =>
    [
      row.file,
      row.status,
      row.detectionScore,
      row.species,
      row.commonName,
      row.confidence,
      row.corrected,
    ]
      .map(csvCell)
      .join(","),
  );
  return [header.join(","), ...lines].join("\r\n");
}

const IMAGE_FILE = /\.(jpe?g|png|webp)$/i;

/** Hidden system files and folders are skipped, as zip archives from a Mac often contain them. */
export function isBatchImage(path: string): boolean {
  const parts = path.split("/");
  const name = parts[parts.length - 1];
  return IMAGE_FILE.test(name) && !name.startsWith(".") && !parts.includes("__MACOSX");
}
