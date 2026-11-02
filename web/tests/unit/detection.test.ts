import { describe, expect, test } from "vitest";
import {
  type BatchRow,
  decodeDetections,
  type Detection,
  isBatchImage,
  letterbox,
  nonMaximumSuppression,
  toCsv,
} from "@/lib/ml/detection";

function detection(overrides: Partial<Detection>): Detection {
  return { x1: 0, y1: 0, x2: 10, y2: 10, classIndex: 0, score: 0.9, ...overrides };
}

describe("letterbox", () => {
  test("a wide image is centred vertically", () => {
    expect(letterbox(1280, 640, 640)).toEqual({
      scale: 0.5,
      padLeft: 0,
      padTop: 160,
      width: 640,
      height: 320,
    });
  });

  test("a tall image is centred horizontally", () => {
    const box = letterbox(320, 640, 640);
    expect(box.padLeft).toBe(160);
    expect(box.padTop).toBe(0);
  });
});

describe("non-maximum suppression", () => {
  test("keeps only the strongest overlapping box", () => {
    const kept = nonMaximumSuppression(
      [detection({ score: 0.5 }), detection({ score: 0.9, x2: 11 })],
      0.5,
    );
    expect(kept).toHaveLength(1);
    expect(kept[0].score).toBe(0.9);
  });

  test("keeps separate objects", () => {
    const kept = nonMaximumSuppression(
      [detection({}), detection({ x1: 100, x2: 110, score: 0.4 })],
      0.5,
    );
    expect(kept).toHaveLength(2);
  });

  test("different classes never suppress each other", () => {
    expect(nonMaximumSuppression([detection({}), detection({ classIndex: 1 })], 0.5)).toHaveLength(
      2,
    );
  });
});

describe("decoding model output", () => {
  test("maps boxes back to the original image", () => {
    const box = letterbox(1280, 640, 640);
    const [found] = decodeDetections(
      [100, 260, 200, 360],
      [0.8, 0.1, 0.1],
      3,
      box,
      1280,
      640,
      0.5,
      0.5,
    );
    expect(found).toMatchObject({ x1: 200, y1: 200, x2: 400, y2: 400, classIndex: 0 });
  });

  test("weak detections are ignored", () => {
    const box = letterbox(640, 640, 640);
    expect(decodeDetections([0, 0, 10, 10], [0.1, 0.1, 0.1], 3, box, 640, 640, 0.5, 0.5)).toEqual(
      [],
    );
  });

  test("boxes are clipped to the image", () => {
    const box = letterbox(640, 640, 640);
    const [found] = decodeDetections([-20, -20, 700, 700], [0.9, 0, 0], 3, box, 640, 640, 0.5, 0.5);
    expect(found).toMatchObject({ x1: 0, y1: 0, x2: 640, y2: 640 });
  });
});

describe("batch files", () => {
  test.each([
    ["a.jpg", true],
    ["folder/b.JPEG", true],
    ["c.webp", true],
    ["notes.txt", false],
    [".hidden.jpg", false],
    ["__MACOSX/a.jpg", false],
  ])("%s gives %s", (name, expected) => {
    expect(isBatchImage(name)).toBe(expected);
  });
});

describe("CSV export", () => {
  const row: BatchRow = {
    file: "a.jpg",
    status: "animal",
    detectionScore: 0.9,
    species: "Dacelo novaeguineae",
    commonName: "Laughing Kookaburra",
    confidence: 0.8,
    corrected: false,
  };

  test("writes a header and one line per image", () => {
    const lines = toCsv([row]).split("\r\n");
    expect(lines[0]).toContain("file,status");
    expect(lines[1]).toBe("a.jpg,animal,0.9,Dacelo novaeguineae,Laughing Kookaburra,0.8,false");
  });

  test("quotes commas and quotation marks", () => {
    expect(toCsv([{ ...row, file: 'a,"b".jpg' }])).toContain('"a,""b"".jpg"');
  });

  test("neutralises spreadsheet formulas in file names", () => {
    expect(toCsv([{ ...row, file: "=HYPERLINK(1).jpg" }])).toContain("'=HYPERLINK(1).jpg");
  });
});
