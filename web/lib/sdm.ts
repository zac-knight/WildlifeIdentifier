export const SCENARIOS = {
  ssp245_2050: "Moderate emissions, 2041–2060",
  ssp245_2070: "Moderate emissions, 2061–2080",
  ssp585_2050: "High emissions, 2041–2060",
  ssp585_2070: "High emissions, 2061–2080",
} as const;

export type Scenario = keyof typeof SCENARIOS;

/** Geographic extent of each layer, matching AU_LAT and AU_LON in ml/wildlens/config.py. */
export const LAYER_BOUNDS: [[number, number], [number, number]] = [
  [-44, 112],
  [-9, 154.5],
];

export interface LayerPixel {
  suitability: number;
  robust: boolean;
  land: boolean;
  novel: boolean;
}

/** Layer channels: red is suitability, green is agreement, blue is land, alpha 128 is novel climate. */
export function decodePixel(r: number, g: number, b: number, a: number): LayerPixel {
  return { suitability: r / 255, robust: g > 127, land: b > 127, novel: a < 200 };
}

const TRANSPARENT = [0, 0, 0, 0] as const;

/** Light to dark green, so a darker colour always means more suitable. Low values fade out. */
export function suitabilityColour(value: number): readonly [number, number, number, number] {
  const t = Math.min(1, Math.max(0, value));
  return [
    Math.round(214 - 190 * t),
    Math.round(232 - 142 * t),
    Math.round(200 - 140 * t),
    Math.round(30 + 200 * t),
  ];
}

export type ChangeClass = "gained" | "lost" | "kept" | "none";

export function classifyChange(present: number, future: number, threshold: number): ChangeClass {
  const was = present >= threshold;
  const is = future >= threshold;
  if (was && is) return "kept";
  if (was) return "lost";
  return is ? "gained" : "none";
}

const CHANGE_COLOURS: Record<ChangeClass, readonly [number, number, number, number]> = {
  gained: [33, 113, 181, 220],
  lost: [200, 80, 30, 220],
  kept: [120, 150, 110, 120],
  none: TRANSPARENT,
};

/** Paint a suitability layer, showing unfamiliar future climates with a checker pattern. */
export function paintSuitability(source: Uint8ClampedArray, width: number): Uint8ClampedArray {
  const output = new Uint8ClampedArray(source.length);
  for (let i = 0; i < source.length; i += 4) {
    const pixel = decodePixel(source[i], source[i + 1], source[i + 2], source[i + 3]);
    if (!pixel.land) continue;
    const x = (i / 4) % width;
    const y = Math.floor(i / 4 / width);
    const colour = suitabilityColour(pixel.suitability);
    const hatched = pixel.novel && (x + y) % 2 === 0;
    output.set(hatched ? [120, 120, 120, 170] : colour, i);
  }
  return output;
}

/**
 * Paint gained and lost habitat. Disputed change, where climate percentiles disagree, is drawn
 * fainter, so it is not read with the same confidence as change they all agree on.
 */
export function paintChange(
  present: Uint8ClampedArray,
  future: Uint8ClampedArray,
  threshold: number,
): Uint8ClampedArray {
  const output = new Uint8ClampedArray(present.length);
  for (let i = 0; i < present.length; i += 4) {
    const before = decodePixel(present[i], present[i + 1], present[i + 2], present[i + 3]);
    const after = decodePixel(future[i], future[i + 1], future[i + 2], future[i + 3]);
    if (!before.land) continue;
    const change = classifyChange(before.suitability, after.suitability, threshold);
    const colour = CHANGE_COLOURS[change];
    const faint = !after.robust && (change === "gained" || change === "lost");
    output.set(faint ? [colour[0], colour[1], colour[2], 90] : colour, i);
  }
  return output;
}

export interface ClimateGrid {
  top: number;
  left: number;
  cell: number;
  present: (number | null)[][];
  scenarios: Record<Scenario, Record<"median" | "p10" | "p90", (number | null)[][]>>;
}

export interface ClimatePoint {
  scenario: Scenario;
  median: number;
  low: number;
  high: number;
}

/** Projected warming at a place, or null over the ocean or outside the grid. */
export function climateAt(
  grid: ClimateGrid,
  latitude: number,
  longitude: number,
): { present: number; points: ClimatePoint[] } | null {
  const row = Math.floor((grid.top - latitude) / grid.cell);
  const column = Math.floor((longitude - grid.left) / grid.cell);
  const present = grid.present[row]?.[column];
  if (present === null || present === undefined) return null;
  const points: ClimatePoint[] = [];
  for (const scenario of Object.keys(SCENARIOS) as Scenario[]) {
    const data = grid.scenarios[scenario];
    const median = data.median[row]?.[column];
    const low = data.p10[row]?.[column];
    const high = data.p90[row]?.[column];
    if (median == null || low == null || high == null) return null;
    points.push({ scenario, median, low, high });
  }
  return { present, points };
}
