"use client";

import "leaflet/dist/leaflet.css";
import { useEffect, useRef, useState } from "react";
import { OSM_ATTRIBUTION, OSM_TILES } from "@/components/location-map";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  climateAt,
  type ClimateGrid,
  type ClimatePoint,
  LAYER_BOUNDS,
  paintChange,
  paintSuitability,
  type Scenario,
  SCENARIOS,
} from "@/lib/sdm";
import type { Species } from "@/lib/species";

type View = "present" | Scenario;

async function readLayer(
  url: string,
): Promise<{ data: Uint8ClampedArray; width: number; height: number }> {
  const response = await fetch(url);
  if (!response.ok) throw new Error("Layer unavailable");
  // Premultiplication would corrupt values stored in the colour channels, so it is turned off.
  const bitmap = await createImageBitmap(await response.blob(), {
    premultiplyAlpha: "none",
    colorSpaceConversion: "none",
  });
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas unavailable");
  context.drawImage(bitmap, 0, 0);
  const { data, width, height } = context.getImageData(0, 0, bitmap.width, bitmap.height);
  return { data, width, height };
}

function toDataUrl(pixels: Uint8ClampedArray, width: number, height: number): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas
    .getContext("2d")
    ?.putImageData(new ImageData(new Uint8ClampedArray(pixels), width, height), 0, 0);
  return canvas.toDataURL();
}

function Warming({ present, points }: { present: number; points: ClimatePoint[] }) {
  const largest = Math.max(...points.map((point) => point.high - present), 1);
  return (
    <div className="space-y-2">
      <p className="text-sm">
        Modelled annual mean temperature for 1995–2014 here: <strong>{present.toFixed(1)}°C</strong>
      </p>
      <ul className="space-y-2" aria-label="Projected warming by scenario">
        {points.map((point) => {
          const median = point.median - present;
          const low = point.low - present;
          const high = point.high - present;
          return (
            <li key={point.scenario} className="text-xs">
              <div className="flex justify-between gap-2">
                <span>{SCENARIOS[point.scenario]}</span>
                <span className="tabular-nums">
                  +{median.toFixed(1)}°C (range +{low.toFixed(1)} to +{high.toFixed(1)})
                </span>
              </div>
              <div className="relative mt-1 h-2 rounded bg-muted" aria-hidden>
                <div
                  className="absolute h-2 rounded bg-chart-2/40"
                  style={{
                    left: `${(Math.max(low, 0) / largest) * 100}%`,
                    width: `${((high - Math.max(low, 0)) / largest) * 100}%`,
                  }}
                />
                <div
                  className="absolute h-2 w-1 rounded bg-chart-4"
                  style={{ left: `calc(${(Math.max(median, 0) / largest) * 100}% - 2px)` }}
                />
              </div>
            </li>
          );
        })}
      </ul>
      <p className="text-xs text-muted-foreground">
        The range spans the 10th to 90th percentile across NASA NEX-GDDP-CMIP6 climate models.
      </p>
    </div>
  );
}

export function HabitatMap({ species }: { species: Species }) {
  const container = useRef<HTMLDivElement>(null);
  const handles = useRef<{
    L: typeof import("leaflet");
    map: import("leaflet").Map;
    overlay: import("leaflet").ImageOverlay | null;
    marker: import("leaflet").CircleMarker | null;
  } | null>(null);
  const [ready, setReady] = useState(false);
  const [view, setView] = useState<View>("present");
  const [showChange, setShowChange] = useState(false);
  const [failed, setFailed] = useState(false);
  const [clicked, setClicked] = useState<{
    latitude: number;
    longitude: number;
    climate: ReturnType<typeof climateAt>;
  } | null>(null);
  const climate = useRef<ClimateGrid | null>(null);
  const threshold = species.sdm?.threshold ?? 0.5;

  useEffect(() => {
    let cancelled = false;
    let map: import("leaflet").Map | undefined;
    void import("leaflet").then((L) => {
      if (cancelled || !container.current) return;
      map = L.map(container.current, { minZoom: 3, maxZoom: 9 }).fitBounds(LAYER_BOUNDS);
      L.tileLayer(OSM_TILES, { attribution: OSM_ATTRIBUTION, maxZoom: 18 }).addTo(map);
      const active = map;
      active.on("click", async (event) => {
        const { lat, lng } = event.latlng;
        try {
          climate.current ??= (await (await fetch("/data/climate.json")).json()) as ClimateGrid;
          const values = climateAt(climate.current, lat, lng);
          setClicked({ latitude: lat, longitude: lng, climate: values });
          const current = handles.current;
          if (current) {
            current.marker?.remove();
            current.marker = current.L.circleMarker([lat, lng], {
              radius: 6,
              color: "#1f2937",
              fillColor: "#fff",
              fillOpacity: 1,
            }).addTo(current.map);
          }
        } catch {
          setClicked({ latitude: lat, longitude: lng, climate: null });
        }
      });
      handles.current = { L, map: active, overlay: null, marker: null };
      setReady(true);
    });
    return () => {
      cancelled = true;
      map?.remove();
      handles.current = null;
    };
  }, []);

  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    const folder = `/data/sdm/${species.slug}`;
    void (async () => {
      try {
        const layer = await readLayer(`${folder}/${view}.png`);
        let pixels: Uint8ClampedArray;
        if (showChange && view !== "present") {
          const present = await readLayer(`${folder}/present.png`);
          pixels = paintChange(present.data, layer.data, threshold);
        } else {
          pixels = paintSuitability(layer.data, layer.width);
        }
        const current = handles.current;
        if (cancelled || !current) return;
        current.overlay?.remove();
        current.overlay = current.L.imageOverlay(
          toDataUrl(pixels, layer.width, layer.height),
          LAYER_BOUNDS,
          { className: "habitat-overlay", opacity: 0.85, interactive: false },
        ).addTo(current.map);
        setFailed(false);
      } catch {
        if (!cancelled) setFailed(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [ready, view, showChange, species.slug, threshold]);

  const change = view === "present" ? null : species.sdm?.changes[view];

  return (
    <div className="space-y-4">
      <Alert>
        <AlertTitle>These are model projections, not predictions of fact</AlertTitle>
        <AlertDescription>
          They use climate alone. Land clearing, food, fire, competition and a bird&apos;s ability
          to move are not modelled, and uncertainty is substantial.
        </AlertDescription>
      </Alert>

      <div className="flex flex-wrap items-end gap-4">
        <div className="space-y-1.5">
          <Label htmlFor="scenario">Climate period</Label>
          <select
            id="scenario"
            value={view}
            onChange={(event) => setView(event.target.value as View)}
            className="h-9 rounded-md border border-input bg-background px-2 text-sm"
          >
            <option value="present">Recent climate, 1995–2014</option>
            {(Object.keys(SCENARIOS) as Scenario[]).map((scenario) => (
              <option key={scenario} value={scenario}>
                {SCENARIOS[scenario]}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2 pb-1.5">
          <Switch
            id="change-view"
            checked={showChange}
            disabled={view === "present"}
            onCheckedChange={setShowChange}
          />
          <Label htmlFor="change-view">Show change</Label>
        </div>
      </div>

      <div
        ref={container}
        role="application"
        aria-label={`Map of modelled habitat for ${species.commonName ?? species.name}. Click a place to see projected warming there.`}
        className="h-[420px] w-full overflow-hidden rounded-lg border border-border"
      />
      {failed ? (
        <p role="alert" className="text-sm text-destructive">
          This map layer could not be loaded.
        </p>
      ) : null}

      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {showChange && view !== "present" ? (
          <>
            <Key colour="rgb(33,113,181)" label="Gained" />
            <Key colour="rgb(200,80,30)" label="Lost" />
            <Key colour="rgb(120,150,110)" label="Remains suitable" />
            <span>Faint colours: climate models disagree</span>
          </>
        ) : (
          <>
            <Key colour="rgb(214,232,200)" label="Less suitable" />
            <Key colour="rgb(24,90,60)" label="More suitable" />
            {view !== "present" ? (
              <Key colour="rgb(120,120,120)" label="Checkered: climate unlike today" />
            ) : null}
          </>
        )}
      </div>

      {species.sdm?.coarsened ? (
        <p className="text-sm text-muted-foreground">
          This species is listed as threatened, so its map is deliberately blurred to a grid of
          about 100 km to avoid helping poachers.
        </p>
      ) : null}

      {change && change.area_change_percent !== null ? (
        <p className="text-sm">
          In this scenario, the model&apos;s suitable area changes by{" "}
          <strong>
            {change.area_change_percent > 0 ? "+" : ""}
            {change.area_change_percent.toFixed(0)}%
          </strong>
          . Climate models fully agree on the outcome across {Math.round(change.robust_share * 100)}
          % of the land shown.
          {change.novel_share > 0.01
            ? ` About ${Math.round(change.novel_share * 100)}% of it has climates unlike those the model learned from.`
            : ""}
        </p>
      ) : null}

      <div aria-live="polite">
        {clicked?.climate ? (
          <Warming present={clicked.climate.present} points={clicked.climate.points} />
        ) : clicked ? (
          <p className="text-sm text-muted-foreground">No climate data for that spot.</p>
        ) : (
          <p className="text-sm text-muted-foreground">
            Click a place on the map to see projected warming there.
          </p>
        )}
      </div>
    </div>
  );
}

function Key({ colour, label }: { colour: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span aria-hidden className="size-3 rounded-sm" style={{ background: colour }} />
      {label}
    </span>
  );
}
