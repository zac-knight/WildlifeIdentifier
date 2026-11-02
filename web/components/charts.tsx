"use client";

import { useState } from "react";

export interface ReliabilityBin {
  low: number;
  high: number;
  n: number;
  confidence: number;
  accuracy: number;
}

const SERIES = {
  before: { colour: "var(--series-2)", label: "Before calibration" },
  after: { colour: "var(--series-1)", label: "After calibration" },
} as const;

type SeriesKey = keyof typeof SERIES;

interface Hovered {
  x: number;
  y: number;
  series: SeriesKey;
  bin: ReliabilityBin;
}

const SIZE = 320;
const MARGIN = { top: 12, right: 12, bottom: 40, left: 44 };
const PLOT = SIZE - MARGIN.left - MARGIN.right;
const GRID = [0, 0.25, 0.5, 0.75, 1];

function toX(value: number): number {
  return MARGIN.left + value * PLOT;
}

function toY(value: number): number {
  return SIZE - MARGIN.bottom - value * (SIZE - MARGIN.top - MARGIN.bottom);
}

export function ReliabilityChart({
  before,
  after,
  title,
}: {
  before: ReliabilityBin[];
  after: ReliabilityBin[];
  title: string;
}) {
  const [hovered, setHovered] = useState<Hovered | null>(null);
  const series: [SeriesKey, ReliabilityBin[]][] = [
    ["before", before],
    ["after", after],
  ];
  return (
    <figure className="space-y-2">
      <figcaption className="text-sm font-medium">{title}</figcaption>
      <ul className="flex flex-wrap gap-4 text-xs text-muted-foreground" aria-label="Legend">
        {series.map(([key]) => (
          <li key={key} className="flex items-center gap-1.5">
            <svg width="20" height="10" aria-hidden>
              <line
                x1="0"
                x2="20"
                y1="5"
                y2="5"
                stroke={SERIES[key].colour}
                strokeWidth="2"
                strokeDasharray={key === "before" ? "4 3" : undefined}
              />
            </svg>
            {SERIES[key].label}
          </li>
        ))}
        <li className="flex items-center gap-1.5">
          <svg width="20" height="10" aria-hidden>
            <line x1="0" x2="20" y1="5" y2="5" stroke="var(--muted-foreground)" strokeWidth="1" />
          </svg>
          Perfect calibration
        </li>
      </ul>
      <div className="relative max-w-sm">
        <svg
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          role="img"
          aria-label={`${title}. Accuracy against stated confidence. The table below has the values.`}
          className="w-full"
          onMouseLeave={() => setHovered(null)}
        >
          {GRID.map((value) => (
            <g key={value}>
              <line
                x1={toX(0)}
                x2={toX(1)}
                y1={toY(value)}
                y2={toY(value)}
                stroke="var(--border)"
              />
              <text
                x={MARGIN.left - 8}
                y={toY(value) + 4}
                textAnchor="end"
                fontSize="11"
                fill="var(--muted-foreground)"
              >
                {Math.round(value * 100)}%
              </text>
              <text
                x={toX(value)}
                y={SIZE - MARGIN.bottom + 16}
                textAnchor="middle"
                fontSize="11"
                fill="var(--muted-foreground)"
              >
                {Math.round(value * 100)}%
              </text>
            </g>
          ))}
          <text
            x={toX(0.5)}
            y={SIZE - 6}
            textAnchor="middle"
            fontSize="11"
            fill="var(--muted-foreground)"
          >
            Stated confidence
          </text>
          <text
            transform={`translate(11 ${toY(0.5)}) rotate(-90)`}
            textAnchor="middle"
            fontSize="11"
            fill="var(--muted-foreground)"
          >
            Actual accuracy
          </text>
          <line
            x1={toX(0)}
            y1={toY(0)}
            x2={toX(1)}
            y2={toY(1)}
            stroke="var(--muted-foreground)"
            strokeWidth="1"
          />
          {series.map(([key, bins]) => (
            <g key={key}>
              <polyline
                fill="none"
                stroke={SERIES[key].colour}
                strokeWidth="2"
                strokeLinejoin="round"
                strokeDasharray={key === "before" ? "4 3" : undefined}
                points={bins.map((bin) => `${toX(bin.confidence)},${toY(bin.accuracy)}`).join(" ")}
              />
              {bins.map((bin) => (
                <g key={bin.low}>
                  <circle
                    cx={toX(bin.confidence)}
                    cy={toY(bin.accuracy)}
                    r="4"
                    fill={SERIES[key].colour}
                    stroke="var(--card)"
                    strokeWidth="2"
                  />
                  {/* A larger invisible target makes small points easy to hover. */}
                  <circle
                    cx={toX(bin.confidence)}
                    cy={toY(bin.accuracy)}
                    r="11"
                    fill="transparent"
                    onMouseEnter={() =>
                      setHovered({
                        x: toX(bin.confidence),
                        y: toY(bin.accuracy),
                        series: key,
                        bin,
                      })
                    }
                  />
                </g>
              ))}
            </g>
          ))}
        </svg>
        {hovered ? (
          <div
            role="status"
            className="pointer-events-none absolute z-10 w-44 rounded-md border border-border bg-popover p-2 text-xs shadow-sm"
            style={{
              left: `min(${(hovered.x / SIZE) * 100}%, calc(100% - 11rem))`,
              top: `${(hovered.y / SIZE) * 100}%`,
            }}
          >
            <p className="font-medium">{SERIES[hovered.series].label}</p>
            <p>Confidence: {(hovered.bin.confidence * 100).toFixed(1)}%</p>
            <p>Accuracy: {(hovered.bin.accuracy * 100).toFixed(1)}%</p>
            <p>Photos: {hovered.bin.n.toLocaleString("en-AU")}</p>
          </div>
        ) : null}
      </div>
      <details className="text-xs">
        <summary className="cursor-pointer text-muted-foreground">View as a table</summary>
        <table className="mt-2 w-full max-w-sm text-left tabular-nums">
          <thead>
            <tr className="text-muted-foreground">
              <th className="font-normal">Series</th>
              <th className="font-normal">Confidence</th>
              <th className="font-normal">Accuracy</th>
              <th className="font-normal">Photos</th>
            </tr>
          </thead>
          <tbody>
            {series.flatMap(([key, bins]) =>
              bins.map((bin) => (
                <tr key={`${key}-${bin.low}`}>
                  <td>{key === "before" ? "Before" : "After"}</td>
                  <td>{(bin.confidence * 100).toFixed(1)}%</td>
                  <td>{(bin.accuracy * 100).toFixed(1)}%</td>
                  <td>{bin.n.toLocaleString("en-AU")}</td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
