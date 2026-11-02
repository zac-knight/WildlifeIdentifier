"use client";

import { useEffect, useRef, useState } from "react";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { heatmapPixels, normaliseMap } from "@/lib/ml/cam";

interface CamOverlayProps {
  src: string;
  alt: string;
  map?: Float32Array;
  size?: number;
  speciesName?: string;
}

export function CamOverlay({ src, alt, map, size, speciesName }: CamOverlayProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [opacity, setOpacity] = useState(55);

  useEffect(() => {
    const element = canvas.current;
    if (!element || !map || !size) return;
    element.width = size;
    element.height = size;
    const pixels = heatmapPixels(normaliseMap(map));
    element
      .getContext("2d")
      ?.putImageData(new ImageData(new Uint8ClampedArray(pixels), size, size), 0, 0);
  }, [map, size]);

  return (
    <figure className="space-y-3">
      <div className="relative overflow-hidden rounded-lg border border-border">
        {/* The model sees a centre-cropped square, so the overlay aligns only if this crops too. */}
        {/* eslint-disable-next-line @next/next/no-img-element -- A local object URL has no size */}
        <img src={src} alt={alt} className="aspect-square w-full object-cover" />
        {map ? (
          <canvas
            ref={canvas}
            data-testid="cam-canvas"
            aria-hidden
            style={{ opacity: opacity / 100, imageRendering: "auto" }}
            className="absolute inset-0 h-full w-full"
          />
        ) : null}
      </div>
      {map ? (
        <figcaption className="space-y-2 text-sm text-muted-foreground">
          <div className="space-y-1.5">
            <Label htmlFor="cam-opacity">Heatmap strength</Label>
            <Slider
              id="cam-opacity"
              aria-label="Heatmap strength"
              value={[opacity]}
              onValueChange={(value) => setOpacity(Array.isArray(value) ? value[0] : value)}
              min={0}
              max={100}
            />
          </div>
          <p>
            Warmer areas raised the score for {speciesName ?? "this species"} the most. This shows
            where the model looked, not why it is right.
          </p>
        </figcaption>
      ) : null}
    </figure>
  );
}
