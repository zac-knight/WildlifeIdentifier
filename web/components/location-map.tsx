"use client";

import "leaflet/dist/leaflet.css";
import { useEffect, useRef } from "react";

interface LocationMapProps {
  latitude?: number;
  longitude?: number;
  onPick: (latitude: number, longitude: number) => void;
}

export const OSM_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
export const OSM_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

/** Click or tap to drop a pin. Leaflet loads only when this map is displayed. */
export function LocationMap({ latitude, longitude, onPick }: LocationMapProps) {
  const container = useRef<HTMLDivElement>(null);
  const handles = useRef<{
    map: import("leaflet").Map;
    marker: import("leaflet").CircleMarker | null;
    L: typeof import("leaflet");
  } | null>(null);
  const pick = useRef(onPick);

  useEffect(() => {
    pick.current = onPick;
  }, [onPick]);

  useEffect(() => {
    let cancelled = false;
    let map: import("leaflet").Map | undefined;
    void import("leaflet").then((L) => {
      if (cancelled || !container.current) return;
      map = L.map(container.current).setView([-27, 134], 3);
      L.tileLayer(OSM_TILES, { attribution: OSM_ATTRIBUTION, maxZoom: 18 }).addTo(map);
      map.on("click", (event) => pick.current(event.latlng.lat, event.latlng.lng));
      handles.current = { map, marker: null, L };
    });
    return () => {
      cancelled = true;
      map?.remove();
      handles.current = null;
    };
  }, []);

  useEffect(() => {
    const current = handles.current;
    if (!current || latitude === undefined || longitude === undefined) return;
    current.marker?.remove();
    current.marker = current.L.circleMarker([latitude, longitude], {
      radius: 8,
      color: "#1f5f4a",
      fillColor: "#e0a030",
      fillOpacity: 0.95,
    }).addTo(current.map);
  }, [latitude, longitude]);

  return (
    <div
      ref={container}
      role="application"
      aria-label="Map of Australia. Click or tap to set where the bird was seen."
      className="h-64 w-full overflow-hidden rounded-lg border border-border"
    />
  );
}
