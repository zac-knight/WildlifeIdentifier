"use client";

import dynamic from "next/dynamic";
import type { Species } from "@/lib/species";

const HabitatMap = dynamic(
  () => import("@/components/habitat-map").then((module) => module.HabitatMap),
  {
    ssr: false,
    loading: () => <div className="h-[420px] w-full animate-pulse rounded-lg bg-muted" />,
  },
);

export function HabitatMapLoader({ species }: { species: Species }) {
  return <HabitatMap species={species} />;
}
