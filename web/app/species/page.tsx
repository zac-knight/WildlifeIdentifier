import type { Metadata } from "next";
import { SpeciesBrowser } from "@/components/species-browser";
import { allSpecies } from "@/lib/species";

export const metadata: Metadata = {
  title: "Species",
  description: "The Australian bird species WildLens can identify.",
};

export default function SpeciesPage() {
  return (
    <div className="mx-auto max-w-6xl space-y-6 px-4 py-10">
      <header className="max-w-2xl space-y-2">
        <h1 className="text-4xl font-semibold">Species</h1>
        <p className="text-muted-foreground">
          WildLens knows {allSpecies.length} Australian birds, chosen because iNaturalist holds many
          openly licensed photos of them. Rarer species are not included yet.
        </p>
      </header>
      <SpeciesBrowser />
    </div>
  );
}
