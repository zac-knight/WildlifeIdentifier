"use client";

import Link from "next/link";
import { useDeferredValue, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { allSpecies, displayName, iucnLabel } from "@/lib/species";

export function SpeciesBrowser() {
  const [query, setQuery] = useState("");
  const deferred = useDeferredValue(query);
  const matches = useMemo(() => {
    const term = deferred.trim().toLowerCase();
    const list = term
      ? allSpecies.filter((species) =>
          [species.name, species.commonName, species.family].some((value) =>
            value?.toLowerCase().includes(term),
          ),
        )
      : allSpecies;
    return [...list].sort((a, b) => displayName(a).localeCompare(displayName(b)));
  }, [deferred]);

  return (
    <div className="space-y-4">
      <div className="max-w-sm space-y-1.5">
        <Label htmlFor="species-search">Search by name or family</Label>
        <Input
          id="species-search"
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </div>
      <p className="text-sm text-muted-foreground" role="status">
        {matches.length} {matches.length === 1 ? "species" : "species"}
      </p>
      {matches.length === 0 ? (
        <p className="text-sm">No species match that search. WildLens covers 200 species so far.</p>
      ) : (
        <ul className="grid gap-x-6 gap-y-px sm:grid-cols-2 lg:grid-cols-3">
          {matches.map((species) => (
            <li key={species.slug} className="border-b border-border">
              <Link
                href={`/species/${species.slug}`}
                className="flex items-baseline justify-between gap-2 py-2 hover:bg-muted/60"
              >
                <span>
                  <span className="font-medium">{displayName(species)}</span>
                  <span className="block text-xs text-muted-foreground italic">{species.name}</span>
                </span>
                {species.threatened ? (
                  <Badge variant="outline" className="shrink-0">
                    {iucnLabel(species.iucn)}
                  </Badge>
                ) : null}
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
