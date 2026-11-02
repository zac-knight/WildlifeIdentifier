"use client";

import { useEffect, useState } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import type { SpeciesCard } from "@/lib/species-card-types";

export function SpeciesCardPanel({ slug }: { slug: string }) {
  const [card, setCard] = useState<SpeciesCard | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/species/${slug}/card`)
      .then((response) => {
        if (!response.ok) throw new Error("Unavailable");
        return response.json() as Promise<SpeciesCard>;
      })
      .then((value) => {
        if (!cancelled) setCard(value);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [slug]);

  if (failed) {
    return (
      <p role="alert" className="text-sm text-muted-foreground">
        The species profile could not be loaded right now.
      </p>
    );
  }
  if (!card) {
    return (
      <div className="space-y-2" aria-label="Loading species profile">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-11/12" />
        <Skeleton className="h-4 w-3/4" />
      </div>
    );
  }

  const number = new Map(card.sources.map((source, index) => [source.id, index + 1]));
  return (
    <div className="space-y-4">
      <div className="max-w-prose space-y-1 leading-7">
        {card.claims.map((claim, index) => {
          const source = card.sources.find((item) => item.id === claim.sourceId);
          return (
            <span key={index}>
              {claim.text}{" "}
              {source ? (
                <a
                  href={source.url}
                  target="_blank"
                  rel="noreferrer"
                  aria-label={`Source: ${source.title}`}
                  className="align-super text-xs text-primary underline"
                >
                  {number.get(claim.sourceId)}
                </a>
              ) : null}{" "}
            </span>
          );
        })}
      </div>
      <ol className="list-decimal space-y-1 pl-5 text-xs text-muted-foreground">
        {card.sources.map((source) => (
          <li key={source.id}>
            <a href={source.url} target="_blank" rel="noreferrer" className="underline">
              {source.title}
            </a>{" "}
            ({source.licence})
          </li>
        ))}
      </ol>
      <p className="text-xs text-muted-foreground">
        {card.mode === "llm"
          ? "Written by an AI model using only these sources. Every sentence links to its source."
          : "Shown directly from these sources."}
      </p>
    </div>
  );
}
