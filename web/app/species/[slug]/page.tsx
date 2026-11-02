import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { HabitatMapLoader } from "@/components/habitat-map-loader";
import { SpeciesCardPanel } from "@/components/species-card-panel";
import { Badge } from "@/components/ui/badge";
import { allSpecies, displayName, iucnLabel, speciesBySlug } from "@/lib/species";

interface PageProps {
  params: Promise<{ slug: string }>;
}

export function generateStaticParams() {
  return allSpecies.map((species) => ({ slug: species.slug }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const species = speciesBySlug((await params).slug);
  return species
    ? { title: displayName(species), description: `Profile and habitat maps for ${species.name}.` }
    : {};
}

export default async function SpeciesDetailPage({ params }: PageProps) {
  const species = speciesBySlug((await params).slug);
  if (!species) notFound();

  return (
    <article className="mx-auto max-w-4xl space-y-10 px-4 py-10">
      <header className="grid gap-6 sm:grid-cols-[minmax(0,1fr)_16rem]">
        <div className="space-y-3">
          <h1 className="text-4xl font-semibold">{displayName(species)}</h1>
          <p className="text-lg text-muted-foreground italic">{species.name}</p>
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
            <dt className="text-muted-foreground">Family</dt>
            <dd>{species.family ?? "Unknown"}</dd>
            <dt className="text-muted-foreground">Conservation status</dt>
            <dd className="flex items-center gap-2">
              {iucnLabel(species.iucn)}
              {species.threatened ? <Badge variant="outline">Threatened</Badge> : null}
            </dd>
            <dt className="text-muted-foreground">Sound model</dt>
            <dd>
              {species.recordings > 0
                ? `Trained on ${species.recordings} recordings`
                : "No recordings available"}
            </dd>
          </dl>
          {species.iucn ? (
            <p className="text-xs text-muted-foreground">
              Status is the global IUCN Red List category, via GBIF. Australian listings can differ.
            </p>
          ) : null}
        </div>
        {species.photoUrl ? (
          <figure className="space-y-1">
            {/* eslint-disable-next-line @next/next/no-img-element -- Remote photo of unknown size */}
            <img
              src={species.photoUrl}
              alt={displayName(species)}
              loading="lazy"
              className="aspect-square w-full rounded-lg border border-border object-cover"
            />
            {species.photoAttribution ? (
              <figcaption className="text-xs text-muted-foreground">
                {species.photoAttribution}
              </figcaption>
            ) : null}
          </figure>
        ) : null}
      </header>

      <section className="space-y-3">
        <h2 className="text-2xl font-semibold">About this bird</h2>
        <SpeciesCardPanel slug={species.slug} />
      </section>

      <section className="space-y-3">
        <h2 className="text-2xl font-semibold">Habitat and climate</h2>
        {species.sdm ? (
          <>
            <p className="max-w-prose text-sm text-muted-foreground">
              Where the climate suits this bird according to a model of its recorded sightings.
              Tested on regions it did not see in training: AUC{" "}
              {species.sdm.auc?.toFixed(2) ?? "not available"}, Boyce index{" "}
              {species.sdm.boyce?.toFixed(2) ?? "not available"}.
            </p>
            <HabitatMapLoader species={species} />
          </>
        ) : (
          <p className="text-sm text-muted-foreground">
            There are too few mappable records to model this species.
          </p>
        )}
      </section>
    </article>
  );
}
