import type { Metadata } from "next";
import creditsData from "@/data/credits.json";

export const metadata: Metadata = {
  title: "About",
  description: "How WildLens works, its data sources, and credits to every contributor.",
};

interface Person {
  n: string;
  p: number;
  r: number;
}

const credits = creditsData as {
  people: Person[];
  licences: Record<string, Record<string, number>>;
};

const SOURCES = [
  {
    name: "iNaturalist",
    url: "https://www.inaturalist.org",
    use: "Bird photos and sound recordings, used under each file's Creative Commons licence.",
  },
  {
    name: "GBIF",
    url: "https://www.gbif.org",
    use: "Occurrence records for the location and habitat models, and IUCN categories.",
  },
  {
    name: "NASA NEX-GDDP-CMIP6",
    url: "https://registry.opendata.aws/nex-gddp-cmip6/",
    use: "Present and future climate, under a CC0 licence.",
  },
  {
    name: "Wikipedia",
    url: "https://en.wikipedia.org",
    use: "Species summaries under CC BY-SA 4.0.",
  },
  {
    name: "OpenStreetMap",
    url: "https://www.openstreetmap.org/copyright",
    use: "Base maps.",
  },
];

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-4xl space-y-10 px-4 py-10">
      <header className="max-w-2xl space-y-2">
        <h1 className="text-4xl font-semibold">About WildLens</h1>
        <p className="text-muted-foreground">
          WildLens is a non-commercial portfolio project. It identifies Australian birds on your own
          device and says when it is unsure.
        </p>
      </header>

      <section className="max-w-prose space-y-2">
        <h2 className="text-2xl font-semibold">Your privacy</h2>
        <p className="text-sm">
          Photos and recordings are processed in your browser and are not uploaded. If you send
          feedback, WildLens stores the species, confidence and an approximate place. Places are
          blurred further for threatened species.
        </p>
      </section>

      <section className="space-y-2">
        <h2 className="text-2xl font-semibold">Data sources</h2>
        <ul className="max-w-prose space-y-2 text-sm">
          {SOURCES.map((source) => (
            <li key={source.name}>
              <a href={source.url} className="font-medium underline" rel="noreferrer">
                {source.name}
              </a>
              : {source.use}
            </li>
          ))}
        </ul>
      </section>

      <section className="max-w-prose space-y-2">
        <h2 className="text-2xl font-semibold">Models</h2>
        <p className="text-sm">
          Models were trained on non-commercial data, so they are for non-commercial use. Full
          details are in the{" "}
          <a
            href="https://github.com/zac-knight/WildlifeIdentifier/tree/main/docs"
            className="underline"
            rel="noreferrer"
          >
            model cards
          </a>
          .
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-2xl font-semibold">Photographers and recordists</h2>
        <p className="max-w-prose text-sm">
          WildLens exists because {credits.people.length.toLocaleString("en-AU")} people shared
          their work under open licences. Thank you.
        </p>
        <details>
          <summary className="cursor-pointer text-sm text-muted-foreground">
            Show all contributors
          </summary>
          <ul className="mt-3 columns-1 gap-6 text-xs sm:columns-2 lg:columns-3">
            {credits.people.map((person) => (
              <li key={person.n} className="break-inside-avoid py-0.5">
                {person.n}
                <span className="text-muted-foreground">
                  {" "}
                  {[
                    person.p ? `${person.p} photo${person.p === 1 ? "" : "s"}` : null,
                    person.r ? `${person.r} recording${person.r === 1 ? "" : "s"}` : null,
                  ]
                    .filter(Boolean)
                    .join(", ")}
                </span>
              </li>
            ))}
          </ul>
        </details>
      </section>
    </div>
  );
}
