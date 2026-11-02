import type { Metadata } from "next";
import type { ReactNode } from "react";
import { type ReliabilityBin, ReliabilityChart } from "@/components/charts";
import metricsData from "@/data/metrics.json";

export const metadata: Metadata = {
  title: "Evaluation",
  description: "How well WildLens performs, including where it falls short.",
};

interface Summary {
  n: number;
  species: number;
  top1: number;
  top5: number;
  macro: number;
}

/** Metrics are produced by scripts. Anything not yet measured is reported as such, never filled in. */
type Loose = Record<string, never> & Record<string, unknown>;
const metrics = metricsData as unknown as Record<string, Loose | undefined>;

function get<T>(value: unknown, ...path: string[]): T | undefined {
  let current = value;
  for (const key of path) {
    if (current === null || typeof current !== "object") return undefined;
    current = (current as Record<string, unknown>)[key];
  }
  return (current ?? undefined) as T | undefined;
}

function percent(value: number | undefined | null, digits = 1): string {
  return typeof value === "number" ? `${(value * 100).toFixed(digits)}%` : "Not measured";
}

function number(value: number | undefined | null, digits = 3): string {
  return typeof value === "number" ? value.toFixed(digits) : "Not measured";
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-3">
      <h2 className="text-2xl font-semibold">{title}</h2>
      {children}
    </section>
  );
}

function MetricTable({
  caption,
  rows,
}: {
  caption: string;
  rows: { label: string; summary: Summary | undefined }[];
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[32rem] text-left text-sm tabular-nums">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-border text-muted-foreground">
            <th scope="col" className="py-2 pr-4 font-medium">
              Setup
            </th>
            <th scope="col" className="pr-4 font-medium">
              Top 1
            </th>
            <th scope="col" className="pr-4 font-medium">
              Top 5
            </th>
            <th scope="col" className="pr-4 font-medium">
              Macro
            </th>
            <th scope="col" className="font-medium">
              Test items
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label} className="border-b border-border/60">
              <th scope="row" className="py-2 pr-4 font-normal">
                {row.label}
              </th>
              <td className="pr-4">{percent(row.summary?.top1)}</td>
              <td className="pr-4">{percent(row.summary?.top5)}</td>
              <td className="pr-4">{percent(row.summary?.macro)}</td>
              <td>{row.summary?.n?.toLocaleString("en-AU") ?? "Not measured"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Pending({ children }: { children: ReactNode }) {
  return <p className="text-sm text-muted-foreground">{children}</p>;
}

export default function EvaluationPage() {
  const image = metrics.image;
  const probe = metrics.dinov2_probe;
  const fewshot = metrics.fewshot;
  const audio = metrics.audio;
  const sdm = metrics.sdm;
  const exported = metrics.export;
  const near = get<Record<string, number>>(image, "trust", "ood", "near");
  const far = get<Record<string, number>>(image, "trust", "ood", "far");
  const before = get<ReliabilityBin[]>(image, "random", "calibration", "reliability_before");
  const after = get<ReliabilityBin[]>(image, "random", "calibration", "reliability_after");
  const pairs = get<{ a: string; b: string; errors: number }[]>(image, "random", "confused_pairs");
  const variants = get<Record<string, Record<string, number | boolean>>>(
    exported,
    "image",
    "variants",
  );

  return (
    <div className="mx-auto max-w-4xl space-y-12 px-4 py-10">
      <header className="max-w-2xl space-y-2">
        <h1 className="text-4xl font-semibold">Evaluation</h1>
        <p className="text-muted-foreground">
          Every number here comes from scripts in the repository, on test data the models never
          trained on. Macro accuracy gives each species equal weight, so common birds cannot hide
          poor results on others.
        </p>
      </header>

      <Section title="Image classifier and location">
        <p className="max-w-prose text-sm">
          The random split keeps each photographer in one split per species. The geographic split
          holds out whole 2° map blocks, which is harder and closer to using the app somewhere new.
          Rows with location use only test photos that have a precise place and date.
        </p>
        <MetricTable
          caption="Accuracy by model and split"
          rows={[
            { label: "DINOv2 linear probe, random", summary: get(probe, "random") },
            { label: "DINOv2 linear probe, geographic", summary: get(probe, "geo") },
            { label: "EfficientNetV2-B2, random", summary: get(image, "random", "test") },
            { label: "EfficientNetV2-B2, geographic", summary: get(image, "geographic", "test") },
            {
              label: "Image only, photos with location, random",
              summary: get(image, "random", "fusion", "image_only"),
            },
            {
              label: "Image + location, random",
              summary: get(image, "random", "fusion", "image_plus_prior"),
            },
            {
              label: "Image only, photos with location, geographic",
              summary: get(image, "geographic", "fusion", "image_only"),
            },
            {
              label: "Image + location, geographic",
              summary: get(image, "geographic", "fusion", "image_plus_prior"),
            },
          ]}
        />
        {pairs ? (
          <details className="text-sm">
            <summary className="cursor-pointer text-muted-foreground">
              Most confused species pairs
            </summary>
            <ol className="mt-2 list-decimal space-y-0.5 pl-5">
              {pairs.slice(0, 15).map((pair) => (
                <li key={`${pair.a}-${pair.b}`}>
                  <span className="italic">{pair.a}</span> and{" "}
                  <span className="italic">{pair.b}</span>: {pair.errors} errors
                </li>
              ))}
            </ol>
          </details>
        ) : null}
      </Section>

      <Section title="Calibration">
        <p className="max-w-prose text-sm">
          When WildLens says 80%, it should be right about 80% of the time. Expected calibration
          error fell from {percent(get(image, "random", "calibration", "ece_before"))} to{" "}
          {percent(get(image, "random", "calibration", "ece_after"))} with temperature scaling, and
          is {percent(get(image, "random", "fusion", "ece_after_fusion"))} after adding location.
        </p>
        {before && after ? (
          <ReliabilityChart
            before={before}
            after={after}
            title="Reliability on the random test split"
          />
        ) : (
          <Pending>The reliability diagram appears once the evaluation has been run.</Pending>
        )}
      </Section>

      <Section title="Recognising unfamiliar inputs">
        <p className="max-w-prose text-sm">
          Near means Australian birds outside the model&apos;s species list, which is hard. Far
          means other wildlife and plants. AUROC is 100% for perfect separation and 50% for
          guessing. False accept is the share of unknown photos still treated as known, at a
          threshold keeping 95% of known-species validation photos.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[28rem] text-left text-sm tabular-nums">
            <caption className="sr-only">Unknown input detection</caption>
            <thead>
              <tr className="border-b border-border text-muted-foreground">
                <th scope="col" className="py-2 pr-4 font-medium">
                  Unknowns
                </th>
                <th scope="col" className="pr-4 font-medium">
                  Energy AUROC
                </th>
                <th scope="col" className="pr-4 font-medium">
                  Top-probability AUROC
                </th>
                <th scope="col" className="font-medium">
                  Energy false accept
                </th>
              </tr>
            </thead>
            <tbody>
              {[
                { label: "Near", values: near },
                { label: "Far", values: far },
              ].map((row) => (
                <tr key={row.label} className="border-b border-border/60">
                  <th scope="row" className="py-2 pr-4 font-normal">
                    {row.label}
                  </th>
                  <td className="pr-4">{percent(row.values?.energy)}</td>
                  <td className="pr-4">{percent(row.values?.msp)}</td>
                  <td>{percent(row.values?.energy_false_accept)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Species with few training photos">
        <p className="max-w-prose text-sm">
          For this test, 60 species were limited to 10 training photos. Prototype matching is only
          used when the classifier is unsure.
        </p>
        <MetricTable
          caption="Few-shot results on limited species"
          rows={[
            {
              label: "Same species, all photos",
              summary: get(image, "tail_with_all_images", "tail_species"),
            },
            { label: "Classifier alone", summary: get(fewshot, "classifier_alone", "tail") },
            {
              label: "Classifier + prototypes",
              summary: get(fewshot, "classifier_plus_prototypes", "tail"),
            },
            { label: "Prototypes alone", summary: get(fewshot, "prototypes_alone", "tail") },
          ]}
        />
      </Section>

      <Section title="Sound">
        <p className="max-w-prose text-sm">
          Each recordist is in one split per species. Fewer recordings exist than photos, so species
          with little audio remain weak.
        </p>
        <MetricTable
          caption="Sound model"
          rows={[{ label: "Sound model, random split", summary: get(audio, "test") }]}
        />
      </Section>

      <Section title="Habitat models">
        {sdm ? (
          <dl className="grid max-w-lg grid-cols-[1fr_auto] gap-x-6 gap-y-1 text-sm tabular-nums">
            <dt>Species with spatial cross validation</dt>
            <dd>{String(get(sdm, "species_with_cv"))}</dd>
            <dt>Median AUC</dt>
            <dd>{number(get(sdm, "auc_median"), 2)}</dd>
            <dt>AUC, 10th to 90th percentile</dt>
            <dd>
              {number(get(sdm, "auc_p10"), 2)} to {number(get(sdm, "auc_p90"), 2)}
            </dd>
            <dt>Median Boyce index</dt>
            <dd>{number(get(sdm, "boyce_median"), 2)}</dd>
            <dt>Boyce, 10th to 90th percentile</dt>
            <dd>
              {number(get(sdm, "boyce_p10"), 2)} to {number(get(sdm, "boyce_p90"), 2)}
            </dd>
          </dl>
        ) : (
          <Pending>Habitat model results appear once they have been run.</Pending>
        )}
      </Section>

      <Section title="Species profiles">
        <Pending>
          Unsupported claim rate: not measured. AI-written profiles need a free API key that has not
          been set up, so this copy shows source facts directly.
        </Pending>
      </Section>

      <Section title="Model files">
        {variants ? (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[28rem] text-left text-sm tabular-nums">
              <caption className="sr-only">Image model variants</caption>
              <thead>
                <tr className="border-b border-border text-muted-foreground">
                  <th scope="col" className="py-2 pr-4 font-medium">
                    Variant
                  </th>
                  <th scope="col" className="pr-4 font-medium">
                    Size
                  </th>
                  <th scope="col" className="pr-4 font-medium">
                    Top 1 on sample
                  </th>
                  <th scope="col" className="font-medium">
                    Largest logit gap from PyTorch
                  </th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(variants).map(([name, row]) => (
                  <tr key={name} className="border-b border-border/60">
                    <th scope="row" className="py-2 pr-4 font-normal">
                      {name}
                    </th>
                    <td className="pr-4">{number(row.megabytes as number, 1)} MB</td>
                    <td className="pr-4">{percent(row.val_sample_top1 as number)}</td>
                    <td>{number(row.max_logit_difference_vs_pytorch as number, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Pending>File sizes appear once models have been exported.</Pending>
        )}
      </Section>

      <Section title="Known limitations">
        <ul className="max-w-prose list-disc space-y-1 pl-5 text-sm">
          <li>Only species with enough photos are covered, so the rarest birds get wrong names.</li>
          <li>
            Photos come from iNaturalist users, who favour cities, roads and easily seen birds.
          </li>
          <li>For very common species, only the newest 10,000 records could be sampled.</li>
          <li>Conservation status is the global IUCN category, not Australian listings.</li>
          <li>
            Habitat models use climate only at about 25 km resolution. They ignore land use and
            dispersal.
          </li>
          <li>
            The classifier was not trained on camera trap images and will name non-birds as birds.
          </li>
        </ul>
      </Section>
    </div>
  );
}
