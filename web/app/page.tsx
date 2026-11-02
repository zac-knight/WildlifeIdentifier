import Link from "next/link";
import { buttonVariants } from "@/components/ui/button";
import { allSpecies } from "@/lib/species";

const FEATURES = [
  {
    title: "Sight and sound",
    text: "Identify a bird from a photo, a recording or both. Your files stay on your device.",
  },
  {
    title: "Place and season",
    text: "Where and when you saw it helps separate similar species, and WildLens shows what changed.",
  },
  {
    title: "Honest about doubt",
    text: "If it cannot name a species, it falls back to the genus or family, or says it is not sure.",
  },
  {
    title: "Climate futures",
    text: "Each species page maps where suitable climate may shift, with the uncertainty shown.",
  },
];

export default function Home() {
  return (
    <div className="mx-auto max-w-6xl px-4">
      <section className="max-w-3xl space-y-6 py-16 sm:py-24">
        <h1 className="text-5xl font-semibold tracking-tight sm:text-6xl">WildLens</h1>
        <p className="text-xl leading-8 text-muted-foreground">
          Identify {allSpecies.length} Australian birds from photos and sound, and see how far to
          trust the answer.
        </p>
        <div className="flex flex-wrap gap-3">
          <Link href="/identify" className={buttonVariants({ size: "lg" })}>
            Identify a bird
          </Link>
          <Link href="/evaluation" className={buttonVariants({ size: "lg", variant: "outline" })}>
            See how well it works
          </Link>
        </div>
      </section>
      <section
        aria-label="Features"
        className="grid gap-8 border-t border-border py-10 sm:grid-cols-2"
      >
        {FEATURES.map((feature) => (
          <div key={feature.title} className="space-y-1.5">
            <h2 className="text-xl font-semibold">{feature.title}</h2>
            <p className="text-sm text-muted-foreground">{feature.text}</p>
          </div>
        ))}
      </section>
    </div>
  );
}
