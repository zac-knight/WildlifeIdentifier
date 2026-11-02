import type { Metadata } from "next";
import { IdentifyApp } from "@/components/identify-app";

export const metadata: Metadata = {
  title: "Identify a bird",
  description: "Identify an Australian bird from a photo or a sound recording, on your own device.",
};

export default function IdentifyPage() {
  return (
    <div className="mx-auto max-w-6xl space-y-8 px-4 py-10">
      <header className="max-w-2xl space-y-2">
        <h1 className="text-4xl font-semibold">Identify a bird</h1>
        <p className="text-muted-foreground">
          Add a photo, a sound recording or both. Adding where and when you saw the bird helps
          WildLens separate similar species.
        </p>
      </header>
      <IdentifyApp />
    </div>
  );
}
