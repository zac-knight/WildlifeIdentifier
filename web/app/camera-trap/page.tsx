import type { Metadata } from "next";
import { CameraTrapApp } from "@/components/camera-trap-app";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export const metadata: Metadata = {
  title: "Camera trap batches",
  description: "Filter empty camera trap frames and suggest bird species, on your own device.",
};

export default function CameraTrapPage() {
  return (
    <div className="mx-auto max-w-6xl space-y-6 px-4 py-10">
      <header className="max-w-2xl space-y-2">
        <h1 className="text-4xl font-semibold">Camera trap batches</h1>
        <p className="text-muted-foreground">
          Upload a batch of images or a zip file. WildLens finds the frames with animals, crops them
          and suggests a bird species for each one.
        </p>
      </header>
      <Alert>
        <AlertTitle>This classifier only knows birds</AlertTitle>
        <AlertDescription>
          It was trained on iNaturalist photos, not camera trap images. Camera traps often capture
          mammals and night-time infrared images, so any non-bird will be given a wrong bird name.
          Review each suggestion before relying on it.
        </AlertDescription>
      </Alert>
      <CameraTrapApp />
    </div>
  );
}
