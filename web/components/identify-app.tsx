"use client";

import { AlertTriangle, HelpCircle, Loader2, LocateFixed } from "lucide-react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useMemo, useRef, useState } from "react";
import { AudioInput } from "@/components/audio-input";
import { CamOverlay } from "@/components/cam-overlay";
import { FeedbackForm } from "@/components/feedback-form";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { type LoadedImage, decodeAudio, loadImage, readPhotoDetails } from "@/lib/media";
import {
  type Analysis,
  analyse,
  dayOfYear,
  explainChange,
  isConfident,
  isInAustralia,
} from "@/lib/ml/analysis";
import { averageLogits, energyClips } from "@/lib/ml/audio";
import { classActivationMap } from "@/lib/ml/cam";
import { getClient } from "@/lib/ml/client";
import { confidenceLabel } from "@/lib/ml/fusion";
import { fetchMetadata, type ImageMetadata, MODEL_BASE_URL } from "@/lib/ml/metadata";
import { fetchCached } from "@/lib/ml/model-cache";
import { displayName, speciesByName } from "@/lib/species";

const LocationMap = dynamic(
  () => import("@/components/location-map").then((module) => module.LocationMap),
  { ssr: false, loading: () => <div className="h-64 w-full animate-pulse rounded-lg bg-muted" /> },
);

interface Place {
  latitude: number;
  longitude: number;
  source: string;
}

interface Result {
  analysis: Analysis;
  metadata: ImageMetadata;
  cam?: { map: Float32Array; size: number };
  provider: string;
  milliseconds: number;
  usedAudio: boolean;
  usedPhoto: boolean;
}

function localDate(date: Date): string {
  const offset = date.getTimezoneOffset() * 60_000;
  return new Date(date.getTime() - offset).toISOString().slice(0, 10);
}

function label(name: string): string {
  const species = speciesByName(name);
  return species ? displayName(species) : name;
}

function headline(analysis: Analysis): { title: string; detail: string } {
  const { decision } = analysis;
  if (analysis.unfamiliar) {
    return {
      title: "This may not be a bird WildLens knows",
      detail:
        "The photo looks unlike the 200 Australian species the model was trained on. It could be another species, or not a bird. Treat the options below as weak guesses.",
    };
  }
  switch (decision.level) {
    case "species": {
      const name = analysis.candidates[0].name;
      return {
        title: `${confidenceLabel(decision.confidence)} ${label(name)}`,
        detail: `About ${Math.round(decision.confidence * 100)}% confidence. Check it against a field guide if the identification matters.`,
      };
    }
    case "genus":
      return {
        title: `Probably in the genus ${decision.name}`,
        detail: `About ${Math.round(decision.confidence * 100)}% confidence in the genus. WildLens is not sure which species it is.`,
      };
    case "family":
      return {
        title: `Probably in the family ${decision.name}`,
        detail: `About ${Math.round(decision.confidence * 100)}% confidence in the family. WildLens is not sure of the genus or species.`,
      };
    case "unknown":
      return {
        title: "I'm not sure",
        detail:
          "No species, genus or family is likely enough to name. A clearer view or a sound recording may help.",
      };
  }
}

export function IdentifyApp() {
  const [image, setImage] = useState<LoadedImage | null>(null);
  const [audio, setAudio] = useState<{ blob: Blob; name: string } | null>(null);
  const [place, setPlace] = useState<Place | null>(null);
  const [date, setDate] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [showMap, setShowMap] = useState(false);
  const running = useRef(false);

  const outside = place ? !isInAustralia(place.latitude, place.longitude) : false;

  async function choosePhoto(file: File) {
    setError(null);
    setResult(null);
    setNotice(null);
    try {
      const [loaded, details] = await Promise.all([loadImage(file), readPhotoDetails(file)]);
      setImage(loaded);
      if (details.date) setDate(localDate(details.date));
      if (details.latitude !== undefined && details.longitude !== undefined) {
        setPlace({
          latitude: details.latitude,
          longitude: details.longitude,
          source: "the photo's location data",
        });
      } else {
        setNotice("This photo has no location data. Add a place for a better result.");
      }
    } catch {
      setError("That file could not be read as an image. Try a JPEG or PNG.");
    }
  }

  function useDeviceLocation() {
    setError(null);
    if (!("geolocation" in navigator)) {
      setError("This browser cannot share a location. Use the map instead.");
      setShowMap(true);
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (position) =>
        setPlace({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          source: "your device",
        }),
      () => {
        setError("Location permission was not given. You can drop a pin on the map.");
        setShowMap(true);
      },
    );
  }

  async function run() {
    if (running.current || (!image && !audio)) return;
    running.current = true;
    setError(null);
    setResult(null);
    const trackProgress = (loaded: number, total: number | null) =>
      setProgress(total ? Math.round((loaded / total) * 100) : null);
    try {
      const client = getClient();
      setStatus("Loading model details");
      const metadata = await fetchMetadata("image");
      let provider = "wasm";
      let milliseconds = 0;
      let imageLogits: Float32Array | undefined;
      let features: Float32Array | undefined;

      if (image) {
        setStatus("Downloading the photo model (first time only)");
        provider = await client.load("image", metadata.files, trackProgress);
        setProgress(null);
        setStatus("Looking at the photo");
        const copy = new Uint8ClampedArray(image.rgba);
        const output = await client.image(copy, image.width, image.height);
        imageLogits = output.logits;
        features = output.features;
        milliseconds += output.milliseconds;
      }

      let audioLogits: Float64Array | undefined;
      let audioMetadata;
      if (audio) {
        audioMetadata = await fetchMetadata("audio");
        setStatus("Downloading the sound model (first time only)");
        provider = await client.load("audio", audioMetadata.files, trackProgress);
        setProgress(null);
        setStatus("Listening to the recording");
        const waveform = await decodeAudio(audio.blob);
        const clips = energyClips(waveform, audioMetadata.clips) as Float32Array<ArrayBuffer>[];
        const output = await client.audio(clips);
        audioLogits = averageLogits(output.logits);
        milliseconds += output.milliseconds;
      }

      let prior: Float32Array | undefined;
      if (place && !outside) {
        setStatus("Checking what is expected at this place and time");
        const priorMetadata = await fetchMetadata("prior");
        await client.load("prior", priorMetadata.files, trackProgress);
        setProgress(null);
        prior = await client.prior(
          place.latitude,
          place.longitude,
          dayOfYear(new Date(`${date || localDate(new Date())}T12:00:00`)),
        );
      }

      const analysis = analyse({
        image: metadata,
        audio: audioMetadata,
        imageLogits,
        audioLogits,
        prior,
      });

      let cam: Result["cam"];
      if (image && features && imageLogits) {
        const weights = await fetchCached(
          `${MODEL_BASE_URL}/image/${metadata.weightsFile.path}`,
          metadata.weightsFile.sha256,
        );
        const matrix = new Float32Array(weights.buffer);
        const channels = metadata.featureChannels;
        const best = analysis.candidates[0].index;
        const row = matrix.subarray(best * channels, (best + 1) * channels);
        cam = {
          map: classActivationMap(
            features,
            channels,
            metadata.featureSize,
            metadata.featureSize,
            row,
          ),
          size: metadata.featureSize,
        };
      }
      setResult({
        analysis,
        metadata,
        cam,
        provider,
        milliseconds,
        usedAudio: Boolean(audio),
        usedPhoto: Boolean(image),
      });
    } catch (failure) {
      const message = failure instanceof Error ? failure.message : "";
      setError(
        /details|Download|Failed to fetch|integrity/i.test(message)
          ? "The models could not be downloaded. Check your connection and try again."
          : "Something went wrong while identifying this. Try again, or use a different file.",
      );
      console.warn("Identification failed", failure);
    } finally {
      setStatus(null);
      setProgress(null);
      running.current = false;
    }
  }

  const summary = useMemo(() => (result ? headline(result.analysis) : null), [result]);
  const busy = status !== null;

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
      <section aria-label="Observation details" className="space-y-6">
        <div className="space-y-1.5">
          <Label htmlFor="photo-file">Photo</Label>
          <Input
            id="photo-file"
            type="file"
            accept="image/*"
            capture={undefined}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void choosePhoto(file);
            }}
          />
        </div>

        <AudioInput
          fileName={audio?.name ?? null}
          onAudio={(blob, name) => {
            setResult(null);
            setAudio({ blob, name });
          }}
        />

        <div className="space-y-2">
          <p className="text-sm font-medium">Where was it?</p>
          {place ? (
            <p className="text-sm text-muted-foreground" data-testid="place">
              {place.latitude.toFixed(3)}, {place.longitude.toFixed(3)}, from {place.source}.
            </p>
          ) : (
            <p className="text-sm text-muted-foreground">
              Optional, but it helps separate similar species.
            </p>
          )}
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={useDeviceLocation}>
              <LocateFixed aria-hidden /> Use my location
            </Button>
            <Button variant="outline" onClick={() => setShowMap((value) => !value)}>
              {showMap ? "Hide map" : "Choose on a map"}
            </Button>
            {place ? (
              <Button variant="ghost" onClick={() => setPlace(null)}>
                Clear
              </Button>
            ) : null}
          </div>
          {showMap ? (
            <LocationMap
              latitude={place?.latitude}
              longitude={place?.longitude}
              onPick={(latitude, longitude) =>
                setPlace({ latitude, longitude, source: "the map pin" })
              }
            />
          ) : null}
          {outside ? (
            <Alert>
              <AlertTriangle aria-hidden />
              <AlertTitle>Outside the area WildLens covers</AlertTitle>
              <AlertDescription>
                The models only know Australian birds, so this place will not be used and results
                may be unreliable.
              </AlertDescription>
            </Alert>
          ) : null}
        </div>

        <div className="max-w-64 space-y-1.5">
          <Label htmlFor="observation-date">Date seen (leave empty for today)</Label>
          <Input
            id="observation-date"
            type="date"
            value={date}
            onChange={(event) => setDate(event.target.value)}
          />
        </div>

        {notice ? <p className="text-sm text-muted-foreground">{notice}</p> : null}

        <Button onClick={() => void run()} disabled={busy || (!image && !audio)} size="lg">
          {busy ? <Loader2 className="animate-spin" aria-hidden /> : null}
          Identify
        </Button>
        <p className="text-xs text-muted-foreground">
          Identification happens on your device. Your photo and recording are not uploaded.
        </p>
      </section>

      <section aria-label="Result" aria-live="polite" className="space-y-6">
        {status ? (
          <Card>
            <CardContent className="space-y-3 pt-6">
              <p className="text-sm">{status}…</p>
              <Progress value={progress} aria-label={status} />
            </CardContent>
          </Card>
        ) : null}

        {error ? (
          <Alert variant="destructive">
            <AlertTriangle aria-hidden />
            <AlertTitle>Could not identify this</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}

        {image && !result ? <CamOverlay src={image.previewUrl} alt="Your selected photo" /> : null}

        {result && summary ? (
          <>
            <Card data-testid="result-summary">
              <CardHeader>
                <CardTitle className="flex items-start gap-2 text-2xl">
                  {!isConfident(result.analysis) ? (
                    <HelpCircle className="mt-1 size-5 shrink-0 text-caution" aria-hidden />
                  ) : null}
                  {summary.title}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <p>{summary.detail}</p>
                <div className="flex flex-wrap gap-2">
                  {result.usedPhoto ? <Badge variant="secondary">Photo</Badge> : null}
                  {result.usedAudio ? <Badge variant="secondary">Sound</Badge> : null}
                  {result.analysis.usedLocation ? (
                    <Badge variant="secondary">Place and season</Badge>
                  ) : null}
                </div>
              </CardContent>
            </Card>

            {image ? (
              <CamOverlay
                src={image.previewUrl}
                alt="Your photo with a heatmap showing where the model looked"
                map={result.cam?.map}
                size={result.cam?.size}
                speciesName={label(result.analysis.candidates[0].name)}
              />
            ) : null}

            <Card>
              <CardHeader>
                <CardTitle>Closest matches</CardTitle>
              </CardHeader>
              <CardContent>
                <ol className="space-y-4" data-testid="candidates">
                  {result.analysis.candidates.map((candidate) => {
                    const species = speciesByName(candidate.name);
                    const why = explainChange(candidate.change);
                    return (
                      <li key={candidate.index} className="space-y-1">
                        <div className="flex items-baseline justify-between gap-3">
                          {species ? (
                            <Link
                              href={`/species/${species.slug}`}
                              className="font-medium underline-offset-2 hover:underline"
                            >
                              {displayName(species)}
                            </Link>
                          ) : (
                            <span className="font-medium">{candidate.name}</span>
                          )}
                          <span className="text-sm text-muted-foreground tabular-nums">
                            {Math.max(1, Math.round(candidate.probability * 100))}%
                          </span>
                        </div>
                        <Progress
                          value={candidate.probability * 100}
                          aria-label={`${label(candidate.name)}: ${Math.round(candidate.probability * 100)} percent`}
                        />
                        <p className="text-xs text-muted-foreground italic">{candidate.name}</p>
                        {why ? <p className="text-xs text-muted-foreground">{why}</p> : null}
                      </li>
                    );
                  })}
                </ol>
              </CardContent>
            </Card>

            <FeedbackForm
              metadata={result.metadata}
              analysis={result.analysis}
              place={place && !outside ? place : null}
              date={date || localDate(new Date())}
            />

            <p className="text-xs text-muted-foreground">
              Ran with {result.provider === "webgpu" ? "WebGPU" : "WebAssembly"} in{" "}
              {Math.round(result.milliseconds)} ms.
            </p>
          </>
        ) : null}

        {!image && !audio && !result && !status ? (
          <div className="rounded-lg border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
            Add a photo or a sound recording to begin.
          </div>
        ) : null}
      </section>
    </div>
  );
}
