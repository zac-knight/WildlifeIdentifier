"use client";

import { useId, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { type Analysis, coarsenForStorage } from "@/lib/ml/analysis";
import { entropy } from "@/lib/ml/math";
import type { ImageMetadata } from "@/lib/ml/metadata";
import { allSpecies, displayName, speciesByName } from "@/lib/species";

interface FeedbackFormProps {
  metadata: ImageMetadata;
  analysis: Analysis;
  place: { latitude: number; longitude: number } | null;
  date: string;
}

type Status = "idle" | "sending" | "sent" | "unavailable" | "failed";

export function FeedbackForm({ metadata, analysis, place, date }: FeedbackFormProps) {
  const [status, setStatus] = useState<Status>("idle");
  const [correcting, setCorrecting] = useState(false);
  const [correction, setCorrection] = useState("");
  const listId = useId();
  const top = analysis.candidates[0];

  async function send(correctName: string, confirmed: boolean) {
    setStatus("sending");
    const threatened =
      speciesByName(correctName)?.threatened || speciesByName(top.name)?.threatened || false;
    try {
      const response = await fetch("/api/feedback", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          modelVersion: metadata.version,
          predictedSpecies: top.name,
          correctSpecies: correctName,
          confirmed,
          confidence: top.probability,
          entropy: entropy(analysis.probabilities),
          latitude: place ? coarsenForStorage(place.latitude, threatened) : null,
          longitude: place ? coarsenForStorage(place.longitude, threatened) : null,
          observedOn: date || null,
        }),
      });
      if (response.status === 503) setStatus("unavailable");
      else setStatus(response.ok ? "sent" : "failed");
    } catch {
      setStatus("failed");
    }
  }

  if (status === "sent") {
    return (
      <p role="status" className="text-sm">
        Thanks. Your feedback has been saved for review.
      </p>
    );
  }
  if (status === "unavailable") {
    return (
      <p role="status" className="text-sm text-muted-foreground">
        Feedback is switched off on this copy of WildLens.
      </p>
    );
  }

  const options = allSpecies.map((species) => ({
    value: species.name,
    text: `${displayName(species)} (${species.name})`,
  }));
  const chosen =
    options.find((option) => option.text === correction || option.value === correction)?.value ??
    null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Was the top match right?</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <p className="text-muted-foreground">
          This saves the species, confidence and an approximate place. It never includes your photo
          or recording.
        </p>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            disabled={status === "sending"}
            onClick={() => void send(top.name, true)}
          >
            Yes, that&apos;s right
          </Button>
          <Button variant="outline" onClick={() => setCorrecting((value) => !value)}>
            No, correct it
          </Button>
        </div>
        {correcting ? (
          <form
            className="space-y-2"
            onSubmit={(event) => {
              event.preventDefault();
              if (chosen) void send(chosen, false);
            }}
          >
            <Label htmlFor={`${listId}-input`}>The correct species</Label>
            <Input
              id={`${listId}-input`}
              list={listId}
              value={correction}
              onChange={(event) => setCorrection(event.target.value)}
              placeholder="Start typing a name"
            />
            <datalist id={listId}>
              {options.map((option) => (
                <option key={option.value} value={option.text} />
              ))}
            </datalist>
            <Button type="submit" disabled={!chosen || status === "sending"}>
              Send correction
            </Button>
          </form>
        ) : null}
        {status === "failed" ? (
          <p role="alert" className="text-destructive">
            Your feedback could not be sent. Please try again.
          </p>
        ) : null}
      </CardContent>
    </Card>
  );
}
