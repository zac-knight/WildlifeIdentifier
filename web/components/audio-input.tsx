"use client";

import { Mic, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RECORDING_CONSTRAINTS } from "@/lib/media";

interface AudioInputProps {
  fileName: string | null;
  onAudio: (audio: Blob, name: string) => void;
}

export function AudioInput({ fileName, onAudio }: AudioInputProps) {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const stop = useRef<(() => void) | null>(null);

  useEffect(() => () => stop.current?.(), []);

  async function start() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia(RECORDING_CONSTRAINTS);
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      recorder.ondataavailable = (event) => chunks.push(event.data);
      recorder.onstop = () =>
        onAudio(new Blob(chunks, { type: recorder.mimeType }), "Your recording");

      const context = new AudioContext();
      const analyser = context.createAnalyser();
      analyser.fftSize = 1024;
      context.createMediaStreamSource(stream).connect(analyser);
      const bins = new Uint8Array(analyser.frequencyBinCount);
      let frame = 0;
      const draw = () => {
        const element = canvas.current;
        const pen = element?.getContext("2d");
        if (element && pen) {
          // Scroll left by one pixel, then paint the newest moment at the right edge.
          pen.drawImage(element, -1, 0);
          analyser.getByteFrequencyData(bins);
          const usable = Math.floor(bins.length * 0.75);
          for (let y = 0; y < element.height; y += 1) {
            const value = bins[Math.floor(((element.height - y) / element.height) * usable)];
            pen.fillStyle = `rgb(${value}, ${Math.round(value * 0.75)}, ${Math.round(value * 0.35)})`;
            pen.fillRect(element.width - 1, y, 1, 1);
          }
        }
        frame = requestAnimationFrame(draw);
      };
      draw();

      stop.current = () => {
        cancelAnimationFrame(frame);
        if (recorder.state !== "inactive") recorder.stop();
        stream.getTracks().forEach((track) => track.stop());
        void context.close();
        stop.current = null;
        setRecording(false);
      };
      recorder.start();
      setRecording(true);
    } catch {
      setError("The microphone is unavailable. You can upload a recording instead.");
    }
  }

  return (
    <div className="space-y-3">
      <div className="space-y-1.5">
        <Label htmlFor="audio-file">Sound recording</Label>
        <Input
          id="audio-file"
          type="file"
          accept="audio/*,video/mp4,.m4a,.mp3,.wav"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onAudio(file, file.name);
          }}
        />
      </div>
      <div className="flex flex-wrap items-center gap-3">
        {recording ? (
          <Button variant="destructive" onClick={() => stop.current?.()}>
            <Square aria-hidden /> Stop recording
          </Button>
        ) : (
          <Button variant="outline" onClick={() => void start()}>
            <Mic aria-hidden /> Record
          </Button>
        )}
        {fileName && !recording ? (
          <span className="text-sm text-muted-foreground">Ready: {fileName}</span>
        ) : null}
      </div>
      <canvas
        ref={canvas}
        width={600}
        height={100}
        aria-label={recording ? "Live spectrogram of the recording" : undefined}
        aria-hidden={!recording}
        className={recording ? "h-24 w-full rounded bg-black" : "hidden"}
      />
      {error ? (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      ) : null}
    </div>
  );
}
