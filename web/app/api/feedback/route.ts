import { NextResponse } from "next/server";
import { getDatabase } from "@/lib/db/client";
import { feedbackInput, hashSession, saveFeedback } from "@/lib/feedback";

const MAX_BODY_BYTES = 4_096;

export async function POST(request: Request) {
  const db = getDatabase();
  const secret = process.env.SESSION_HASH_SECRET;
  if (!db || !secret) {
    return NextResponse.json({ error: "Feedback is not enabled." }, { status: 503 });
  }
  const text = await request.text();
  if (text.length > MAX_BODY_BYTES) {
    return NextResponse.json({ error: "Request is too large." }, { status: 413 });
  }
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return NextResponse.json({ error: "Request is not valid JSON." }, { status: 400 });
  }
  const parsed = feedbackInput.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: "Request details are invalid." }, { status: 400 });
  }
  // No accounts: the forwarded address is hashed at once and only used to limit abuse.
  const address = request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "unknown";
  try {
    const result = await saveFeedback(db, parsed.data, hashSession(address, secret));
    if (result === "rate_limited") {
      return NextResponse.json({ error: "Too many requests." }, { status: 429 });
    }
    return NextResponse.json({ ok: true }, { status: 201 });
  } catch (error) {
    console.error("Saving feedback failed", error);
    return NextResponse.json({ error: "Feedback could not be saved." }, { status: 502 });
  }
}
