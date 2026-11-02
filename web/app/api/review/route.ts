import { timingSafeEqual } from "node:crypto";
import { NextResponse } from "next/server";
import { z } from "zod";
import { getDatabase } from "@/lib/db/client";
import { recordReview, reviewQueue } from "@/lib/feedback";
import { speciesByName } from "@/lib/species";

function authorised(request: Request): boolean {
  const expected = process.env.REVIEW_TOKEN;
  const given = request.headers.get("authorization")?.replace(/^Bearer /, "");
  if (!expected || !given) return false;
  const a = Buffer.from(given);
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}

const decision = z.object({
  id: z.uuid(),
  approved: z.boolean(),
  species: z
    .string()
    .refine((name) => Boolean(speciesByName(name)))
    .optional(),
});

/** The review queue is for the project owner only, so it needs a private token. */
export async function GET(request: Request) {
  if (!authorised(request)) return NextResponse.json({ error: "Not authorised." }, { status: 401 });
  const db = getDatabase();
  if (!db) return NextResponse.json({ error: "Feedback is not enabled." }, { status: 503 });
  return NextResponse.json({ items: await reviewQueue(db) });
}

export async function POST(request: Request) {
  if (!authorised(request)) return NextResponse.json({ error: "Not authorised." }, { status: 401 });
  const db = getDatabase();
  if (!db) return NextResponse.json({ error: "Feedback is not enabled." }, { status: 503 });
  const parsed = decision.safeParse(await request.json().catch(() => null));
  if (!parsed.success) {
    return NextResponse.json({ error: "Request details are invalid." }, { status: 400 });
  }
  const updated = await recordReview(db, parsed.data.id, parsed.data);
  return updated
    ? NextResponse.json({ ok: true })
    : NextResponse.json({ error: "No pending item has that ID." }, { status: 404 });
}
