import { createHash } from "node:crypto";
import { and, desc, eq } from "drizzle-orm";
import { z } from "zod";
import type { Database } from "./db/client";
import { allowRequest } from "./db/rate-limit";
import { feedback } from "./db/schema";
import { allSpecies } from "./species";

const known = new Set(allSpecies.map((species) => species.name));
const speciesName = z.string().refine((name) => known.has(name), "Unknown species");

export const feedbackInput = z.object({
  modelVersion: z.string().min(1).max(64),
  predictedSpecies: speciesName,
  correctSpecies: speciesName,
  confirmed: z.boolean(),
  confidence: z.number().min(0).max(1),
  entropy: z.number().min(0).max(20),
  latitude: z.number().min(-90).max(90).nullable(),
  longitude: z.number().min(-180).max(180).nullable(),
  observedOn: z.iso.date().nullable(),
});
export type FeedbackInput = z.infer<typeof feedbackInput>;

/** A one-way hash, so abuse can be limited without storing anything that identifies a person. */
export function hashSession(identifier: string, secret: string): string {
  return createHash("sha256").update(`${secret}:${identifier}`).digest("hex").slice(0, 32);
}

export const FEEDBACK_LIMIT_PER_HOUR = 30;

export type SaveResult = "saved" | "rate_limited";

export async function saveFeedback(
  db: Database,
  input: FeedbackInput,
  sessionHash: string,
): Promise<SaveResult> {
  const allowed = await allowRequest(db, `feedback:${sessionHash}`, FEEDBACK_LIMIT_PER_HOUR, 3600);
  if (!allowed) return "rate_limited";
  await db.insert(feedback).values({ ...input, sessionHash });
  return "saved";
}

/**
 * The review queue puts the most uncertain predictions first. These are where a human label teaches
 * the model most, while confident predictions it already gets right add little.
 */
export async function reviewQueue(db: Database, limit = 50) {
  return db
    .select({
      id: feedback.id,
      predictedSpecies: feedback.predictedSpecies,
      correctSpecies: feedback.correctSpecies,
      confirmed: feedback.confirmed,
      confidence: feedback.confidence,
      entropy: feedback.entropy,
      latitude: feedback.latitude,
      longitude: feedback.longitude,
      observedOn: feedback.observedOn,
      createdAt: feedback.createdAt,
    })
    .from(feedback)
    .where(eq(feedback.reviewStatus, "pending"))
    .orderBy(desc(feedback.entropy))
    .limit(limit);
}

export async function recordReview(
  db: Database,
  id: string,
  decision: { approved: boolean; species?: string },
): Promise<boolean> {
  const rows = await db
    .update(feedback)
    .set({
      reviewStatus: decision.approved ? "approved" : "rejected",
      reviewedSpecies: decision.approved ? (decision.species ?? null) : null,
    })
    .where(and(eq(feedback.id, id), eq(feedback.reviewStatus, "pending")))
    .returning({ id: feedback.id });
  return rows.length === 1;
}
