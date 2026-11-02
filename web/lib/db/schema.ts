import {
  boolean,
  doublePrecision,
  index,
  integer,
  jsonb,
  pgTable,
  text,
  timestamp,
  uuid,
} from "drizzle-orm/pg-core";

/** User feedback. No photos or recordings are stored, and places are already coarsened. */
export const feedback = pgTable(
  "feedback",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    sessionHash: text("session_hash").notNull(),
    modelVersion: text("model_version").notNull(),
    predictedSpecies: text("predicted_species").notNull(),
    correctSpecies: text("correct_species").notNull(),
    confirmed: boolean("confirmed").notNull(),
    confidence: doublePrecision("confidence").notNull(),
    entropy: doublePrecision("entropy").notNull(),
    latitude: doublePrecision("latitude"),
    longitude: doublePrecision("longitude"),
    observedOn: text("observed_on"),
    /** Review state for the active learning queue. */
    reviewStatus: text("review_status").notNull().default("pending"),
    reviewedSpecies: text("reviewed_species"),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (table) => [
    index("feedback_queue_idx").on(table.reviewStatus, table.entropy),
    index("feedback_session_idx").on(table.sessionHash, table.createdAt),
  ],
);

/** Generated species profiles, regenerated only when stale. */
export const speciesCards = pgTable("species_cards", {
  species: text("species").primaryKey(),
  card: jsonb("card").notNull(),
  sourceRevision: text("source_revision"),
  generatedBy: text("generated_by").notNull(),
  generatedAt: timestamp("generated_at", { withTimezone: true }).notNull().defaultNow(),
});

/** A simple shared counter for rate limits, because serverless instances keep no shared memory. */
export const rateLimits = pgTable("rate_limits", {
  key: text("key").primaryKey(),
  count: integer("count").notNull(),
  windowStart: timestamp("window_start", { withTimezone: true }).notNull(),
});
