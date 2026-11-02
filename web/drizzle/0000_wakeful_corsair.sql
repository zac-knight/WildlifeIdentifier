CREATE TABLE "feedback" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"session_hash" text NOT NULL,
	"model_version" text NOT NULL,
	"predicted_species" text NOT NULL,
	"correct_species" text NOT NULL,
	"confirmed" boolean NOT NULL,
	"confidence" double precision NOT NULL,
	"entropy" double precision NOT NULL,
	"latitude" double precision,
	"longitude" double precision,
	"observed_on" text,
	"review_status" text DEFAULT 'pending' NOT NULL,
	"reviewed_species" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "rate_limits" (
	"key" text PRIMARY KEY NOT NULL,
	"count" integer NOT NULL,
	"window_start" timestamp with time zone NOT NULL
);
--> statement-breakpoint
CREATE TABLE "species_cards" (
	"species" text PRIMARY KEY NOT NULL,
	"card" jsonb NOT NULL,
	"source_revision" text,
	"generated_by" text NOT NULL,
	"generated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE INDEX "feedback_queue_idx" ON "feedback" USING btree ("review_status","entropy");--> statement-breakpoint
CREATE INDEX "feedback_session_idx" ON "feedback" USING btree ("session_hash","created_at");