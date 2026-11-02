import { neon } from "@neondatabase/serverless";
import { drizzle } from "drizzle-orm/neon-http";
import type { PgDatabase, PgQueryResultHKT } from "drizzle-orm/pg-core";
import * as schema from "./schema";

export type Database = PgDatabase<PgQueryResultHKT, typeof schema>;

let database: Database | null | undefined;

/** Returns null when no database is configured, so features can degrade instead of crashing. */
export function getDatabase(): Database | null {
  if (database !== undefined) return database;
  const url = process.env.DATABASE_URL;
  database = url ? (drizzle(neon(url), { schema }) as unknown as Database) : null;
  return database;
}

/** Tests inject an in-memory database here. */
export function setDatabaseForTests(value: Database | null | undefined): void {
  database = value;
}
