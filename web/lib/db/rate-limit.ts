import { sql } from "drizzle-orm";
import type { Database } from "./client";
import { rateLimits } from "./schema";

/** Fixed-window limit stored in Postgres. Returns true when this request is allowed. */
export async function allowRequest(
  db: Database,
  key: string,
  limit: number,
  windowSeconds: number,
  now: Date = new Date(),
): Promise<boolean> {
  const windowStart = new Date(now.getTime() - windowSeconds * 1000);
  const [row] = await db
    .insert(rateLimits)
    .values({ key, count: 1, windowStart: now })
    .onConflictDoUpdate({
      target: rateLimits.key,
      set: {
        count: sql`case when ${rateLimits.windowStart} < ${windowStart.toISOString()}::timestamptz then 1 else ${rateLimits.count} + 1 end`,
        windowStart: sql`case when ${rateLimits.windowStart} < ${windowStart.toISOString()}::timestamptz then ${now.toISOString()}::timestamptz else ${rateLimits.windowStart} end`,
      },
    })
    .returning({ count: rateLimits.count });
  return row.count <= limit;
}
