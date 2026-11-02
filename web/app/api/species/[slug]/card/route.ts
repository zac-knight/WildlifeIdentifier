import { NextResponse } from "next/server";
import { getDatabase } from "@/lib/db/client";
import { geminiWriter, getSpeciesCard } from "@/lib/species-card";
import { speciesBySlug } from "@/lib/species";

export async function GET(_request: Request, context: { params: Promise<{ slug: string }> }) {
  const { slug } = await context.params;
  const species = speciesBySlug(slug);
  if (!species) return NextResponse.json({ error: "Unknown species." }, { status: 404 });
  // Writing cards is limited to where they can be stored. Otherwise every page view would spend the
  // free LLM quota regenerating the same text.
  const db = getDatabase();
  const writer = db && process.env.GOOGLE_GENERATIVE_AI_API_KEY ? geminiWriter : null;
  const card = await getSpeciesCard(species, { db, writer });
  return NextResponse.json(card, {
    headers: { "cache-control": "public, s-maxage=86400, stale-while-revalidate=604800" },
  });
}
