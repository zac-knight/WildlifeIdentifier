export interface Source {
  id: string;
  title: string;
  url: string;
  licence: string;
  text: string;
}

export interface Claim {
  text: string;
  sourceId: string;
}

export interface SpeciesCard {
  species: string;
  /** "llm" text was written from the sources. "facts" shows the sources directly. */
  mode: "llm" | "facts";
  claims: Claim[];
  sources: Source[];
  generatedAt: string;
  sourceRevision: string | null;
}
