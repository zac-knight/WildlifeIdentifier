import speciesData from "@/data/species.json";

export interface SdmChange {
  area_change_percent: number | null;
  gained_cells: number;
  lost_cells: number;
  robust_share: number;
  novel_share: number;
}

export interface Species {
  name: string;
  slug: string;
  commonName: string | null;
  genus: string | null;
  family: string | null;
  order: string | null;
  observations: number;
  iucn: string | null;
  threatened: boolean;
  photoUrl: string | null;
  photoAttribution: string | null;
  recordings: number;
  sdm: {
    threshold: number;
    auc: number | null;
    boyce: number | null;
    coarsened: boolean;
    changes: Record<string, SdmChange>;
  } | null;
}

export const allSpecies = speciesData as Species[];

const byName = new Map(allSpecies.map((species) => [species.name, species]));
const bySlug = new Map(allSpecies.map((species) => [species.slug, species]));

export function speciesByName(name: string): Species | undefined {
  return byName.get(name);
}

export function speciesBySlug(slug: string): Species | undefined {
  return bySlug.get(slug);
}

export function displayName(species: Pick<Species, "commonName" | "name">): string {
  return species.commonName ?? species.name;
}

const IUCN_LABELS: Record<string, string> = {
  LC: "Least Concern",
  NT: "Near Threatened",
  VU: "Vulnerable",
  EN: "Endangered",
  CR: "Critically Endangered",
  EW: "Extinct in the Wild",
  EX: "Extinct",
  DD: "Data Deficient",
  NE: "Not Evaluated",
};

export function iucnLabel(code: string | null): string {
  return code ? (IUCN_LABELS[code] ?? code) : "Not available";
}
