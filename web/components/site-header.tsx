import Link from "next/link";
import { ThemeToggle } from "./theme-toggle";

const LINKS = [
  { href: "/identify", label: "Identify" },
  { href: "/species", label: "Species" },
  { href: "/camera-trap", label: "Camera traps" },
  { href: "/evaluation", label: "Evaluation" },
  { href: "/about", label: "About" },
];

export function SiteHeader() {
  return (
    <header className="border-b border-border">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:rounded focus:bg-card focus:p-2"
      >
        Skip to content
      </a>
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
        <Link href="/" className="font-heading text-xl font-semibold tracking-tight">
          WildLens
        </Link>
        <nav aria-label="Main" className="order-3 w-full sm:order-none sm:w-auto sm:flex-1">
          <ul className="flex flex-wrap gap-x-4 gap-y-1 text-sm">
            {LINKS.map((link) => (
              <li key={link.href}>
                <Link
                  href={link.href}
                  className="text-muted-foreground hover:text-foreground hover:underline"
                >
                  {link.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
        <div className="ml-auto sm:ml-0">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="mt-16 border-t border-border">
      <div className="mx-auto max-w-6xl px-4 py-6 text-sm text-muted-foreground">
        Photos and recordings are by iNaturalist contributors.{" "}
        <Link href="/about" className="underline">
          See credits and licences
        </Link>
        . Climate data from NASA NEX-GDDP-CMIP6. Identifications are model estimates and can be
        wrong.
      </div>
    </footer>
  );
}
