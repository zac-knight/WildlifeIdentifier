import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import Home from "@/app/page";

test("landing page names the product", () => {
  render(<Home />);
  expect(screen.getByRole("heading", { name: "WildLens" })).toBeTruthy();
});
