import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "GoaVaani Voice RAG", description: "A fast, multilingual and grounded voice RAG experience for HH Goa 2026.", other: { "codex-preview": "development" }, icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" } };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
