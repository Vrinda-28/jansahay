import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "JanSahay — Government Scheme Discovery",
  description: "A multilingual government welfare scheme discovery platform using NLP.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${inter.variable} font-sans antialiased bg-[#fafaf9] text-[#1c1917] min-h-screen flex flex-col`}
      >
        <header className="sticky top-0 z-50 bg-[#fafaf9]/80 backdrop-blur-md border-b border-slate-200">
          <div className="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between">
            <div className="font-semibold tracking-tight text-lg text-slate-800">
              JANSAHAY
            </div>
            <nav className="flex items-center gap-6 text-sm font-medium text-slate-600">
              <a href="#" className="hover:text-slate-900 transition-colors">Search</a>
              <a href="#how-it-works" className="hover:text-slate-900 transition-colors">How it works</a>
            </nav>
          </div>
        </header>
        
        <main className="flex-1 max-w-5xl w-full mx-auto p-6">
          {children}
        </main>
        
        <footer className="border-t border-slate-200 py-8 mt-12 text-center text-sm text-slate-500">
          <p>JanSahay NLP Project • Government Scheme Discovery</p>
        </footer>
      </body>
    </html>
  );
}
