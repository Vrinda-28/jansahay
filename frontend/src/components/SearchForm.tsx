"use client";

import { useState } from "react";
import { Search, Loader2, Info } from "lucide-react";
import { cn } from "@/lib/utils";

interface SearchFormProps {
  onSearch: (params: any) => void;
  isLoading: boolean;
}

export function SearchForm({ onSearch, isLoading }: SearchFormProps) {
  const [query, setQuery] = useState("");
  const [model, setModel] = useState<"semantic" | "tfidf" | "both">("semantic");
  const [topK, setTopK] = useState(5);
  const [level, setLevel] = useState("");
  const [intent, setIntent] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    onSearch({
      query: query.trim(),
      model,
      top_k: topK,
      level: level || null,
      intent: intent || null,
      include_explanation: true,
    });
  };

  return (
    <div className="bg-white p-6 md:p-8 rounded-2xl shadow-sm border border-slate-100 mb-8 transition-all">
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
            <Search className="h-5 w-5 text-slate-400" />
          </div>
          <input
            type="text"
            className="block w-full pl-11 pr-4 py-4 bg-slate-50 border-none rounded-xl text-slate-900 placeholder-slate-400 focus:ring-2 focus:ring-slate-300 transition-shadow text-lg"
            placeholder="Describe what you need..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !query.trim()}
            className="absolute inset-y-2 right-2 px-6 bg-slate-900 hover:bg-slate-800 text-white font-medium rounded-lg disabled:opacity-50 transition-colors flex items-center justify-center"
          >
            {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : "Search"}
          </button>
        </div>

        <div className="flex flex-col md:flex-row gap-6 md:gap-12 pt-2 border-t border-slate-100">
          <div className="space-y-3">
            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Retrieval Method
            </label>
            <div className="flex flex-wrap gap-3">
              {(["semantic", "tfidf", "both"] as const).map((m) => (
                <label
                  key={m}
                  className={cn(
                    "flex items-center gap-2 px-4 py-2 rounded-lg cursor-pointer text-sm font-medium border transition-colors",
                    model === m
                      ? "bg-slate-900 text-white border-slate-900"
                      : "bg-white text-slate-700 border-slate-200 hover:bg-slate-50"
                  )}
                >
                  <input
                    type="radio"
                    name="model"
                    value={m}
                    checked={model === m}
                    onChange={() => setModel(m)}
                    className="hidden"
                  />
                  {m === "semantic" && "Semantic Search"}
                  {m === "tfidf" && "TF-IDF"}
                  {m === "both" && "Compare Both"}
                </label>
              ))}
            </div>
          </div>

          <div className="flex gap-6 flex-wrap">
            <div className="space-y-3">
              <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Results
              </label>
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="block w-full rounded-lg border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 focus:border-slate-300 focus:ring-0 shadow-sm"
              >
                <option value={3}>Top 3</option>
                <option value={5}>Top 5</option>
                <option value={10}>Top 10</option>
              </select>
            </div>

            <div className="space-y-3">
              <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-1">
                Level <Info className="w-3 h-3" />
              </label>
              <select
                value={level}
                onChange={(e) => setLevel(e.target.value)}
                className="block w-full rounded-lg border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 focus:border-slate-300 focus:ring-0 shadow-sm"
              >
                <option value="">All Levels</option>
                <option value="Central">Central</option>
                <option value="State">State</option>
              </select>
            </div>
          </div>
        </div>
      </form>
      
      <div className="mt-6 pt-4 border-t border-slate-100">
        <p className="text-xs text-slate-500 flex gap-4">
          <span className="font-medium text-slate-700">Try:</span>
          <button onClick={() => setQuery("engineering student scholarship")} className="hover:text-slate-900 underline underline-offset-2 decoration-slate-300">English</button>
          <button onClick={() => setQuery("व्यवसाय शुरू करने के लिए सरकारी सब्सिडी")} className="hover:text-slate-900 underline underline-offset-2 decoration-slate-300">हिन्दी</button>
          <button onClick={() => setQuery("kisan machine subsidy")} className="hover:text-slate-900 underline underline-offset-2 decoration-slate-300">Hinglish</button>
        </p>
      </div>
    </div>
  );
}
