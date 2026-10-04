"use client";

import { useState } from "react";
import { SearchForm } from "@/components/SearchForm";
import { SchemeCard } from "@/components/SchemeCard";
import { SchemeModal } from "@/components/SchemeModal";
import { CompareView } from "@/components/CompareView";
import { SearchResponse, SchemeResult } from "@/lib/types";
import { Network, Database, Cpu } from "lucide-react";

export default function Home() {
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedScheme, setSelectedScheme] = useState<SchemeResult | null>(null);

  const handleSearch = async (params: any) => {
    setIsLoading(true);
    setError(null);
    setResponse(null);
    
    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${baseUrl}/api/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });

      if (!res.ok) {
        throw new Error("We couldn't connect to the search service.");
      }

      const data = await res.json();
      setResponse(data);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="pb-24">
      <div className="mb-12 text-center md:text-left mt-8">
        <span className="text-xs font-bold tracking-widest uppercase text-blue-600 bg-blue-50 px-3 py-1 rounded-full">
          Government Scheme Discovery
        </span>
        <h1 className="mt-6 text-4xl md:text-5xl font-extrabold text-slate-900 tracking-tight">
          Find the support <br className="hidden md:block" /> you're looking for.
        </h1>
        <p className="mt-4 text-lg text-slate-600 max-w-2xl">
          Search across government welfare schemes using natural language. 
          JanSahay understands queries in English, Hindi, or Hinglish.
        </p>
      </div>

      <SearchForm onSearch={handleSearch} isLoading={isLoading} />

      {error && (
        <div className="p-4 mb-8 bg-red-50 text-red-800 rounded-lg border border-red-100 flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
          {error}
        </div>
      )}

      {isLoading && (
        <div className="py-12 flex flex-col items-center justify-center text-slate-500 gap-4">
          <div className="w-8 h-8 border-4 border-slate-200 border-t-slate-800 rounded-full animate-spin" />
          <p className="font-medium animate-pulse">Finding relevant schemes...</p>
        </div>
      )}

      {!isLoading && response && (
        <div className="mt-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <div className="mb-6 flex items-center justify-between">
            <h2 className="text-2xl font-bold text-slate-800">
              Results for <span className="text-blue-600 font-medium">"{response.query}"</span>
            </h2>
            {response.status === "low_relevance_warning" && (
              <span className="text-xs bg-amber-50 text-amber-700 border border-amber-200 px-3 py-1 rounded-md">
                Low confidence match
              </span>
            )}
          </div>

          {response.status === "no_results" ? (
            <div className="bg-white p-12 text-center rounded-2xl border border-dashed border-slate-300">
              <h3 className="text-xl font-bold text-slate-800 mb-2">We couldn't find a strong match.</h3>
              <p className="text-slate-500 mb-6 max-w-md mx-auto">
                The retrieval model didn't find any schemes closely matching your query.
              </p>
              <ul className="text-sm text-slate-600 text-left max-w-sm mx-auto space-y-2 list-disc pl-4">
                <li>Try describing your need differently</li>
                <li>Try writing in pure English, Hindi, or Hinglish</li>
                <li>Use more specific details (e.g. "student", "farmer", "business")</li>
              </ul>
            </div>
          ) : response.model === "both" ? (
            <CompareView 
              semanticResults={response.semantic_results || []}
              tfidfResults={response.tfidf_results || []}
              onSchemeClick={setSelectedScheme}
            />
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {response.results.map((scheme) => (
                <SchemeCard 
                  key={scheme.scheme_id} 
                  scheme={scheme} 
                  onClick={setSelectedScheme} 
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* How it Works Section */}
      <div id="how-it-works" className="mt-32 pt-16 border-t border-slate-200">
        <h2 className="text-2xl font-bold text-slate-900 mb-8 text-center">How JanSahay Works</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
          <div className="bg-white p-6 rounded-xl border border-slate-100 shadow-sm text-center">
            <div className="w-12 h-12 bg-blue-50 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <Network className="w-6 h-6" />
            </div>
            <h3 className="font-bold text-slate-800 mb-2">Multilingual Embeddings</h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Using <code className="text-xs bg-slate-100 px-1 py-0.5 rounded">intfloat/multilingual-e5-base</code>, queries and documents are mapped into a shared 768-dimensional semantic space, allowing cross-lingual matching without translation.
            </p>
          </div>
          
          <div className="bg-white p-6 rounded-xl border border-slate-100 shadow-sm text-center">
            <div className="w-12 h-12 bg-emerald-50 text-emerald-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <Database className="w-6 h-6" />
            </div>
            <h3 className="font-bold text-slate-800 mb-2">Lexical Baseline</h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              A classical TF-IDF matrix acts as a strict lexical baseline. It performs well on English keyword searches but fails on Hindi/Hinglish due to vocabulary mismatch.
            </p>
          </div>

          <div className="bg-white p-6 rounded-xl border border-slate-100 shadow-sm text-center">
            <div className="w-12 h-12 bg-purple-50 text-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <Cpu className="w-6 h-6" />
            </div>
            <h3 className="font-bold text-slate-800 mb-2">Cosine Similarity</h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Schemes are ranked using pure Cosine Similarity between vectors. Scores represent geometric proximity in the vector space, not a probability of human eligibility.
            </p>
          </div>
        </div>
      </div>

      <SchemeModal scheme={selectedScheme} onClose={() => setSelectedScheme(null)} />
    </div>
  );
}
