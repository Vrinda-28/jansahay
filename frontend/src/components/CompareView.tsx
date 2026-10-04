import { SchemeResult } from "@/lib/types";
import { SchemeCard } from "./SchemeCard";

interface CompareViewProps {
  semanticResults: SchemeResult[];
  tfidfResults: SchemeResult[];
  onSchemeClick: (scheme: SchemeResult) => void;
}

export function CompareView({ semanticResults, tfidfResults, onSchemeClick }: CompareViewProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
      {/* Semantic Column */}
      <div className="space-y-4">
        <div className="sticky top-16 z-10 bg-[#fafaf9] py-2 border-b border-slate-200 mb-4">
          <h2 className="text-lg font-bold text-slate-800 flex items-center justify-between">
            Semantic Retrieval
            <span className="text-xs font-normal text-slate-500 bg-slate-100 px-2 py-1 rounded">
              Embeddings
            </span>
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Matches based on sentence meaning and cross-lingual intent.
          </p>
        </div>
        
        {semanticResults.length === 0 ? (
          <div className="p-8 text-center text-slate-500 bg-white rounded-xl border border-dashed border-slate-300">
            No semantic matches found.
          </div>
        ) : (
          <div className="grid gap-4">
            {semanticResults.map((s) => (
              <SchemeCard key={`sem-${s.scheme_id}`} scheme={s} onClick={onSchemeClick} />
            ))}
          </div>
        )}
      </div>

      {/* TF-IDF Column */}
      <div className="space-y-4">
        <div className="sticky top-16 z-10 bg-[#fafaf9] py-2 border-b border-slate-200 mb-4">
          <h2 className="text-lg font-bold text-slate-800 flex items-center justify-between">
            Lexical Retrieval
            <span className="text-xs font-normal text-slate-500 bg-slate-100 px-2 py-1 rounded">
              TF-IDF
            </span>
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Matches based on exact keyword overlap (biased towards English).
          </p>
        </div>
        
        {tfidfResults.length === 0 ? (
          <div className="p-8 text-center text-slate-500 bg-white rounded-xl border border-dashed border-slate-300">
            No lexical overlap found. Try a different query.
          </div>
        ) : (
          <div className="grid gap-4">
            {tfidfResults.map((s) => (
              <SchemeCard key={`tfidf-${s.scheme_id}`} scheme={s} onClick={onSchemeClick} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
