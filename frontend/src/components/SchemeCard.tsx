import { SchemeResult } from "@/lib/types";
import { Info, ExternalLink } from "lucide-react";
import { cn } from "@/lib/utils";

interface SchemeCardProps {
  scheme: SchemeResult;
  onClick: (scheme: SchemeResult) => void;
}

export function SchemeCard({ scheme, onClick }: SchemeCardProps) {
  return (
    <div 
      onClick={() => onClick(scheme)}
      className="group bg-white rounded-xl p-5 border border-slate-200 hover:border-slate-300 shadow-sm hover:shadow-md transition-all cursor-pointer flex flex-col h-full relative overflow-hidden"
    >
      <div className="flex justify-between items-start gap-4 mb-3">
        <h3 className="font-semibold text-lg text-slate-900 group-hover:text-blue-700 transition-colors line-clamp-2">
          {scheme.scheme_name}
        </h3>
        <div className="flex-shrink-0 flex flex-col items-end gap-1">
          <span className="inline-flex items-center justify-center px-2.5 py-1 text-xs font-semibold rounded-md bg-slate-100 text-slate-700">
            Rank {scheme.rank}
          </span>
        </div>
      </div>
      
      <p className="text-sm text-slate-600 line-clamp-3 mb-4 flex-grow">
        {scheme.description}
      </p>
      
      <div className="flex items-center gap-2 mb-4">
        <span className="px-2 py-1 text-[10px] font-medium tracking-wider uppercase rounded bg-slate-50 border border-slate-100 text-slate-500">
          {scheme.level}
        </span>
        <span className="px-2 py-1 text-[10px] font-medium tracking-wider uppercase rounded bg-slate-50 border border-slate-100 text-slate-500">
          {scheme.intent}
        </span>
      </div>

      <div className="mt-auto pt-4 border-t border-slate-100 flex justify-between items-center">
        <div className="flex items-center gap-1.5 group/tooltip relative">
          <span className="text-sm font-medium text-slate-700">
            Score: {scheme.similarity_score.toFixed(3)}
          </span>
          <Info className="w-3.5 h-3.5 text-slate-400 cursor-help" />
          <div className="absolute bottom-full left-0 mb-2 w-48 p-2 bg-slate-800 text-white text-xs rounded shadow-lg opacity-0 invisible group-hover/tooltip:opacity-100 group-hover/tooltip:visible transition-all z-10 pointer-events-none">
            {scheme.score_type}. This indicates text similarity, not a probability of eligibility.
          </div>
        </div>
        
        {scheme.low_relevance && (
          <span className="text-xs text-amber-600 bg-amber-50 px-2 py-0.5 rounded border border-amber-100">
            Low Match
          </span>
        )}
      </div>
    </div>
  );
}
