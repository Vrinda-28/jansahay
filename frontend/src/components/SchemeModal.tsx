import { SchemeResult } from "@/lib/types";
import { X } from "lucide-react";
import { useEffect } from "react";

interface SchemeModalProps {
  scheme: SchemeResult | null;
  onClose: () => void;
}

export function SchemeModal({ scheme, onClose }: SchemeModalProps) {
  // Prevent body scroll when modal is open
  useEffect(() => {
    if (scheme) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }
    return () => { document.body.style.overflow = "unset"; };
  }, [scheme]);

  if (!scheme) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6">
      <div 
        className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm transition-opacity" 
        onClick={onClose}
      />
      
      <div className="relative w-full max-w-3xl max-h-[90vh] bg-white rounded-2xl shadow-xl flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        <div className="flex items-center justify-between p-6 border-b border-slate-100 bg-slate-50/50">
          <div>
            <div className="flex gap-2 mb-2">
              <span className="px-2 py-0.5 text-[10px] font-bold tracking-wider uppercase rounded bg-slate-200 text-slate-700">
                {scheme.level}
              </span>
              <span className="px-2 py-0.5 text-[10px] font-bold tracking-wider uppercase rounded bg-blue-100 text-blue-800">
                {scheme.intent}
              </span>
            </div>
            <h2 className="text-xl font-bold text-slate-900 pr-8">
              {scheme.scheme_name}
            </h2>
          </div>
          <button 
            onClick={onClose}
            className="absolute top-6 right-6 p-2 rounded-full hover:bg-slate-200 text-slate-500 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        
        <div className="p-6 overflow-y-auto space-y-8">
          <section>
            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-3">Description</h3>
            <p className="text-slate-700 leading-relaxed whitespace-pre-line">
              {scheme.description}
            </p>
          </section>
          
          <section>
            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-3">Benefits</h3>
            <div className="bg-green-50/50 border border-green-100 rounded-xl p-4 text-slate-700 leading-relaxed whitespace-pre-line">
              {scheme.benefits}
            </div>
          </section>
          
          <section>
            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-3">Eligibility</h3>
            <div className="bg-blue-50/50 border border-blue-100 rounded-xl p-4 text-slate-700 leading-relaxed whitespace-pre-line">
              {scheme.eligibility}
            </div>
          </section>

          {scheme.explanation && (
            <section className="pt-4 border-t border-slate-100">
              <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Retrieval Explanation</h3>
              <p className="text-sm text-slate-500 italic">
                {scheme.explanation}
              </p>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
