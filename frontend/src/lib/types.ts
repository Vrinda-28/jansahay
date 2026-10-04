export interface SchemeResult {
  scheme_id: number;
  scheme_name: string;
  description: string;
  benefits: string;
  eligibility: string;
  level: string;
  intent: string;
  similarity_score: number;
  score_type: string;
  rank: number;
  low_relevance: boolean;
  explanation?: string;
}

export interface SearchResponse {
  query: string;
  model: "tfidf" | "semantic" | "both";
  status: "success" | "no_results" | "low_relevance_warning";
  message?: string;
  results: SchemeResult[];
  tfidf_results?: SchemeResult[];
  semantic_results?: SchemeResult[];
}
