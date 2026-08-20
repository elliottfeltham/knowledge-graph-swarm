import { useState } from "react";
import type { FormEvent } from "react";
import {
  AlertCircle,
  Info,
  LoaderCircle,
  Search,
  SlidersHorizontal,
  X,
} from "lucide-react";

import {
  adaptiveSearchDocuments,
  searchDocuments,
} from "../api/client";
import type {
  AdaptiveSearchResponse,
  GraphUpdateSummary,
  SearchResponse,
} from "../api/types";
import "./SearchPanel.css";

interface SearchPanelProps {
  backendAvailable: boolean;
  onSearchStateChange: (searching: boolean) => void;
  onSearchComplete: (response: SearchResponse) => void;
  onSearchClear: () => void;
}

const EXAMPLE_QUESTIONS = [
  "Which magazine was started first, Arthur's Magazine or First for Women?",
  "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
  "Were Pavel Urysohn and Leonid Levin known for the same type of work?",
];

function isAdaptiveResponse(
  response: SearchResponse,
): response is AdaptiveSearchResponse {
  return "update_summary" in response;
}

function SearchPanel({
  backendAvailable,
  onSearchStateChange,
  onSearchComplete,
  onSearchClear,
}: SearchPanelProps) {
  const [query, setQuery] = useState("");
  const [adaptive, setAdaptive] = useState(false);
  const [topK, setTopK] = useState(10);
  const [seedK, setSeedK] = useState(3);
  const [graphWeight, setGraphWeight] = useState(0.05);
  const [updateK, setUpdateK] = useState(3);
  const [strengthenAmount, setStrengthenAmount] = useState(0.01);
  const [decayRate, setDecayRate] = useState(0.995);
  const [pruningEnabled, setPruningEnabled] = useState(false);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [lastSearchSettings, setLastSearchSettings] = useState<{
    topK: number;
    seedK: number;
  } | null>(null);
  const [updateSummary, setUpdateSummary] =
    useState<GraphUpdateSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);

  function clearSearch() {
    setQuery("");
    setResponse(null);
    setLastSearchSettings(null);
    setUpdateSummary(null);
    setError(null);
    onSearchClear();
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedQuery = query.trim();

    if (!trimmedQuery || searching || !backendAvailable) {
      return;
    }

    setSearching(true);
    setError(null);
    setUpdateSummary(null);
    onSearchStateChange(true);

    const request = {
      query: trimmedQuery,
      top_k: topK,
      seed_k: seedK,
      graph_weight: graphWeight,
    };

    try {
      const nextResponse = adaptive
        ? await adaptiveSearchDocuments({
            ...request,
            update_k: updateK,
            strengthen_amount: strengthenAmount,
            decay_rate: decayRate,
            enable_pruning: pruningEnabled,
          })
        : await searchDocuments(request);

      setResponse(nextResponse);
      setLastSearchSettings({ topK, seedK });
      onSearchComplete(nextResponse);

      if (isAdaptiveResponse(nextResponse)) {
        setUpdateSummary(nextResponse.update_summary);
      }
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : "The search could not be completed.";

      setError(message);
    } finally {
      setSearching(false);
      onSearchStateChange(false);
    }
  }

  return (
    <div className="search-panel">
      <form className="search-form" onSubmit={handleSubmit}>
        <label className="field-label" htmlFor="search-query">
          Search documents
        </label>

        <div className="search-input-row">
          <div className="search-query-control">
            <input
              id="search-query"
              type="search"
              value={query}
              onChange={(event) => {
                const nextQuery = event.target.value;
                setQuery(nextQuery);

                if (!nextQuery) {
                  clearSearch();
                }
              }}
              placeholder="Enter a question"
              autoComplete="off"
              required
            />
            {query && (
              <button
                type="button"
                title="Clear search"
                aria-label="Clear search"
                onClick={clearSearch}
              >
                <X size={15} aria-hidden="true" />
              </button>
            )}
          </div>
          <button
            className="search-button"
            type="submit"
            disabled={!backendAvailable || searching || !query.trim()}
          >
            {searching ? (
              <LoaderCircle className="spin" size={17} aria-hidden="true" />
            ) : (
              <Search size={17} aria-hidden="true" />
            )}
            <span>{searching ? "Searching" : "Search"}</span>
          </button>
        </div>

        <details className="example-questions">
          <summary>Example HotpotQA queries</summary>
          <div>
            {EXAMPLE_QUESTIONS.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => setQuery(example)}
              >
                {example}
              </button>
            ))}
          </div>
        </details>

        <div className="search-options">
          <label className="toggle-control">
            <input
              type="checkbox"
              checked={adaptive}
              onChange={(event) => setAdaptive(event.target.checked)}
            />
            <span className="toggle-track" aria-hidden="true">
              <span className="toggle-thumb" />
            </span>
            <span>Adaptive search</span>
          </label>

          <details className="retrieval-settings">
            <summary>
              <SlidersHorizontal size={15} aria-hidden="true" />
              Retrieval settings
            </summary>
            <div className="settings-grid">
              <label>
                <span>Results</span>
                <input
                  type="number"
                  min="1"
                  max="100"
                  required
                  value={topK}
                  onChange={(event) => {
                    const nextTopK = Number(event.target.value);
                    setTopK(nextTopK);
                    setUpdateK((current) =>
                      Math.min(current, Math.max(2, nextTopK)),
                    );
                  }}
                />
              </label>
              <label>
                <span>Seeds</span>
                <input
                  type="number"
                  min="1"
                  max="100"
                  required
                  value={seedK}
                  onChange={(event) => setSeedK(Number(event.target.value))}
                />
              </label>
              <label>
                <span>Graph weight</span>
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.05"
                  required
                  value={graphWeight}
                  onChange={(event) =>
                    setGraphWeight(Number(event.target.value))
                  }
                />
              </label>
              {adaptive && (
                <>
                  <label>
                    <span title="Use 3 for selective strengthening or 10 for the original setup">
                      Documents updated
                    </span>
                    <input
                      type="number"
                      min="2"
                      max={Math.max(2, topK)}
                      required
                      value={updateK}
                      onChange={(event) =>
                        setUpdateK(Number(event.target.value))
                      }
                    />
                  </label>
                  <label>
                    <span title="Amount added to relationships between documents retrieved together">
                      Strengthening
                    </span>
                    <input
                      type="number"
                      min="0.001"
                      max="1"
                      step="0.001"
                      required
                      value={strengthenAmount}
                      onChange={(event) =>
                        setStrengthenAmount(Number(event.target.value))
                      }
                    />
                  </label>
                  <label>
                    <span title="Multiplier applied to competing local relationships">
                      Decay rate
                    </span>
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.001"
                      required
                      value={decayRate}
                      onChange={(event) =>
                        setDecayRate(Number(event.target.value))
                      }
                    />
                  </label>
                </>
              )}
            </div>
            {adaptive && (
              <label className="toggle-control pruning-control">
                <input
                  type="checkbox"
                  checked={pruningEnabled}
                  onChange={(event) =>
                    setPruningEnabled(event.target.checked)
                  }
                />
                <span className="toggle-track" aria-hidden="true">
                  <span className="toggle-thumb" />
                </span>
                <span>Enable pruning</span>
              </label>
            )}
          </details>
        </div>
      </form>

      {error && (
        <div className="search-error" role="alert">
          <AlertCircle size={16} aria-hidden="true" />
          <span>{error}</span>
        </div>
      )}

      {updateSummary && (
        <section
          className="update-summary"
          aria-label="Graph update summary"
          role="status"
        >
          <div className="update-heading">
            <span>Graph updated</span>
          </div>
          <dl>
            <div>
              <dt>Updated</dt>
              <dd>
                {updateSummary.updated_documents ??
                  updateSummary.retrieved_documents}
              </dd>
            </div>
            <div>
              <dt>Strengthened</dt>
              <dd>{updateSummary.strengthened_edges}</dd>
            </div>
            <div>
              <dt>Created</dt>
              <dd>{updateSummary.created_edges}</dd>
            </div>
            <div>
              <dt>Decayed</dt>
              <dd>{updateSummary.decayed_edges}</dd>
            </div>
            <div>
              <dt>Pruned</dt>
              <dd>{updateSummary.pruned_edges}</dd>
            </div>
          </dl>
        </section>
      )}

      {response && (
        <section className="search-results" aria-label="Search results">
          <header className="results-header">
            <span>
              {response.results.length}
              {lastSearchSettings &&
              response.results.length < lastSearchSettings.topK
                ? ` of ${lastSearchSettings.topK} requested`
                : " results"}
            </span>
            <span>Retrieval time {response.latency_ms.toFixed(1)} ms</span>
          </header>

          {lastSearchSettings &&
            response.results.length < lastSearchSettings.topK && (
              <div className="candidate-notice" role="status">
                <Info size={14} aria-hidden="true" />
                <span>
                  The {lastSearchSettings.seedK} seeds produced{" "}
                  {response.results.length} unique candidates. Increase Seeds
                  to widen the search.
                </span>
              </div>
            )}

          {response.results.length ? (
            <ol>
              {response.results.map((result, index) => (
                <li key={result.doc_id}>
                  <div className="result-rank">{index + 1}</div>
                  <article>
                    <div className="result-title-row">
                      <h3>{result.title}</h3>
                      <span className="result-score">
                        <small>Retrieval score</small>
                        {result.score.toFixed(3)}
                      </span>
                    </div>
                    <p>{result.text_preview}</p>
                  </article>
                </li>
              ))}
            </ol>
          ) : (
            <p className="empty-results">No documents found.</p>
          )}
        </section>
      )}
    </div>
  );
}

export default SearchPanel;
