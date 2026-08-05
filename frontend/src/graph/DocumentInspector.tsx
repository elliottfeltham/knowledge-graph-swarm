import { AlertCircle, ArrowLeft, FileText, RefreshCw } from "lucide-react";

import type { DocumentDetail, SearchResult } from "../api/types";
import "./GraphInspector.css";
import "./DocumentInspector.css";

interface DocumentInspectorProps {
  document: DocumentDetail | null;
  loading: boolean;
  error: string | null;
  searchResult: SearchResult | null;
  searchRank: number | null;
  onBack: () => void;
  onRetry: () => void;
}

function DocumentInspector({
  document,
  loading,
  error,
  searchResult,
  searchRank,
  onBack,
  onRetry,
}: DocumentInspectorProps) {
  return (
    <aside className="graph-inspector document-inspector" aria-label="Document details">
      <header>
        <div>
          <p className="eyebrow">Document agent</p>
          <h3>{document?.title ?? "Document details"}</h3>
        </div>
        <button
          type="button"
          title="Back to graph controls"
          aria-label="Back to graph controls"
          onClick={onBack}
        >
          <ArrowLeft size={17} aria-hidden="true" />
        </button>
      </header>

      {loading && (
        <div className="document-inspector-message" role="status">
          <FileText size={17} aria-hidden="true" />
          <span>Loading document</span>
        </div>
      )}

      {error && (
        <div className="document-inspector-error" role="alert">
          <AlertCircle size={16} aria-hidden="true" />
          <span>{error}</span>
          <button type="button" onClick={onRetry}>
            <RefreshCw size={13} aria-hidden="true" />
            Retry
          </button>
        </div>
      )}

      {document && !loading && (
        <>
          <dl className="document-metadata">
            <div>
              <dt>Document ID</dt>
              <dd>{document.doc_id}</dd>
            </div>
            <div>
              <dt>Connections</dt>
              <dd>{document.degree}</dd>
            </div>
            {searchResult && searchRank && (
              <>
                <div>
                  <dt>Search rank</dt>
                  <dd>#{searchRank}</dd>
                </div>
                <div>
                  <dt>Retrieval score</dt>
                  <dd>{searchResult.score.toFixed(3)}</dd>
                </div>
              </>
            )}
          </dl>

          <div className="document-preview">
            <h4>Text preview</h4>
            <p>{document.text_preview || "No document text available."}</p>
          </div>
        </>
      )}
    </aside>
  );
}

export default DocumentInspector;
