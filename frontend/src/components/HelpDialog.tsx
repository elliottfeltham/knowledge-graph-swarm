import { useEffect } from "react";
import { X } from "lucide-react";

interface HelpDialogProps {
	onClose: () => void;
}

function HelpDialog({ onClose }: HelpDialogProps) {
	useEffect(() => {
		function handleKeyDown(event: KeyboardEvent) {
			if (event.key === "Escape") {
				onClose();
			}
		}

		window.addEventListener("keydown", handleKeyDown);
		return () => window.removeEventListener("keydown", handleKeyDown);
	}, [onClose]);

	return (
		<div className="help-backdrop" onMouseDown={onClose}>
			<section
				className="help-dialog"
				role="dialog"
				aria-modal="true"
				aria-labelledby="help-title"
				onMouseDown={(event) => event.stopPropagation()}
			>
				<header>
					<div>
						<p className="eyebrow">About this project</p>
						<h2 id="help-title">How it works</h2>
					</div>
					<button
						type="button"
						aria-label="Close help"
						onClick={onClose}
					>
						<X size={18} aria-hidden="true" />
					</button>
				</header>

				<p>
					Knowledge Graph Swarm searches documents using semantic
					similarity and relationships between neighbouring document
					agents. The current demonstration uses documents from the
					HotpotQA dataset and there are queries available under the
					search bar that relate to the current dataset.
				</p>

				<dl>
					<div>
						<dt>Standard search</dt>
						<dd>
							Retrieves documents without changing the graph. This
							is the default behavior. This is useful for seeing
							how well the current graph retrieves documents.
						</dd>
					</div>
					<div>
						<dt>Adaptive search</dt>
						<dd>
							Retrieves documents and updates their relationships
							so the graph can adapt to repeated use.
						</dd>
					</div>
					<div>
						<dt>Retrieval score</dt>
						<dd>
							Indicates how strongly a document matched this
							query. Higher scores rank first.
						</dd>
					</div>
					<div>
						<dt>Retrieval time</dt>
						<dd>
							The time taken to encode and retrieve results, in
							milliseconds.
						</dd>
					</div>
				</dl>
			</section>
		</div>
	);
}

export default HelpDialog;
