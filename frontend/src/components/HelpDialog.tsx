import { useEffect, useRef } from "react";
import { X } from "lucide-react";

import "./HelpDialog.css";

interface HelpDialogProps {
	onClose: () => void;
}

function HelpDialog({ onClose }: HelpDialogProps) {
	const dialogRef = useRef<HTMLElement>(null);
	const onCloseRef = useRef(onClose);
	const previouslyFocusedElementRef = useRef<HTMLElement | null>(null);
	onCloseRef.current = onClose;

	useEffect(() => {
		if (!previouslyFocusedElementRef.current) {
			previouslyFocusedElementRef.current = document.activeElement as HTMLElement | null;
		}

		const dialog = dialogRef.current;
		const focusableElements = dialog?.querySelectorAll<HTMLElement>(
			'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
		);
		const firstFocusableElement = focusableElements?.[0];
		const lastFocusableElement = focusableElements?.[focusableElements.length - 1];

		firstFocusableElement?.focus();

		function handleKeyDown(event: KeyboardEvent) {
			if (event.key === "Escape") {
				onCloseRef.current();
				return;
			}

			if (event.key !== "Tab" || !firstFocusableElement || !lastFocusableElement) {
				return;
			}

			if (event.shiftKey && document.activeElement === firstFocusableElement) {
				event.preventDefault();
				lastFocusableElement.focus();
			} else if (!event.shiftKey && document.activeElement === lastFocusableElement) {
				event.preventDefault();
				firstFocusableElement.focus();
			}
		}

		window.addEventListener("keydown", handleKeyDown);

		return () => {
			window.removeEventListener("keydown", handleKeyDown);
			window.setTimeout(() => {
				if (dialog && !document.body.contains(dialog)) {
					previouslyFocusedElementRef.current?.focus();
				}
			}, 0);
		};
	}, []);

	return (
		<div className="help-backdrop" onMouseDown={onClose}>
			<section
				ref={dialogRef}
				className="help-dialog"
				role="dialog"
				aria-modal="true"
				aria-labelledby="help-title"
				aria-describedby="help-description"
				onMouseDown={(event) => event.stopPropagation()}
			>
				<header>
					<h2 id="help-title">About this project</h2>
					<button
						type="button"
						aria-label="Close help"
						onClick={onClose}
					>
						<X size={18} aria-hidden="true" />
					</button>
				</header>

				<div className="help-introduction" id="help-description">
					<p>
						<strong>Knowledge Graph Swarm</strong> searches documents
						using semantic similarity and relationships between
						neighbouring document agents.
					</p>
					<p>
						The current demonstration uses documents from the HotpotQA
						dataset and there are queries available under the search bar
						that relate to the current dataset.
					</p>
					<p>
						Search results are highlighted in the graph alongside their
						neighbouring documents. Clear the search field to return to
						the default graph view.
					</p>
				</div>

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
