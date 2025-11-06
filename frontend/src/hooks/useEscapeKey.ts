import { useEffect } from "react";

/**
 * Hook to handle Escape key press
 * @param onEscape - Callback function to execute when Escape is pressed
 * @param enabled - Whether the hook is enabled (default: true)
 */
export function useEscapeKey(onEscape: () => void, enabled: boolean = true) {
	useEffect(() => {
		if (!enabled) return;

		const handleKeyDown = (event: KeyboardEvent) => {
			if (event.key === "Escape" || event.key === "Esc") {
				event.preventDefault();
				onEscape();
			}
		};

		document.addEventListener("keydown", handleKeyDown);

		return () => {
			document.removeEventListener("keydown", handleKeyDown);
		};
	}, [onEscape, enabled]);
}
