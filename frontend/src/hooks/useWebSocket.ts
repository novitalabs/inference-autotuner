import { useEffect, useRef, useState, useCallback } from "react";

/**
 * WebSocket connection state
 */
export enum WebSocketState {
	CONNECTING = "CONNECTING",
	OPEN = "OPEN",
	CLOSING = "CLOSING",
	CLOSED = "CLOSED",
}

/**
 * WebSocket event received from server
 */
export interface WebSocketEvent {
	type: string;
	task_id?: number;
	experiment_id?: number;
	message?: string;
	data?: Record<string, any>;
	timestamp?: number;
}

/**
 * Configuration options for useWebSocket hook
 */
export interface UseWebSocketOptions {
	/** Whether to automatically connect on mount (default: true) */
	autoConnect?: boolean;
	/** Whether to automatically reconnect on disconnect (default: true) */
	autoReconnect?: boolean;
	/** Base reconnection delay in ms (default: 1000) */
	reconnectDelay?: number;
	/** Maximum reconnection delay in ms (default: 30000) */
	maxReconnectDelay?: number;
	/** Maximum number of reconnection attempts (default: Infinity) */
	maxReconnectAttempts?: number;
	/** Callback when connection opens */
	onOpen?: (event: Event) => void;
	/** Callback when connection closes */
	onClose?: (event: CloseEvent) => void;
	/** Callback when error occurs */
	onError?: (event: Event) => void;
	/** Callback when message is received */
	onMessage?: (event: WebSocketEvent) => void;
}

/**
 * Return type of useWebSocket hook
 */
export interface UseWebSocketReturn {
	/** Current connection state */
	state: WebSocketState;
	/** Latest message received */
	lastMessage: WebSocketEvent | null;
	/** All messages received (limited to last 100) */
	messageHistory: WebSocketEvent[];
	/** Send a message through the WebSocket */
	sendMessage: (message: any) => void;
	/** Manually connect to WebSocket */
	connect: () => void;
	/** Manually disconnect from WebSocket */
	disconnect: () => void;
	/** Whether currently connected */
	isConnected: boolean;
	/** Number of reconnection attempts made */
	reconnectAttempts: number;
}

/**
 * React hook for managing WebSocket connections with automatic reconnection
 *
 * @param url - WebSocket URL (e.g., "ws://localhost:8000/api/ws/tasks/1")
 * @param options - Configuration options
 * @returns WebSocket state and control functions
 *
 * @example
 * ```tsx
 * const { state, lastMessage, messageHistory, isConnected } = useWebSocket(
 *   `ws://localhost:8000/api/ws/tasks/${taskId}`,
 *   {
 *     onMessage: (event) => {
 *       console.log('Received:', event);
 *     },
 *     onOpen: () => console.log('Connected'),
 *     onClose: () => console.log('Disconnected'),
 *   }
 * );
 * ```
 */
export function useWebSocket(
	url: string | null,
	options: UseWebSocketOptions = {}
): UseWebSocketReturn {
	const {
		autoConnect = true,
		autoReconnect = true,
		reconnectDelay = 1000,
		maxReconnectDelay = 30000,
		maxReconnectAttempts = Infinity,
		onOpen,
		onClose,
		onError,
		onMessage,
	} = options;

	const [state, setState] = useState<WebSocketState>(WebSocketState.CLOSED);
	const [lastMessage, setLastMessage] = useState<WebSocketEvent | null>(null);
	const [messageHistory, setMessageHistory] = useState<WebSocketEvent[]>([]);
	const [reconnectAttempts, setReconnectAttempts] = useState(0);

	const wsRef = useRef<WebSocket | null>(null);
	const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
	const shouldReconnectRef = useRef(autoReconnect);
	const currentUrlRef = useRef(url);

	// Update URL ref when it changes
	useEffect(() => {
		currentUrlRef.current = url;
	}, [url]);

	/**
	 * Clear reconnection timeout
	 */
	const clearReconnectTimeout = useCallback(() => {
		if (reconnectTimeoutRef.current) {
			clearTimeout(reconnectTimeoutRef.current);
			reconnectTimeoutRef.current = null;
		}
	}, []);

	/**
	 * Calculate exponential backoff delay for reconnection
	 */
	const getReconnectDelay = useCallback(
		(attempt: number): number => {
			// Exponential backoff: delay * 2^attempt, capped at maxReconnectDelay
			const delay = Math.min(reconnectDelay * Math.pow(2, attempt), maxReconnectDelay);
			// Add jitter (±25%) to prevent thundering herd
			const jitter = delay * 0.25 * (Math.random() * 2 - 1);
			return Math.floor(delay + jitter);
		},
		[reconnectDelay, maxReconnectDelay]
	);

	/**
	 * Connect to WebSocket
	 */
	const connect = useCallback(() => {
		// Don't connect if no URL or already connecting/connected
		if (!currentUrlRef.current || wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) {
			return;
		}

		// Clear any pending reconnection
		clearReconnectTimeout();

		try {
			setState(WebSocketState.CONNECTING);

			// Create WebSocket connection
			const ws = new WebSocket(currentUrlRef.current);
			wsRef.current = ws;

			// Handle connection open
			ws.onopen = (event) => {
				setState(WebSocketState.OPEN);
				setReconnectAttempts(0); // Reset reconnection counter on successful connection
				shouldReconnectRef.current = autoReconnect; // Re-enable auto-reconnect
				onOpen?.(event);
			};

			// Handle incoming messages
			ws.onmessage = (event) => {
				try {
					const message: WebSocketEvent = JSON.parse(event.data);
					setLastMessage(message);

					// Add to history (keep last 100 messages)
					setMessageHistory((prev) => {
						const newHistory = [...prev, message];
						return newHistory.slice(-100);
					});

					onMessage?.(message);
				} catch (error) {
					console.error("[useWebSocket] Failed to parse message:", error);
				}
			};

			// Handle connection close
			ws.onclose = (event) => {
				setState(WebSocketState.CLOSED);
				wsRef.current = null;
				onClose?.(event);

				// Attempt reconnection if enabled
				if (shouldReconnectRef.current && reconnectAttempts < maxReconnectAttempts) {
					const delay = getReconnectDelay(reconnectAttempts);
					console.log(`[useWebSocket] Reconnecting in ${delay}ms (attempt ${reconnectAttempts + 1}/${maxReconnectAttempts === Infinity ? "∞" : maxReconnectAttempts})...`);

					reconnectTimeoutRef.current = setTimeout(() => {
						setReconnectAttempts((prev) => prev + 1);
						connect();
					}, delay);
				} else if (reconnectAttempts >= maxReconnectAttempts) {
					console.error("[useWebSocket] Max reconnection attempts reached");
				}
			};

			// Handle errors
			ws.onerror = (event) => {
				console.error("[useWebSocket] WebSocket error:", event);
				onError?.(event);
			};
		} catch (error) {
			console.error("[useWebSocket] Failed to create WebSocket:", error);
			setState(WebSocketState.CLOSED);
		}
	}, [autoReconnect, clearReconnectTimeout, getReconnectDelay, maxReconnectAttempts, onClose, onError, onMessage, onOpen, reconnectAttempts]);

	/**
	 * Disconnect from WebSocket
	 */
	const disconnect = useCallback(() => {
		// Disable auto-reconnect
		shouldReconnectRef.current = false;

		// Clear any pending reconnection
		clearReconnectTimeout();

		// Close WebSocket connection
		if (wsRef.current) {
			setState(WebSocketState.CLOSING);
			wsRef.current.close();
			wsRef.current = null;
		}
	}, [clearReconnectTimeout]);

	/**
	 * Send message through WebSocket
	 */
	const sendMessage = useCallback((message: any) => {
		if (wsRef.current?.readyState === WebSocket.OPEN) {
			wsRef.current.send(typeof message === "string" ? message : JSON.stringify(message));
		} else {
			console.warn("[useWebSocket] Cannot send message: WebSocket not connected");
		}
	}, []);

	// Auto-connect on mount if enabled
	useEffect(() => {
		if (autoConnect && url) {
			connect();
		}

		// Cleanup on unmount
		return () => {
			shouldReconnectRef.current = false;
			clearReconnectTimeout();
			if (wsRef.current) {
				wsRef.current.close();
			}
		};
	}, [url, autoConnect, connect, clearReconnectTimeout]);

	return {
		state,
		lastMessage,
		messageHistory,
		sendMessage,
		connect,
		disconnect,
		isConnected: state === WebSocketState.OPEN,
		reconnectAttempts,
	};
}
