"""
Development server runner.
"""

import uvicorn

if __name__ == "__main__":
	uvicorn.run(
		"web.app:app",
		host="0.0.0.0",
		port=8000,
		reload=True,  # Enable hot reload for development
		log_level="info",
	)
