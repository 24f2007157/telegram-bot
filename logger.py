import json
import os
import tempfile
import uuid
from datetime import datetime

# In serverless environments (Vercel / AWS Lambda), the app root (/var/task) is READ-ONLY.
# The only writable directory is /tmp.
if os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
    LOG_DIR = os.path.join(tempfile.gettempdir(), "logs")
else:
    LOG_DIR = os.path.join(os.path.dirname(__file__), "static", "logs")

os.makedirs(LOG_DIR, exist_ok=True)


class JSONLLogger:
    """
    JSONL Run Logger that records agent execution steps and generates
    a public wget-able URL for grading compliance.
    """

    def __init__(self, log_base_url: str = None):
        self.run_id = (
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        )
        self.filename = f"{self.run_id}.jsonl"
        self.filepath = os.path.join(LOG_DIR, self.filename)
        self.log_base_url = log_base_url or os.getenv(
            "LOG_BASE_URL", "http://localhost:8080/logs"
        )

    def log(self, event_type: str, details: dict):
        """Appends a single JSON event line to the JSONL log file."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "details": details,
        }
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def get_log_url(self) -> str:
        """Returns the public wget-able URL for this run log."""
        base = self.log_base_url.rstrip("/")
        return f"{base}/{self.filename}"
