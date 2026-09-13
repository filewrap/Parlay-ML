"""ops package — what the agent needs to see, test, and undo.

  joblog.py  every scheduler job writes (job, started, secs, ok, note)
  status.py  one JSON blob: daemon, jobs, prod/staging, pools, disk
"""

from .joblog import log_job, recent_jobs
from .status import status

__all__ = ["log_job", "recent_jobs", "status"]
