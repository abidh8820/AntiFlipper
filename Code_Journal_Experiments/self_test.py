from __future__ import annotations

import tempfile
from pathlib import Path

from antiflipper_exp.attacks import PartiallyPoisonedDataset
from antiflipper_exp.manifest import build


def main() -> None:
    jobs = build("core")
    assert len(jobs) == 258, len(jobs)
    assert len({job.job_id for job in jobs}) == len(jobs)
    assert len(build("full")) == 330
    for job in jobs: job.validate()
    print("Manifest validation passed: core=258 jobs, full=330 jobs, all IDs unique.")


if __name__ == "__main__": main()
