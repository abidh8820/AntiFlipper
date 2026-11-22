from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from antiflipper_exp.manifest import build

CORE_JOBS = 231
FULL_JOBS = 306

# Completed under the v1 manifest; the v2 manifest must keep their job ids so the
# launcher skips them instead of re-running.
COMPLETED_V1 = {
    "a1_headline__mnist__iid__antiflipper__s7__4bc51d584b59",
    "a1_headline__mnist__iid__antiflipper__s19__59b9f4b171d7",
    "a1_headline__mnist__iid__antiflipper__s31__1fe5433c6092",
    "a1_headline__mnist__iid__fedavg__s7__632dc8976de7",
    "a1_headline__mnist__iid__fedavg__s19__7bc548c7e8c7",
    "a1_headline__mnist__iid__fedavg__s31__56d6dbfd6a89",
}


def main() -> None:
    core = build("core"); full = build("full")
    cc_v2 = build("cc_v2")
    assert len(core) == CORE_JOBS, len(core)
    assert len(full) == FULL_JOBS, len(full)
    assert len(cc_v2) == 12, len(cc_v2)
    for jobs in (core, full, build("smoke"), cc_v2):
        assert len({job.job_id for job in jobs}) == len(jobs)
        for job in jobs: job.validate()
        assert max(job.malicious_client_fraction for job in jobs) <= 0.4, "threat model caps attackers at 40%"
    ids = {job.job_id for job in core}
    assert all(replace(job, method="antiflipper_cc").job_id in ids for job in cc_v2)
    assert [job.job_id for job in core if job.method == "antiflipper_cc_v2"] == [job.job_id for job in cc_v2]
    missing = COMPLETED_V1 - ids
    assert not missing, f"completed v1 jobs no longer in manifest: {missing}"
    on_disk = {p.parent.name for p in (Path(__file__).parent / "results" / "core").glob("*/summary.json")}
    print(f"Manifest validation passed: core={len(core)}, full={len(full)}, all IDs unique; "
          f"{len(ids & on_disk)} completed core results on disk will be skipped.")


if __name__ == "__main__": main()
