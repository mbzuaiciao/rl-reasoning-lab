"""Run the frozen 512-token EXP-016 repair diagnostic or regrade saved patches."""

if __package__:
    from . import exp15_repair_feasibility as pilot
else:
    import exp15_repair_feasibility as pilot

CAP = 512


if __name__ == "__main__":
    raise SystemExit(
        pilot.main(cap=CAP, smoke=False, grade_during_sampling=True, description=__doc__)
    )
