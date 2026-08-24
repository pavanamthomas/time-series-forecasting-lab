# Contributing to the time-series lab

Useful work is a leakage demonstration, a walk-forward split that actually withholds the future, or a metric that does not treat in-sample fit as a forecast.

1. Open an issue naming the origin, the horizon, and the information-set violation.
2. Add a failing test before a numerical change.
3. Keep commits to one forecasting claim.
4. Comment leakage risks and evaluation windows, not obvious syntax.

See `docs/failures_and_corrections.md`, `ROADMAP.md`, and `.github/workflows/ci.yml`.
