# Contributing

Thank you for improving this analytics case study.

## Development workflow

1. Create a virtual environment and install `requirements.txt`.
2. Keep demo generation deterministic by preserving explicit random seeds.
3. Add or update tests for every analytical change.
4. Run `make analyze` and `make test` before opening a pull request.
5. Update the methodology and data dictionary when metric meaning or grain changes.

## Analytical standards

- Do not present synthetic output as real business performance.
- Keep uncertainty with reported estimates.
- Preserve nulls and legitimate empty states.
- Document every exclusion, denominator, and material assumption.
- Separate association, prediction, and causal claims.

## Pull requests

Describe the decision question, changed methodology, verification performed, and any backward-incompatible data-contract changes.

