# Release Checklist

This checklist covers the GitHub and PyPI setup needed before the first public release of `crucible-ai`.

## Package identity

- PyPI distribution name: `crucible-ai`
- Python import package: `crucible`
- CLI command: `crucible`

## Before the first release

### 1. Confirm package metadata

Check:

- [pyproject.toml](./pyproject.toml)
- [README.md](./README.md)
- [LICENCE](./LICENCE)
- [CITATION.cff](./CITATION.cff)

Make sure the version in `pyproject.toml` matches the intended release.

### 2. Enable GitHub Pages

In the GitHub repository settings:

- go to `Settings -> Pages`
- set the source to `GitHub Actions`

This is required for [`.github/workflows/docs.yml`](./.github/workflows/docs.yml) to deploy the Sphinx site.

### 3. Configure PyPI trusted publishing

In PyPI:

- create the `crucible-ai` project if needed, or reserve the name through first publish
- open the project publishing settings
- add a trusted publisher for this GitHub repository

Recommended trusted publisher settings:

- owner: `ccaprani`
- repository: `crucible`
- workflow: `publish.yml`
- environment: `pypi`

Then in GitHub:

- go to `Settings -> Environments`
- create an environment named `pypi`

The publish workflow already expects this environment.

### 4. Verify GitHub Actions permissions

Check repository settings so Actions are allowed to:

- run workflows
- deploy Pages
- request OIDC tokens for trusted publishing

### 5. Sanity-check local build

In a clean environment:

```bash
pip install -e .[dev]
python -m py_compile crucible/*.py
pytest -q
python -m build
```

Optional:

```bash
pip install twine
twine check dist/*
```

### 6. Sanity-check docs build

```bash
pip install -e .[docs]
sphinx-build -W -b html docs docs/_build/html
```

### 7. Review workflows

Check these files:

- [`.github/workflows/ci.yml`](./.github/workflows/ci.yml)
- [`.github/workflows/docs.yml`](./.github/workflows/docs.yml)
- [`.github/workflows/publish.yml`](./.github/workflows/publish.yml)

## First release

### 1. Bump version

Update the version in [pyproject.toml](./pyproject.toml).

### 2. Commit and tag

```bash
git add .
git commit -m "Release v0.1.0"
git tag v0.1.0
git push origin main --tags
```

### 3. Create GitHub release

Create a GitHub release from the tag:

- tag: `v0.1.0`
- title: `v0.1.0`

Publishing the GitHub release should trigger [`.github/workflows/publish.yml`](./.github/workflows/publish.yml).

### 4. Confirm outputs

Check:

- CI passed
- docs deployed to GitHub Pages
- package published on PyPI

## After release

- verify `pip install crucible-ai` works in a fresh environment
- verify the `crucible` CLI is installed
- verify the docs site is live
- add the PyPI badge and docs badge to [README.md](./README.md) if desired

## Nice-to-have follow-ups

- add status badges to the README
- add a changelog
- add pinned example screenshots of the TUI and HTML report to the docs
- add a release-drafter or changelog workflow if release cadence grows
