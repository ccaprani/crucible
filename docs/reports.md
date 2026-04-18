# Reports

Crucible reports are standalone HTML files.
They are designed to work as local artifacts, not as pages that depend on the rest of the documentation site.

## Generate a report

```bash
crucible report
crucible report -m gemma4:31b qwen3.5:27b
crucible report --no-open -o results/report_24gb.html
```

The report includes:

- summary model cards
- pass-rate and category charts
- per-test grouped comparisons
- tier and model filtering
- model metadata when available

## Why keep it standalone

The report is meant to be:

- opened locally from disk
- attached to emails or shared folders
- archived as an artifact of a benchmark run

That is why the report keeps its own HTML, CSS, and JavaScript rather than depending on the docs theme.

## Example report

An example report artifact is checked into the docs tree here:

- [Open the raw example report](./_static/example-report.html)

You can also view it inline below when browsing the built docs site.

<iframe
  src="_static/example-report.html"
  title="Example Crucible report"
  style="width: 100%; min-height: 820px; border: 1px solid var(--color-border); border-radius: 10px; background: white;"
></iframe>
