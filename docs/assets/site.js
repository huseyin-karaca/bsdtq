'use strict';

// The HTML table is the accessible, no-JavaScript source for this comparison.
const tableRows = [...document.querySelectorAll('tbody tr')];
const datasets = [...document.querySelectorAll('thead th')].slice(1).map(cell => cell.textContent);
const results = tableRows.map(row => ({
  model: row.querySelector('th').textContent.replace('Random Forest', 'RandomForest'),
  values: [...row.querySelectorAll('td')].map(cell => Number(cell.textContent)),
}));
const ours = results.find(row => row.model === 'BSDT-Q');
const select = document.getElementById('baseline');
function drawComparison() {
  const selected = select.value;
  const rows = datasets.map((dataset, index) => {
    const baseline = selected === 'best'
      ? results.filter(row => row !== ours).reduce((best, row) => row.values[index] < best.values[index] ? row : best)
      : results.find(row => row.model === selected);
    const reduction = 100 * (1 - ours.values[index] / baseline.values[index]);
    const row = document.createElement('div');
    row.className = 'bar-row';
    const label = document.createElement('span');
    label.textContent = dataset;
    const track = document.createElement('div');
    track.className = 'bar-track';
    track.setAttribute('aria-hidden', 'true');
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.width = `${Math.max(0, Math.min(100, reduction))}%`;
    track.append(bar);
    const value = document.createElement('output');
    value.textContent = `${reduction.toFixed(1)}%`;
    row.setAttribute('aria-label', `${dataset}: ${reduction.toFixed(1)} percent lower MSE than ${baseline.model}`);
    row.title = `${baseline.model}: ${baseline.values[index]} → BSDT-Q: ${ours.values[index]}`;
    row.append(label, track, value);
    return { element: row, reduction };
  });
  document.getElementById('comparison').replaceChildren(...rows.map(row => row.element));
  const percentages = rows.map(row => row.reduction);
  const summary = document.getElementById('comparison-summary');
  summary.textContent = `${Math.min(...percentages).toFixed(1)}–${Math.max(...percentages).toFixed(1)}% lower reported MSE across the five datasets ${selected === 'best' ? 'versus the strongest baseline in each column' : `versus ${selected.replace('RandomForest', 'Random Forest')}`}.`;
}
select.addEventListener('change', drawComparison);
drawComparison();
document.getElementById('interactive-results').hidden = false;
const copyButton = document.getElementById('copy-citation');
if (navigator.clipboard && window.isSecureContext) {
  copyButton.hidden = false;
  copyButton.addEventListener('click', async () => {
    const status = document.getElementById('copy-status');
    try {
      await navigator.clipboard.writeText(document.getElementById('bibtex').textContent);
      status.textContent = 'Citation copied.';
    } catch {
      status.textContent = 'Copy unavailable. Select the text or download the .bib file.';
    }
  });
}
