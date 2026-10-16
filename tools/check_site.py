"""Dependency-free checks for local links and the published results transcription."""
import csv
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.rows = set(), [], []
        self.in_body, self.cell = False, None
        self.row = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if 'id' in attrs:
            assert attrs['id'] not in self.ids, f'Duplicate ID: {attrs["id"]}'
            self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append(attrs[key])
        if tag == 'tbody':
            self.in_body = True
        if self.in_body and tag == 'tr':
            self.row = []
        if self.in_body and tag in ('th', 'td'):
            self.cell = ''

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if self.in_body and tag in ('th', 'td'):
            self.row.append(self.cell.strip())
            self.cell = None
        if self.in_body and tag == 'tr':
            self.rows.append(self.row)
        if tag == 'tbody':
            self.in_body = False


page = Page()
page.feed((DOCS / 'index.html').read_text())
for link in page.links:
    parts = urlsplit(link)
    if parts.scheme or parts.netloc:
        continue
    if parts.path:
        path = (DOCS / unquote(parts.path)).resolve()
        assert path.is_relative_to(DOCS), f'Link outside site: {link}'
        assert path.is_file(), f'Missing asset: {link}'
    elif parts.fragment:
        assert parts.fragment in page.ids, f'Missing anchor: {link}'

with (DOCS / 'data/results.csv').open() as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == len(page.rows) == 9
readme = (ROOT / 'README.md').read_text()
for data, html in zip(rows, page.rows):
    assert data['model'] == html[0].replace('Random Forest', 'RandomForest')
    values = [float(v) for key, v in data.items() if key != 'model']
    assert values == [float(v) for v in html[1:]], f'Results mismatch: {data["model"]}'
    readme_rows = [line for line in readme.splitlines() if line.startswith('| ') and line.count('|') == 7]
    matches = [line for line in readme_rows if line.split('|')[1].strip().replace('**', '').replace('Random Forest', 'RandomForest') == data['model']]
    assert len(matches) == 1, f'Missing README result: {data["model"]}'
    assert values == [float(v.strip().replace('**', '')) for v in matches[0].split('|')[2:-1]]

for svg in (DOCS / 'assets').glob('*.svg'):
    ET.parse(svg)
assert '10.24433/CO.9409420.v1' in (DOCS / 'index.html').read_text()
print(f'OK: {len(page.links)} links, {len(page.ids)} IDs, 9 model rows in HTML/CSV/README, SVG syntax.')
