# Research website

A buildless static project page for BSDT-Q. Serve this directory directly; no Node.js, package installation, or framework build is required.

```bash
python3 -m http.server 8000 --directory docs
```

Open http://localhost:8000. The page supports narrow screens, keyboard navigation, reduced motion, and a readable no-JavaScript fallback. The comparison reads the HTML results table; citation download and all resource links work without JavaScript. The optional Google Fonts request falls back to local sans-serif and monospace fonts.

## Edit content

| File | Edit here |
| :--- | :--- |
| `index.html` | Research narrative, resource links, published results table, article citation |
| `assets/style.css` | Typography, colors, spacing, responsive layout |
| `assets/method*.svg` | Original editable diagrams of Algorithm 1; include accessible titles and descriptions |
| `assets/site.js` | Baseline comparison and clipboard enhancement |
| `assets/paper.bib` | Article and capsule BibTeX |
| `data/results.csv` | Machine-readable transcription of published Table II |
| `data/README.md` | Result provenance and calculation definitions |

When editing results, update both the HTML table and CSV, plus the root README. Run `python3 tools/check_site.py` from the repository root to verify they match and to check local assets, anchor links, and SVG syntax.

## Hosting

All asset and download paths are relative. Copy this directory's contents to the `/bsdtq/` route on hkaraca.com, or publish it as a GitHub Pages project site. The main `huseyin-karaca.github.io` Pages site already has the verified custom domain `hkaraca.com`; HIT-ASR confirms the project-site pattern. This project uses `https://hkaraca.com/bsdtq/`. No DNS change or project-level `CNAME` is needed.

For GitHub Pages, enable **Settings → Pages → Source → GitHub Actions**, then manually run **Deploy research website** from the Actions tab. The workflow uploads only `docs/`. It does not publish the supplied reference PDF or Python research files. The article button points to the published paper.

The design uses the same Schibsted Grotesk / JetBrains Mono font pairing and blue/gray palette as HIT-ASR, with a lighter single-page structure for this paper. Deployment uses the repository’s GitHub Pages workflow and inherits the account’s existing custom domain.
