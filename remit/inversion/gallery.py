"""
A local gallery of the inversion figures: inversions/results/index.html, one page with every result map (a section
per region folder) and every report figure (a section per report), newest first, each with the description from its
config's leading comments or the report section it sits in. Figures from the last 24 h are marked new. Rewritten after
every run; also
    python -m remit.inversion gallery [--open]
"""
import glob
import html
import os
import re
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS = os.path.join(ROOT, 'inversions', 'results')
REPORTS = os.path.join(ROOT, 'docs', 'reports')
OUT = os.path.join(RESULTS, 'index.html')
REGIONS = [('global', 'Global'), ('aus', 'Australia'), ('afr', 'Africa'), ('sam', 'South America'),
           ('nam', 'North America'), ('ant', 'Antarctica'), ('eur', 'Eurasia'), ('zea', 'Zealandia')]


def config_note(stem):
    """the leading comment lines of the config a figure came from (its name minus the figure suffix)"""
    base = re.sub(r'_(maps|[a-z]+)$', '', stem)
    for name in (base, stem):
        path = os.path.join(ROOT, 'inversions', name + '.toml')
        if os.path.exists(path):
            lines = []
            for line in open(path):
                if not line.startswith('#'):
                    break
                lines.append(line.lstrip('#').strip())
            return ' '.join(lines)
    return ''


def report_notes(folder):
    """{figure file: the report section heading it sits under}"""
    readme = os.path.join(folder, 'README.md')
    notes, heading = {}, ''
    if os.path.exists(readme):
        for line in open(readme):
            if line.startswith('#'):
                heading = line.lstrip('#').strip()
            for f in re.findall(r'!\[[^\]]*\]\(([^)]+\.png)\)', line):
                notes[f] = heading
    return notes


def card(path, note, now):
    t = os.path.getmtime(path)
    new = now - t < 86400
    rel = os.path.relpath(path, RESULTS)
    name = os.path.basename(path)
    return t, (f'<figure class="card{" new" if new else ""}" data-text="{html.escape((name + " " + note).lower())}">'
               f'<a href="{html.escape(rel)}" target="_blank"><img loading="lazy" src="{html.escape(rel)}" alt=""></a>'
               f'<figcaption><b>{html.escape(name)}</b>{" <span class=tag>new</span>" if new else ""}'
               f'<span class="when">{time.strftime("%d %b %H:%M", time.localtime(t))}</span>'
               f'<span class="note">{html.escape(note)}</span></figcaption></figure>')


def write():
    now = time.time()
    sections = []
    for folder in sorted(glob.glob(os.path.join(REPORTS, '*')), reverse=True):
        notes = report_notes(folder)
        cards = [card(p, notes.get(os.path.basename(p), ''), now) for p in glob.glob(os.path.join(folder, '*.png'))]
        if cards:
            sections.append((f'Report {os.path.basename(folder)}', cards))
    for tag, title in REGIONS:
        cards = [card(p, config_note(os.path.splitext(os.path.basename(p))[0]), now)
                 for p in glob.glob(os.path.join(RESULTS, tag, '*.png'))]
        if cards:
            sections.append((title, cards))
    nav = ' · '.join(f'<a href="#s{i}">{html.escape(t)} ({len(c)})</a>' for i, (t, c) in enumerate(sections))
    body = ''.join(f'<h2 id="s{i}">{html.escape(t)}</h2><div class="grid">'
                   + ''.join(c for _, c in sorted(cards, reverse=True)) + '</div>'
                   for i, (t, cards) in enumerate(sections))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Inversion figures</title><style>
:root {{ --bg:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --line:#e1e0d9; --accent:#2a78d6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d0d0d; --card:#1a1a19; --ink:#fff; --muted:#c3c2b7;
  --line:#2c2c2a; --accent:#3987e5; }} }}
body {{ margin:0; padding:16px 24px; background:var(--bg); color:var(--ink); font:14px system-ui, sans-serif; }}
header {{ position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line); z-index:1; }}
h1 {{ font-size:18px; margin:0 0 6px; }} h2 {{ font-size:16px; margin:24px 0 8px; }}
nav a {{ color:var(--accent); text-decoration:none; }} input {{ margin-top:6px; padding:4px 8px; width:320px;
  max-width:100%; background:var(--card); color:var(--ink); border:1px solid var(--line); border-radius:4px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(340px, 1fr)); gap:12px; }}
.card {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:6px; overflow:hidden; }}
.card.new {{ border-color:var(--accent); }} .card img {{ width:100%; display:block; background:#fff; }}
figcaption {{ padding:6px 8px; font-size:12px; display:flex; flex-wrap:wrap; gap:4px 8px; }}
.when {{ color:var(--muted); }} .note {{ color:var(--muted); flex-basis:100%; }}
.tag {{ color:var(--accent); font-weight:600; }}
</style></head><body><header><h1>Inversion figures</h1><nav>{nav}</nav>
<input id="q" placeholder="Filter by name or description" oninput="f(this.value)">
<span style="color:var(--muted)"> updated {time.strftime('%d %b %Y %H:%M')}; blue frame: last 24 h</span></header>
{body}<script>function f(q){{q=q.toLowerCase();document.querySelectorAll('.card').forEach(c=>
c.style.display=c.dataset.text.includes(q)?'':'none');}}</script></body></html>"""
    with open(OUT, 'w') as fh:
        fh.write(page)
    return OUT
