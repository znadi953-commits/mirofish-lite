import { makeLLM, runAll, estimateCalls, DEFAULTS } from './engine.js';

const HIST_KEY = 'mfl.history';

const $ = (id) => document.getElementById(id);
const PROVIDERS = [
  { id: 'gemini', name: 'Google Gemini (free API key)', base: 'https://generativelanguage.googleapis.com/v1beta/openai', model: 'gemini-3.5-flash-lite' },
  { id: 'openai', name: 'OpenAI', base: 'https://api.openai.com/v1', model: 'gpt-4o-mini' },
  { id: 'groq', name: 'Groq', base: 'https://api.groq.com/openai/v1', model: 'openai/gpt-oss-120b' },
  { id: 'openrouter', name: 'OpenRouter', base: 'https://openrouter.ai/api/v1', model: 'openai/gpt-4o-mini' },
  { id: 'together', name: 'Together AI', base: 'https://api.together.xyz/v1', model: 'meta-llama/Llama-3.3-70B-Instruct-Turbo' },
  { id: 'deepseek', name: 'DeepSeek', base: 'https://api.deepseek.com/v1', model: 'deepseek-chat' },
  { id: 'custom', name: 'Custom (OpenAI-compatible)', base: '', model: '' },
];

const store = {
  get(k) { try { return localStorage.getItem(k) || sessionStorage.getItem(k) || ''; } catch { return ''; } },
  set(k, v, persist) { try { (persist ? localStorage : sessionStorage).setItem(k, v); } catch { /* storage blocked */ } },
  del(k) { try { localStorage.removeItem(k); sessionStorage.removeItem(k); } catch { /* ignore */ } },
};

let files = [];          // {name, text}
let abort = null;
let lastMarkdown = '';

// ---------- setup form ----------

PROVIDERS.forEach((p) => $('provider').add(new Option(p.name, p.id)));
function applyProvider(id, keepModel) {
  const p = PROVIDERS.find((x) => x.id === id) || PROVIDERS[0];
  if (p.id !== 'custom') { $('baseUrl').value = p.base; if (!keepModel) $('model').value = p.model; }
}
$('provider').addEventListener('change', (e) => applyProvider(e.target.value));

(function restore() {
  const prov = store.get('mfl.provider') || 'openai';
  $('provider').value = prov;
  applyProvider(prov);
  if (store.get('mfl.baseUrl')) $('baseUrl').value = store.get('mfl.baseUrl');
  if (store.get('mfl.model')) $('model').value = store.get('mfl.model');
  const key = store.get('mfl.key');
  if (key) { $('apiKey').value = key; $('remember').checked = !!localStorage.getItem('mfl.key'); }
})();

function persistSettings() {
  store.set('mfl.provider', $('provider').value, true);
  store.set('mfl.baseUrl', $('baseUrl').value.trim(), true);
  store.set('mfl.model', $('model').value.trim(), true);
  store.del('mfl.key');
  const key = $('apiKey').value.trim();
  if (key) store.set('mfl.key', key, $('remember').checked);
}

function conn() {
  return { baseUrl: $('baseUrl').value.trim().replace(/\/+$/, ''), apiKey: $('apiKey').value.trim(), model: $('model').value.trim() };
}

$('test').addEventListener('click', async () => {
  const c = conn(), out = $('testResult');
  if (!c.baseUrl || !c.apiKey || !c.model) { out.className = 'small err'; out.textContent = 'Fill in base URL, model and key first.'; return; }
  out.className = 'small muted'; out.textContent = 'Testing…';
  try {
    const llm = makeLLM({ ...c, direct: true, concurrency: 1 });
    const t0 = performance.now();
    await llm([{ role: 'user', content: 'Say hi' }], { maxTokens: 20 });
    out.className = 'small ok'; out.textContent = `Connected (${Math.round(performance.now() - t0)} ms)`;
  } catch (e) { out.className = 'small err'; out.textContent = e.message; }
});

async function loadModels(quiet) {
  const c = conn(), out = $('testResult');
  if (!c.baseUrl || !c.apiKey) { if (!quiet) { out.className = 'small err'; out.textContent = 'Enter base URL and API key first.'; } return; }
  out.className = 'small muted'; out.textContent = 'Loading models…';
  try {
    const r = await fetch(`${c.baseUrl.replace(/\/+$/, '')}/models`, { headers: { Authorization: `Bearer ${c.apiKey}` } });
    const d = await r.json();
    if (!r.ok) throw new Error(d?.error?.message || d?.error || `HTTP ${r.status}`);
    const all = (d.data || d.models || []).map((m) => (typeof m === 'string' ? m : m.id)).filter(Boolean).map((m) => m.replace(/^models\//, ''));
    const skip = /whisper|tts|image|audio|embed|moderation|realtime|transcribe|lyria|banana|robotics|antigravity|deep-research|computer-use|-live|aqa/i;
    const models = all.filter((id) => !skip.test(id)).sort();
    const list = $('modelList'); list.replaceChildren(...models.map((m) => new Option(m, m)));
    const known = models.includes($('model').value.trim());
    const preferred = ['gemini-3.5-flash-lite', 'gemini-3.1-flash-lite', 'gemini-3-flash-preview', 'gpt-4o-mini', 'gpt-oss-120b', 'llama-3.3-70b', 'deepseek-chat'];
    if (!known && models.length) $('model').value = preferred.map((p) => models.find((m) => m.includes(p))).find(Boolean) || models[0];
    out.className = 'small ok'; out.textContent = `${models.length} models available to your key. Using ${$('model').value}.`;
  } catch (e) { out.className = 'small err'; out.textContent = `Could not list models: ${e.message}`; }
}
$('loadModels').addEventListener('click', () => loadModels(false));

// ---------- documents ----------

function renderFiles() {
  const box = $('fileList');
  box.replaceChildren();
  files.forEach((f, i) => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = `${f.name} (${f.text.length.toLocaleString()} chars)`;
    const x = document.createElement('button');
    x.textContent = '×'; x.title = 'Remove';
    x.onclick = () => { files.splice(i, 1); renderFiles(); updateCount(); };
    chip.append(x); box.append(chip);
  });
}
function totalDocs() { return [...files.map((f) => `# ${f.name}\n${f.text}`), $('docs').value.trim()].filter(Boolean).join('\n\n'); }
function updateCount() { $('docCount').textContent = `${totalDocs().length.toLocaleString()} characters (about the first 14,000 are used)`; }
$('docs').addEventListener('input', updateCount);

async function readPdf(file) {
  if (!window.pdfjsLib) {
    await new Promise((res, rej) => {
      const s = document.createElement('script');
      s.src = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js';
      s.onload = res; s.onerror = () => rej(new Error('Could not load the PDF reader'));
      document.head.append(s);
    });
    window.pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
  }
  const pdf = await window.pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise;
  let text = '';
  for (let i = 1; i <= Math.min(pdf.numPages, 40); i++) {
    const page = await pdf.getPage(i);
    text += (await page.getTextContent()).items.map((t) => t.str).join(' ') + '\n';
  }
  return text;
}

async function addFiles(list) {
  for (const f of list) {
    try {
      const text = /\.pdf$/i.test(f.name) ? await readPdf(f) : await f.text();
      if (text.trim()) files.push({ name: f.name, text: text.trim() });
    } catch (e) { showError(`Could not read ${f.name}: ${e.message}`); }
  }
  renderFiles(); updateCount();
}
$('pick').addEventListener('click', (e) => { e.preventDefault(); $('files').click(); });
$('files').addEventListener('change', (e) => { addFiles([...e.target.files]); e.target.value = ''; });
const drop = $('drop');
['dragover', 'dragenter'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => addFiles([...e.dataTransfer.files]));

document.querySelectorAll('[data-ex]').forEach((b) => b.addEventListener('click', async () => {
  try {
    const r = await fetch(`examples/${b.dataset.ex}.md`);
    if (!r.ok) throw new Error('not found');
    $('docs').value = await r.text(); files = []; renderFiles(); updateCount();
    $('docs').scrollIntoView({ behavior: 'smooth', block: 'center' });
  } catch { showError('Could not load the example.'); }
}));

const POP_MIN = 20, POP_MAX = 100000;
const popFromSlider = (v) => Math.round(Math.exp(Math.log(POP_MIN) + (v / 100) * (Math.log(POP_MAX) - Math.log(POP_MIN))));
const sliderFromPop = (p) => Math.round(((Math.log(p) - Math.log(POP_MIN)) / (Math.log(POP_MAX) - Math.log(POP_MIN))) * 100);
const fmt = (n) => n.toLocaleString('en-US');
function populationNow() { return Math.min(POP_MAX, Math.max(POP_MIN, Math.round(+$('population').value) || 1000)); }

function paramsNow() {
  const population = populationNow();
  const custom = +$('agents').value;
  const depth = $('depth').value;
  let agents = custom >= 6 ? Math.min(120, custom) : depth === 'auto' ? 24 : +depth;
  agents = Math.min(agents, population);
  return {
    population, agents: Math.max(6, agents),
    rounds: Math.min(5, Math.max(2, +$('rounds').value || DEFAULTS.rounds)),
    concurrency: Math.min(8, Math.max(1, +$('concurrency').value || DEFAULTS.concurrency)),
  };
}
function updateHint() {
  const p = paramsNow(), n = estimateCalls(p);
  const min = Math.max(1, Math.round(n / p.concurrency / 4)), max = Math.max(2, Math.round(n / p.concurrency / 1.5));
  $('runHint').textContent = `About ${n} AI calls, roughly ${min}-${max} minutes on a normal key; slower on free tiers.`;
  const same = p.population <= p.agents;
  $('scaleSummary').className = 'callout' + (n > 300 ? ' warn' : '');
  $('scaleSummary').textContent = (same
    ? `Every one of the ${fmt(p.population)} people will be simulated individually by your model. `
    : `${fmt(p.agents)} AI personas will be simulated by your model and scaled up to represent ${fmt(p.population)} people by audience segment. `)
    + `That is about ${n} AI calls. ` + (n > 300 ? 'This is a large run: use a paid key with high rate limits, or choose a lighter depth.' : p.agents <= 12 ? 'Light runs suit free or rate-limited keys.' : 'Needs a key with reasonable rate limits.');
}
const DEPTHS = [['auto', 'Auto', 'recommended'], ['12', 'Light', '12 personas'], ['24', 'Standard', '24 personas'], ['48', 'Deep', '48 personas'], ['96', 'Maximum', '96 personas']];
(function buildSeg() {
  const seg = $('seg');
  DEPTHS.forEach(([v, name, sub]) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'segbtn'; b.dataset.v = v; b.setAttribute('role', 'radio');
    b.innerHTML = `${name}<small>${sub}</small>`;
    b.onclick = () => { $('depth').value = v; $('depth').dispatchEvent(new Event('input', { bubbles: true })); syncSeg(); };
    seg.append(b);
  });
  syncSeg();
})();
function syncSeg() { document.querySelectorAll('#seg button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.v === $('depth').value))); }

function setPopulation(p) { $('population').value = p; $('popSlider').value = sliderFromPop(Math.min(POP_MAX, Math.max(POP_MIN, p))); updateHint(); }
$('popSlider').addEventListener('input', (e) => { const v = popFromSlider(+e.target.value); const r = v >= 1000 ? Math.round(v / 100) * 100 : v >= 100 ? Math.round(v / 10) * 10 : v; $('population').value = r; updateHint(); });
$('population').addEventListener('input', () => { const p = populationNow(); $('popSlider').value = sliderFromPop(p); updateHint(); });
$('presets').addEventListener('click', (e) => { const b = e.target.closest('[data-pop]'); if (b) setPopulation(+b.dataset.pop); });
['depth', 'agents', 'rounds', 'concurrency'].forEach((id) => $(id).addEventListener('input', updateHint));

// ---------- helpers ----------

function showError(msg) { const el = $('error'); el.textContent = msg; el.hidden = false; }
function clearError() { $('error').hidden = true; }
const show = (id) => { ['setup', 'progress', 'result'].forEach((s) => ($(s).hidden = s !== id)); $('learn').hidden = id !== 'setup'; window.scrollTo(0, 0); };

function postEl(p) {
  const d = document.createElement('div');
  d.className = 'post';
  const who = document.createElement('div');
  const b = document.createElement('span'); b.className = 'who'; b.textContent = p.name;
  const m = document.createElement('span'); m.className = 'meta'; m.textContent = ` ${p.segment} · ${p.role} · round ${p.round}`;
  const t = document.createElement('span'); t.className = `tag s${p.stance}`; t.textContent = `stance ${p.stance > 0 ? '+' : ''}${p.stance}`;
  who.append(b, m, t);
  const body = document.createElement('div'); body.textContent = p.text;
  d.append(who, body);
  return d;
}

function agentEl(a) {
  const d = document.createElement('div');
  d.className = 'agent'; d.id = `ag-${a.id}`;
  const n = document.createElement('b'); n.textContent = a.name;
  const s = document.createElement('span'); s.className = 'muted'; s.textContent = `${a.segment} · ${a.role}`;
  const bar = document.createElement('div'); bar.className = 'bar';
  const fill = document.createElement('i'); bar.append(fill);
  d.append(n, s, bar);
  setBar(fill, 0);
  return d;
}
function setBar(fill, stance) {
  const w = Math.abs(stance) * 25;
  fill.style.width = `${w}%`;
  fill.style.left = stance >= 0 ? '50%' : `${50 - w}%`;
  fill.style.background = stance > 0 ? 'var(--good)' : stance < 0 ? 'var(--bad)' : 'var(--muted)';
}

// ---------- charts (inline SVG) ----------

function trendChart(stats) {
  const W = 320, H = 150, pad = 26;
  const pts = stats.rounds.map((r, i) => [pad + (i * (W - 2 * pad)) / Math.max(1, stats.rounds.length - 1), H / 2 - (r.mean / 2) * (H / 2 - 18)]);
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('width', '100%');
  const mk = (tag, attrs, text) => { const e = document.createElementNS(ns, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text) e.textContent = text; svg.append(e); return e; };
  mk('line', { x1: pad, x2: W - pad, y1: H / 2, y2: H / 2, stroke: 'var(--line)', 'stroke-width': 1 });
  mk('text', { x: 2, y: 16, fill: 'var(--muted)', 'font-size': 10 }, '+2');
  mk('text', { x: 2, y: H - 6, fill: 'var(--muted)', 'font-size': 10 }, '-2');
  mk('polyline', { points: pts.map((p) => p.join(',')).join(' '), fill: 'none', stroke: 'var(--brand)', 'stroke-width': 2.5 });
  pts.forEach((p, i) => {
    mk('circle', { cx: p[0], cy: p[1], r: 4, fill: 'var(--brand)' });
    mk('text', { x: p[0], y: H - 6, 'text-anchor': 'middle', fill: 'var(--muted)', 'font-size': 10 }, `R${i + 1}`);
    mk('text', { x: p[0], y: p[1] - 9, 'text-anchor': 'middle', fill: 'var(--ink)', 'font-size': 10 }, `${stats.rounds[i].mean >= 0 ? '+' : ''}${stats.rounds[i].mean.toFixed(2)}`);
  });
  return svg;
}

function segmentChart(stats) {
  const wrap = document.createElement('div');
  stats.segments.forEach((s) => {
    const row = document.createElement('div');
    row.style.cssText = 'display:grid;grid-template-columns:42% 1fr 44px;gap:8px;align-items:center;font-size:12px;margin:5px 0';
    const name = document.createElement('span'); name.textContent = s.name;
    const bar = document.createElement('div'); bar.className = 'bar'; bar.style.height = '8px';
    const fill = document.createElement('i'); bar.append(fill); setBar(fill, s.mean);
    const val = document.createElement('span'); val.className = 'muted'; val.textContent = `${s.mean >= 0 ? '+' : ''}${s.mean.toFixed(2)}`;
    row.append(name, bar, val); wrap.append(row);
  });
  return wrap;
}

function renderCharts(stats) {
  const box = $('charts'); box.replaceChildren();
  const mk = (title, node) => { const c = document.createElement('div'); c.className = 'chartCard'; const h = document.createElement('h4'); h.textContent = title; c.append(h, node); box.append(c); };
  mk('Mean stance by round (-2 hostile .. +2 enthusiastic)', trendChart(stats));
  mk('Final stance by segment', segmentChart(stats));
}

// ---------- learning & calibration ----------

function loadHist() { try { return JSON.parse(localStorage.getItem(HIST_KEY) || '[]'); } catch { return []; } }
function saveHist(h) { try { localStorage.setItem(HIST_KEY, JSON.stringify(h.slice(-40))); } catch { /* storage blocked */ } }

function calibrationFor(key) {
  const done = loadHist().filter((r) => r.key === key && r.real && r.raw > 0);
  if (!done.length) return null;
  const logs = done.map((r) => Math.log(Math.min(3, Math.max(0.1, r.real.rate / Math.max(0.02, r.raw)))));
  const geo = Math.exp(logs.reduce((a, b) => a + b, 0) / logs.length);
  const n = done.length;
  return { ratio: Math.min(3, Math.max(0.1, 1 + (geo - 1) * (n / (n + 2)))), n };   // shrink toward 1 while data is scarce
}

function saveRun(result, key, params) {
  const h = loadHist(), pr = result.projection;
  h.push({ id: String(Date.now()), ts: new Date().toISOString(), key, product: result.world.product?.name || key, population: pr.population, panel: pr.panel, rounds: params.rounds,
    raw: pr.overall.rate, lo: pr.overall.lo, hi: pr.overall.hi, verdict: result.report.head.verdict, real: null });
  saveHist(h);
}

function renderLearn() {
  const h = loadHist().slice().reverse(), box = $('learnList');
  box.replaceChildren();
  if (!h.length) { box.innerHTML = '<p class="muted small">No runs yet. Your runs will appear here.</p>'; return; }
  h.forEach((r) => {
    const row = document.createElement('div'); row.className = 'lrow';
    const a1 = document.createElement('div');
    a1.innerHTML = `<b></b><div class="muted"></div>`;
    a1.querySelector('b').textContent = r.product;
    a1.querySelector('.muted').textContent = `${new Date(r.ts).toLocaleDateString()} · ${fmt(r.panel)} personas for ${fmt(r.population)} people`;
    const a2 = document.createElement('div');
    a2.textContent = `Predicted interest: ${Math.round(r.raw * 100)}%` + (r.population > r.panel ? ` (${Math.round(r.lo * 100)}-${Math.round(r.hi * 100)}%)` : '');
    const a3 = document.createElement('div'); a3.className = 'real';
    const inp = document.createElement('input'); inp.type = 'number'; inp.min = 0; inp.max = 100; inp.placeholder = 'Real %'; inp.value = r.real ? Math.round(r.real.rate * 100) : '';
    inp.setAttribute('aria-label', 'Real-world interest rate in percent');
    const btn = document.createElement('button'); btn.className = 'ghost small'; btn.textContent = r.real ? 'Update' : 'Save real result';
    btn.onclick = () => {
      const v = parseFloat(inp.value);
      if (!(v >= 0 && v <= 100)) { $('learnStatus').textContent = 'Enter a percentage from 0 to 100.'; return; }
      const all = loadHist(), t = all.find((x) => x.id === r.id); t.real = { rate: v / 100, ts: new Date().toISOString() }; saveHist(all); renderLearn();
      const c = calibrationFor(r.key); $('learnStatus').textContent = c ? `Saved. Future runs for "${r.key}" will be adjusted by x${c.ratio.toFixed(2)}.` : 'Saved.';
    };
    const hint = document.createElement('span'); hint.className = 'muted'; hint.textContent = '% of people who really showed interest';
    a3.append(inp, btn, hint);
    if (r.real) { const b = document.createElement('span'); b.className = 'calbadge'; b.textContent = `Real ${Math.round(r.real.rate * 100)}%`; a3.append(b); }
    row.append(a1, a2, a3); box.append(row);
  });
}
$('expLearn').addEventListener('click', () => download('mirofish-lite-learning.json', JSON.stringify(loadHist(), null, 2), 'application/json'));
$('impFile').addEventListener('change', async (e) => {
  try { const data = JSON.parse(await e.target.files[0].text()); if (!Array.isArray(data)) throw new Error('bad file'); saveHist([...loadHist(), ...data.filter((r) => r && r.key && typeof r.raw === 'number')]); renderLearn(); $('learnStatus').textContent = 'Imported.'; }
  catch { $('learnStatus').textContent = 'Could not read that file.'; }
  e.target.value = '';
});
$('impLearn').addEventListener('click', () => $('impFile').click());
$('clrLearn').addEventListener('click', () => { if (confirm('Delete all saved runs and real results from this browser?')) { saveHist([]); renderLearn(); } });

// ---------- report cover ----------

function esc(s) { return String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
function renderCover(result) {
  const { stats, report, world } = result, last = stats.rounds[stats.rounds.length - 1];
  const v = report.head.verdict, cls = v === 'Go' ? 'v-go' : v === 'Hold' ? 'v-hold' : 'v-chg';
  const sign = (x) => (x >= 0 ? '+' : '') + x.toFixed(2);
  $('cover').innerHTML = `
    <div class="eyebrow">Simulation report · ${esc(world.product?.name || 'Product')} · ${new Date().toISOString().slice(0, 10)}</div>
    <h1>${esc(report.head.title)}</h1>
    <p class="lede">${esc(report.head.summary)}</p>
    <div><span class="verdict ${cls}">${esc(v)}</span><span class="muted small">${esc(report.head.verdict_reason || '')}</span></div>
    <div class="stats">
      <div class="stat"><b>${fmt(result.projection.population)}</b><span>people represented</span></div>
      <div class="stat"><b>${stats.n}</b><span>AI personas simulated</span></div>
      <div class="stat"><b>${sign(stats.final)}</b><span>final average stance (-2 to +2)</span></div>
      <div class="stat"><b>${Math.round(result.projection.overall.rate * 100)}%</b><span>projected positive intent${result.projection.full ? '' : ` (${Math.round(result.projection.overall.lo * 100)}-${Math.round(result.projection.overall.hi * 100)}%)`}</span></div>
    </div><p class="small muted" style="margin:12px 0 0">When you know what really happened, record it under "Learning &amp; calibration" on the home screen (New run) to improve future projections.</p>${result.projection.calibrated ? `<p class="small muted" style="margin:12px 0 0">Calibrated with ${result.projection.calibrated.n} real result(s): about ${Math.round(result.projection.calibrated.rate * 100)}% after adjustment.</p>` : ''}`;
}

// ---------- report output ----------

function renderReport(md) {
  lastMarkdown = md;
  const html = window.marked ? window.marked.parse(md, { gfm: true }) : `<pre>${md.replace(/</g, '&lt;')}</pre>`;
  $('report').innerHTML = window.DOMPurify ? window.DOMPurify.sanitize(html) : html;
}

function download(name, text, type) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
const slug = () => (lastMarkdown.match(/^# (.+)$/m)?.[1] || 'mirofish-report').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 60);

$('copy').addEventListener('click', async (e) => { await navigator.clipboard.writeText(lastMarkdown); e.target.textContent = 'Copied'; setTimeout(() => (e.target.textContent = 'Copy Markdown'), 1500); });
$('dlmd').addEventListener('click', () => download(`${slug()}.md`, lastMarkdown, 'text/markdown'));
$('dlhtml').addEventListener('click', () => {
  const css = [...document.styleSheets].map((s) => { try { return [...s.cssRules].map((r) => r.cssText).join('\n'); } catch { return ''; } }).join('\n');
  download(`${slug()}.html`, `<!doctype html><meta charset="utf-8"><title>${slug()}</title><style>${css}</style><body><div class="wrap">${$('cover').outerHTML}${$('charts').outerHTML}<article class="report">${$('report').innerHTML}</article></div>`, 'text/html');
});
$('print').addEventListener('click', () => window.print());
$('again').addEventListener('click', () => { clearError(); show('setup'); });

const STAGE_TEXT = {
  world: 'Reading your documents: who the buyers are, what they care about, what competes with you.',
  agents: 'Creating the simulated people who will react to your product.',
  simulate: 'Running the simulation: people post, reply to each other, and change their minds as events happen.',
  interview: 'Interviewing the most important simulated people.',
  report: 'Writing the report from the feed, the interviews and the counted statistics.',
};

// ---------- run ----------

$('cancel').addEventListener('click', () => { if (abort) abort.cancelled = true; show('setup'); });

$('run').addEventListener('click', async () => {
  clearError();
  const c = conn();
  const evidence = $('evidence').value.trim();
  const productKey = ($('productName').value.trim() || (totalDocs().split('\n').find((l) => l.replace(/[#*\s]/g, '').length > 3) || 'product').replace(/[#*]/g, '').trim().slice(0, 60)).toLowerCase();
  const calibration = calibrationFor(productKey);
  const past = calibration ? `\n\nPAST RUNS FOR THIS PRODUCT: simulated results have been ${calibration.ratio.toFixed(2)} times the real results on average (${calibration.n} comparisons).` : '';
  const docs = totalDocs() + (evidence ? `\n\nREAL-WORLD EVIDENCE SUPPLIED BY THE USER (treat as facts):\n${evidence}` : '') + past;
  if (window.mflAccess && !window.mflAccess.canRun()) return;
  if (!c.baseUrl || !c.apiKey || !c.model) return showError('Step 1: base URL, model and API key are all required.');
  if (docs.length < 80) return showError('Step 2: add your product documents (at least a short description).');
  persistSettings();
  if (window.mflAccess) window.mflAccess.countRun();

  const params = paramsNow();
  const totalCalls = estimateCalls(params);
  let done = 0, lastActivity = Date.now();
  const state = { cancelled: false };
  abort = state;

  const llm = makeLLM({ ...c, direct: true, concurrency: params.concurrency, isCancelled: () => state.cancelled,
    onCallDone: () => { done++; lastActivity = Date.now(); const pct = Math.min(96, Math.round((done / totalCalls) * 100)); $('pfill').style.width = `${Math.max(2, pct)}%`; $('pcalls').textContent = `${done} of about ${totalCalls} AI calls done`; $('slow').hidden = true; },
    onRetry: (e, ms) => { $('slow').hidden = false; $('slow').textContent = `Your AI provider is rate-limiting this key, so the run is waiting ${Math.round(ms / 1000)}s and retrying automatically. Nothing is wrong. Free plans are slow; lowering "Parallel calls" to 1 helps.`; } });

  $('feed').replaceChildren(); $('agentGrid').replaceChildren(); $('status').textContent = '';
  document.querySelectorAll('#stages li').forEach((li) => (li.className = ''));
  show('progress');

  const order = ['world', 'agents', 'simulate', 'interview', 'report'];
  const t0 = Date.now();
  $('pfill').style.width = '2%'; $('pcalls').textContent = 'Starting…'; $('slow').hidden = true;
  const timer = setInterval(() => {
    if ($('progress').hidden) return clearInterval(timer);
    const s = Math.round((Date.now() - t0) / 1000);
    $('elapsed').textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')} elapsed`;
    if (Date.now() - lastActivity > 45000 && $('slow').hidden) { $('slow').hidden = false; $('slow').textContent = 'Still working: one AI call is taking longer than usual (large models can be slow). It will time out and retry on its own. You can also Cancel and try a faster model.'; }
  }, 1000);

  try {
    const result = await runAll(llm, { docs, calibration, scenario: $('scenario').value.trim() || 'How will buyers react?', outreach: $('outreach').value.trim(), language: $('language').value.trim(), params }, {
      onStage: (s, detail) => {
        if (state.cancelled) return;
        const i = order.indexOf(s);
        document.querySelectorAll('#stages li').forEach((li) => { const j = order.indexOf(li.dataset.s); li.className = i === -1 || j < i ? 'done' : j === i ? 'active' : ''; });
        $('status').textContent = STAGE_TEXT[s] || detail;
        if (s === 'done') $('pfill').style.width = '100%';
      },
      onAgents: (agents) => agents.forEach((a) => $('agentGrid').append(agentEl(a))),
      onPost: (p) => { const f = $('feed'); f.prepend(postEl(p)); },
      onRound: (r, h) => { $('status').textContent = `Round ${r} of ${params.rounds} finished: ${h.event}`; h.snapshot.forEach((s) => { const el = $(`ag-${s.id}`); if (el) setBar(el.querySelector('i'), s.stance); }); },
    });
    if (state.cancelled) return;
    saveRun(result, productKey, params);
    renderCover(result);
    renderCharts(result.stats);
    renderReport(result.report.markdown);
    $('rawFeed').replaceChildren(...result.sim.feed.map(postEl));
    show('result');
    renderLearn();
  } catch (e) {
    if (state.cancelled) return;
    show('setup');
    const msg = /does not exist|do not have access|model_not_found/i.test(e.message) ? `${e.message} Click "Load my models" in step 1 to pick a model your key can use.` : /401|incorrect api key|invalid api key|unauthor/i.test(e.message) ? 'The provider rejected your API key. Check it in step 1 and use "Test connection".' : e.message;
    showError(`Run failed: ${msg}`);
  } finally { clearInterval(timer); }
});

function updateReady() {
  const c = conn();
  $('s1').classList.toggle('ready', !!(c.baseUrl && c.apiKey && c.model));
  $('s2').classList.toggle('ready', totalDocs().length >= 80);
  $('s3').classList.toggle('ready', $('scenario').value.trim().length > 10);
  ['1', '2', '3'].forEach((k) => $('c' + k).classList.toggle('ok', $('s' + k).classList.contains('ready')));
}
['baseUrl', 'apiKey', 'model', 'docs', 'scenario', 'provider'].forEach((id) => $(id).addEventListener('input', updateReady));
$('provider').addEventListener('change', updateReady);
document.querySelectorAll('[data-ex]').forEach((b) => b.addEventListener('click', () => setTimeout(updateReady, 400)));
$('files').addEventListener('change', () => setTimeout(updateReady, 400));
setPopulation(1000);
renderLearn();
updateCount();
updateHint();
updateReady();
