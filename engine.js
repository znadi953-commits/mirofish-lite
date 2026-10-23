// MiroFish-Lite engine: runs in the browser (and in Node for tests).
// Pipeline: world model -> agent personas -> multi-round social simulation
// -> agent interviews -> computed statistics -> chaptered report.

export const DEFAULTS = { agents: 12, rounds: 3, concurrency: 4, batch: 4, population: 12 };

// Tolerant extraction: models sometimes return the wanted list under another key,
// as a single object, or as a map keyed by agent id. Find something shaped right.
export function pickList(d, preferred, probe) {
  if (!d || typeof d !== 'object') return [];
  const toItem = (kk, vv) => (vv && typeof vv === 'object' ? Object.assign({ agent_id: kk, id: kk }, vv) : null);
  for (const k of preferred) {
    const v = d[k];
    if (Array.isArray(v)) return v.filter((x) => x && typeof x === 'object');
    if (v && typeof v === 'object') return Object.entries(v).map(([kk, vv]) => toItem(kk, vv)).filter(Boolean);
  }
  for (const [k, v] of Object.entries(d)) {
    if (preferred.includes(k)) continue;
    if (Array.isArray(v) && v.some((x) => x && typeof x === 'object' && probe(x))) return v.filter((x) => x && typeof x === 'object');
    if (v && typeof v === 'object' && !Array.isArray(v) && probe(v)) return [v];
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      const ents = Object.entries(v);
      if (ents.length && ents.every(([, vv]) => vv && typeof vv === 'object')) return ents.map(([kk, vv]) => toItem(kk, vv)).filter(Boolean);
    }
  }
  return [];
}

const SYS = `You are one component of MiroFish-Lite, a social-simulation engine that predicts how people react to a product.
Rules:
- The SOURCE DOCUMENTS are the only factual input. Never invent statistics, prices, percentages, customer counts, revenue, latencies or market data. If the source has no figure, speak qualitatively.
- Claims made by the vendor (for example "10-20x cheaper") stay labelled as claims, never as results.
- Simulated people express opinions and reactions, not facts.
- Output exactly the format requested, nothing else.`;

export function estimateCalls({ agents, rounds, batch }) {
  batch = batch || (agents > 40 ? 6 : 4);
  return 1 + Math.ceil(agents / 6) + rounds * Math.ceil(agents / batch) + Math.min(6, agents) + 7;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clip = (s, n) => (s.length > n ? s.slice(0, n) + '…' : s);

export function parseJSON(text) {
  let t = String(text || '').replace(/<think>[\s\S]*?<\/think>/g, '').trim();
  t = t.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
  try { return JSON.parse(t); } catch { /* fall through */ }
  const o = t.indexOf('{'), a = t.indexOf('[');
  const start = o === -1 ? a : a === -1 ? o : Math.min(o, a);
  if (start === -1) throw new Error('No JSON in model output');
  const end = Math.max(t.lastIndexOf('}'), t.lastIndexOf(']'));
  return JSON.parse(t.slice(start, end + 1));
}

// ---------- LLM access (through the /api/chat proxy) ----------

export function makeLLM({ baseUrl, apiKey, model, concurrency = DEFAULTS.concurrency, endpoint = '/api/chat', direct = false, fetchImpl, onUsage, onRetry, onCallDone, isCancelled }) {
  const f = fetchImpl || ((...a) => fetch(...a));
  let active = 0;
  const waiting = [];
  const acquire = () => new Promise((res) => { if (active < concurrency) { active++; res(); } else waiting.push(res); });
  const release = () => { const n = waiting.shift(); if (n) n(); else active--; };

  async function once(messages, o) {
    if (direct) return onceDirect(messages, o);
    const r = await f(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ baseUrl, apiKey, model, messages, json: !!o.json, max_tokens: o.maxTokens || 6000, temperature: o.temperature ?? 0.7 }),
    });
    let data = {};
    try { data = await r.json(); } catch { /* non-JSON */ }
    if (!r.ok) {
      const err = new Error(data.error || `HTTP ${r.status}`);
      err.status = r.status;
      err.retryAfter = data.retryAfter;
      // Daily/billing quota or a wait over 90 s will not fix itself mid-run: stop with a clear message.
      const quota = r.status === 429 && (data.quota || (data.retryAfter || 0) > 90);
      if (quota) err.message = `Your provider's rate or quota limit was reached${data.retryAfter ? ` (resets in about ${Math.ceil(data.retryAfter / 60)} min)` : ''}. Try again later, lower "Parallel calls", use another model, or upgrade the plan. Provider said: ${err.message}`;
      err.fatal = quota || r.status === 401 || r.status === 400 || r.status === 404;
      throw err;
    }
    if (data.usage && onUsage) onUsage(data.usage);
    if (onCallDone) onCallDone();
    return data.content || '';
  }

  async function onceDirect(messages, o) {
    const base = String(baseUrl).replace(/\/+$/, '');
    const models = String(model).split(',').map((s) => s.trim()).filter(Boolean);
    let last = null, sawNonQuota = false;
    for (const m of models) {
      const mkBody = () => {
        const b = { model: m, messages, max_tokens: o.maxTokens || 6000, temperature: o.temperature ?? 0.7 };
        if (o.json) b.response_format = { type: 'json_object' };
        if (/gemini/i.test(m)) b.reasoning_effort = 'none';
        return b;
      };
      const call = (body) => f(`${base}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify(body),
      });
      const body = mkBody();
      let r = await call(body);
      let data = await r.json().catch(() => ({}));
      if (!r.ok && body.response_format && r.status === 400 && /response_format|json/i.test(JSON.stringify(data))) {
        delete body.response_format;
        r = await call(body); data = await r.json().catch(() => ({}));
      }
      if (!r.ok && body.reasoning_effort && r.status === 400) {
        delete body.reasoning_effort;
        r = await call(body); data = await r.json().catch(() => ({}));
      }
      if (r.ok) {
        const content = data?.choices?.[0]?.message?.content ?? '';
        if (data.usage && onUsage) onUsage(data.usage);
        if (onCallDone) onCallDone();
        return content;
      }
      const msg = data?.error?.message || data?.message || `HTTP ${r.status}`;
      const err = new Error(msg);
      err.status = r.status;
      const rm = /try again in (?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?/i.exec(msg);
      if (rm) err.retryAfter = (+rm[1] || 0) * 3600 + (+rm[2] || 0) * 60 + (+rm[3] || 0) || null;
      const daily = /per day|\bTPD\b|daily|insufficient_quota|exceeded your current quota/i.test(msg);
      err.quota = daily;
      last = err;
      if (r.status === 401 || r.status === 403) { err.fatal = true; throw err; }
      const skippable = daily || r.status === 429 || r.status === 503 || r.status === 504 || r.status === 404;
      if (skippable) continue;              // try next model in the pool
      sawNonQuota = true;
      throw err;                            // unexpected error: let the outer retry handle it
    }
    if (last) {
      if (!last.quota && !sawNonQuota && (last.status === 429)) last.quota = true;
      if (last.quota) {
        last.fatal = true;
        last.message = `Your provider's rate or quota limit was reached${last.retryAfter ? ` (resets in about ${Math.ceil(last.retryAfter / 60)} min)` : ''}. Try again later, use another model, or upgrade the plan. Provider said: ${last.message}`;
      }
      throw last;
    }
    throw new Error('No model produced a response');
  }

  const llm = async (messages, o = {}) => {
    await acquire();
    try {
      for (let attempt = 0; ; attempt++) {
        if (isCancelled && isCancelled()) throw new Error('Cancelled');
        try { return await once(messages, o); } catch (e) {
          if (e.fatal || attempt >= 12) throw e;
          const wait = Math.max((e.retryAfter || 0) * 1000 + 300, 1500 * 2 ** attempt);
          if (onRetry) onRetry(e, wait);
          await sleep(Math.min(wait, 60000));
        }
      }
    } finally { release(); }
  };

  llm.json = async (messages, o = {}) => {
    for (let i = 0; ; i++) {
      const msgs = i === 0 ? messages : [...messages, { role: 'user', content: 'Your previous reply was not valid JSON. Return ONLY the raw JSON object now: no markdown fences, no commentary, no extra text.' }];
      const text = await llm(msgs, { ...o, json: true });
      try { if (!String(text || '').trim()) throw new Error('empty content'); return parseJSON(text); } catch (e) { if (i >= 3) throw new Error('Model returned invalid JSON after 4 attempts'); }
    }
  };
  return llm;
}

// ---------- Stage 1: world model ----------

export async function buildWorld(llm, { docs, scenario, outreach, rounds, language }) {
  const user = `SOURCE DOCUMENTS:
${clip(docs, 14000)}

SCENARIO QUESTION: ${scenario}
${outreach ? `\nOUTREACH MESSAGE TO TEST:\n${clip(outreach, 3000)}\n` : ''}
Build the world to simulate. Return JSON:
{
 "product": {"name": str, "summary": "2 sentences", "claims": [vendor claims verbatim-ish, max 6]},
 "segments": [{"name": str, "description": str, "pains": [str], "buying_role": "buyer|user|influencer|blocker", "share": number}],  // 3-6 segments; share = your estimate of this segment's fraction of the whole target audience (all shares sum to 1). It is an assumption, base it on the documents and say nothing about real market size
 "competitors": [str],   // incumbents named in the source or obviously implied (max 6)
 "tensions": [str],      // the 4-6 sharpest trade-offs buyers will weigh
 "events": [str]         // exactly ${rounds} short events, one per round, drawn from the source's scenario variables (pricing change, outage, competitor move, privacy news, launch...). Round 1 should be the pitch/outreach landing.
}${language ? `\nWrite all string values in ${language}.` : ''}`;
  const w = await llm.json([{ role: 'system', content: SYS }, { role: 'user', content: user }], { maxTokens: 3500, temperature: 0.4 });
  w.segments = (w.segments || []).slice(0, 6);
  const tot = w.segments.reduce((t, s) => t + (Number(s.share) > 0 ? Number(s.share) : 0), 0);
  w.segments.forEach((s) => { s.share = tot > 0 && Number(s.share) > 0 ? Number(s.share) / tot : 1 / w.segments.length; });
  w.events = (w.events || []).slice(0, rounds);
  while (w.events.length < rounds) w.events.push('Continued word of mouth and comparison shopping.');
  return w;
}

// ---------- Stage 2: agents ----------

// Split `count` panel seats across segments by audience share (largest remainder, at least one seat each).
export function allocate(segments, count) {
  const n = segments.length;
  const shares = segments.map((s) => Math.max(0.01, s.share || 1 / n));
  const tot = shares.reduce((a, b) => a + b, 0);
  const exact = shares.map((s) => (count * s) / tot);
  const seats = exact.map((x) => Math.max(count >= n ? 1 : 0, Math.floor(x)));
  let diff = count - seats.reduce((a, b) => a + b, 0);
  const order = exact.map((x, i) => [x - Math.floor(x), i]).sort((a, b) => b[0] - a[0]);
  for (let k = 0; diff > 0; k++, diff--) seats[order[k % n][1]]++;
  for (let k = 0; diff < 0; k++) { const i = seats.indexOf(Math.max(...seats)); if (seats[i] > 1) { seats[i]--; diff++; } else break; }
  return seats;
}

export async function buildAgents(llm, world, { count, language }) {
  const perBatch = 6;
  const seats = allocate(world.segments, count);
  const queues = world.segments.map((s, i) => Array(seats[i]).fill(s.name));
  const slots = [];
  while (slots.length < count) for (const q of queues) if (q.length) slots.push(q.shift());   // round-robin so batches mix segments
  const batches = [];
  for (let i = 0; i < count; i += perBatch) batches.push({ start: i, n: Math.min(perBatch, count - i), slots: slots.slice(i, i + perBatch) });
  const results = await Promise.all(batches.map(async ({ start, n, slots: sl }) => {
    const user = `PRODUCT: ${world.product?.name} — ${world.product?.summary}
SEGMENTS: ${JSON.stringify(world.segments)}
COMPETITORS: ${(world.competitors || []).join(', ')}
Create ${n} distinct simulated people with exactly these segment assignments: ${sl.map((s, i) => `a${start + i + 1} = ${s}`).join('; ')}. Within each segment mix enthusiasts, skeptics and neutral people, with a realistic spread of skepticism and price sensitivity.
Return JSON: {"agents":[{"id":"a${start + 1}","name":str,"segment":<segment name>,"role":"champion|blocker|decider|influencer|user","bio":"1 sentence","current_tools":[str],"pain":str,"skepticism":1-5,"price_sensitivity":"low|medium|high"}]}${language ? `\nWrite text values in ${language}.` : ''}`;
    const d = await llm.json([{ role: 'system', content: SYS }, { role: 'user', content: user }], { maxTokens: 3000, temperature: 0.9 });
    return pickList(d, ['agents', 'personas', 'people'], (x) => x && (x.name || x.segment || x.id || x.bio));
  }));
  const agents = results.flat().slice(0, count).map((a, i) => ({
    ...a,
    id: `a${i + 1}`,
    name: a.name || `Agent ${i + 1}`,
    segment: slots[i] || a.segment || world.segments?.[0]?.name || 'General',
    role: a.role || 'user',
    skepticism: Math.min(5, Math.max(1, Number(a.skepticism) || 3)),
    stance: 0,
  }));
  if (!agents.length) throw new Error('No agents were generated');
  return agents;
}

// ---------- Stage 3: simulation rounds ----------

const INTENTS = ['adopt', 'trial', 'wait', 'reject', 'churn', 'none'];
const OBJECTIONS = ['cost', 'quality', 'privacy', 'lockin', 'integration', 'trust', 'support', 'proof', 'other', 'none'];

async function runBatch(llm, world, round, event, feed, group, language) {
  const recent = feed.slice(-14).map((p) => `[${p.id}] ${p.name} (${p.segment}): ${p.text}`).join('\n') || '(no posts yet)';
  const people = group.map((a) => `- ${a.id} ${a.name}, ${a.segment}, ${a.role}. ${a.bio} Uses: ${(a.current_tools || []).join(', ') || 'n/a'}. Pain: ${a.pain}. Skepticism ${a.skepticism}/5, price sensitivity ${a.price_sensitivity}. Current stance ${a.stance} (-2 hostile .. +2 enthusiastic).`).join('\n');
  const user = `PRODUCT: ${world.product?.name} — ${world.product?.summary}
VENDOR CLAIMS (unverified): ${(world.product?.claims || []).join(' | ')}
COMPETITORS: ${(world.competitors || []).join(', ')}
TENSIONS: ${(world.tensions || []).join(' | ')}

ROUND ${round} EVENT: ${event}

RECENT FEED:
${recent}

PEOPLE ACTING THIS ROUND:
${people}

For each person decide how they react in character: post a take, reply to a feed post (reply_to = its id), or ignore. Stances move gradually (at most 1 step per round unless the event is dramatic) and should reflect their skepticism and pains. Keep text under 45 words, first person, no statistics.
Return JSON: {"actions":[{"agent_id":str,"action":"post|reply|ignore","text":str,"reply_to":str|null,"stance":-2..2,"intent":"${INTENTS.join('|')}","objection":"${OBJECTIONS.join('|')}"}]}${language ? `\nWrite text in ${language}.` : ''}`;
  const d = await llm.json([{ role: 'system', content: SYS }, { role: 'user', content: user }], { maxTokens: 2500, temperature: 0.9 });
  return pickList(d, ['actions', 'results', 'responses'], (x) => x && (x.agent_id || x.action || x.stance !== undefined)).map((x) => {
    if (!x.agent_id) x.agent_id = x.agent || x.id;
    return x;
  });
}

export async function runSimulation(llm, world, agents, { rounds, batch = DEFAULTS.batch, language, onRound, onPost }) {
  const feed = [];
  const history = [];
  for (let r = 1; r <= rounds; r++) {
    const event = world.events[r - 1];
    const groups = [];
    for (let i = 0; i < agents.length; i += batch) groups.push(agents.slice(i, i + batch));
    const out = await Promise.all(groups.map(async (g) => {
      try { return await runBatch(llm, world, r, event, feed, g, language); } catch (err) {
        if (err.fatal) throw err;                       // quota / auth problems must stop the run, not vanish
        try { return await runBatch(llm, world, r, event, feed, g, language); } catch (err2) { if (err2.fatal) throw err2; return []; }
      }
    }));
    if (!out.flat().length) throw new Error(`Round ${r}: the model returned no usable agent actions. Try a more capable model.`);
    const byId = Object.fromEntries(agents.map((a) => [a.id, a]));
    const roundPosts = [];
    for (const act of out.flat()) {
      const a = byId[act.agent_id];
      if (!a) continue;
      const stance = Math.max(-2, Math.min(2, Math.round(Number(act.stance))));
      if (Number.isFinite(stance)) a.stance = stance;
      a.intent = INTENTS.includes(act.intent) ? act.intent : 'none';
      a.objection = OBJECTIONS.includes(act.objection) ? act.objection : 'none';
      const spoke = act.action !== 'ignore' && act.text;
      const post = {
        id: `r${r}-${a.id}`, round: r, agent_id: a.id, name: a.name, segment: a.segment, role: a.role,
        action: spoke ? act.action : 'ignore', text: spoke ? String(act.text) : '', reply_to: act.reply_to || null,
        stance: a.stance, intent: a.intent, objection: a.objection,
      };
      roundPosts.push(post);
      if (spoke) { feed.push(post); if (onPost) onPost(post); }
    }
    // Snapshot every agent's stance for this round (silent agents carry their last stance).
    history.push({ round: r, event, snapshot: agents.map((a) => ({ id: a.id, segment: a.segment, role: a.role, stance: a.stance, intent: a.intent || 'none', objection: a.objection || 'none' })), posts: roundPosts });
    if (onRound) onRound(r, history[history.length - 1]);
  }
  return { feed, history };
}

// ---------- Stage 4: interviews ----------

export async function interviewAgents(llm, world, agents, feed, { language, max = 6 }) {
  const pick = [];
  const add = (a) => { if (a && !pick.includes(a)) pick.push(a); };
  const sorted = [...agents].sort((x, y) => y.stance - x.stance);
  add(sorted[0]); add(sorted[sorted.length - 1]);
  for (const role of ['decider', 'blocker', 'champion', 'influencer']) add(agents.find((a) => a.role === role));
  for (const a of agents) { if (pick.length >= max) break; add(a); }
  const chosen = pick.slice(0, max);

  const qs = ['Would you adopt this, and what is the single most decisive factor?', 'What is your biggest objection and what proof would remove it?', 'Who would you tell about it (or warn), and what would you say?'];
  return Promise.all(chosen.map(async (a) => {
    const mine = feed.filter((p) => p.agent_id === a.id).map((p) => p.text).join(' / ');
    const user = `You are ${a.name}, ${a.segment}, ${a.role}. ${a.bio} Pain: ${a.pain}. Current stance ${a.stance} (-2..+2). You said in the feed: ${clip(mine, 600) || '(nothing)'}
Product: ${world.product?.name} — ${world.product?.summary}
Answer as yourself, first person, 1-2 sentences each, no statistics.
Return JSON: {"answers":[str,str,str]} answering in order: ${qs.map((q, i) => `${i + 1}) ${q}`).join(' ')}${language ? `\nWrite in ${language}.` : ''}`;
    try {
      const d = await llm.json([{ role: 'system', content: SYS }, { role: 'user', content: user }], { maxTokens: 1500, temperature: 0.8 });
      return { agent: a, questions: qs, answers: d.answers || [] };
    } catch { return { agent: a, questions: qs, answers: [] }; }
  }));
}

// ---------- Stage 5: statistics (computed, not generated) ----------

export function computeStats(agents, history) {
  const rounds = history.map((h) => {
    const s = h.snapshot, n = s.length || 1;
    const pos = s.filter((x) => x.stance > 0).length, neg = s.filter((x) => x.stance < 0).length;
    const intents = {}, objections = {};
    for (const x of s) {
      intents[x.intent] = (intents[x.intent] || 0) + 1;
      if (x.objection !== 'none') objections[x.objection] = (objections[x.objection] || 0) + 1;
    }
    return { round: h.round, event: h.event, mean: s.reduce((t, x) => t + x.stance, 0) / n, pos, neu: s.length - pos - neg, neg, intents, objections, posts: h.posts.filter((p) => p.text).length };
  });
  const last = history[history.length - 1]?.snapshot || [];
  const seg = {};
  for (const x of last) { (seg[x.segment] ||= []).push(x.stance); }
  const segments = Object.entries(seg).map(([name, v]) => ({ name, n: v.length, mean: v.reduce((a, b) => a + b, 0) / v.length }));
  const role = {};
  for (const x of last) { (role[x.role] ||= []).push(x.stance); }
  const roles = Object.entries(role).map(([name, v]) => ({ name, n: v.length, mean: v.reduce((a, b) => a + b, 0) / v.length }));
  const totals = {};
  for (const r of rounds) for (const [k, v] of Object.entries(r.objections)) totals[k] = (totals[k] || 0) + v;
  const objections = Object.entries(totals).sort((a, b) => b[1] - a[1]).map(([name, count]) => ({ name, count }));
  const first = rounds[0]?.mean ?? 0, final = rounds[rounds.length - 1]?.mean ?? 0;
  return { n: agents.length, rounds, segments, roles, objections, first, final, delta: final - first, finalIntents: rounds[rounds.length - 1]?.intents || {} };
}

const f1 = (x) => (x >= 0 ? '+' : '') + x.toFixed(2);

export function statsMarkdown(st) {
  const r = ['| Round | Event | Positive | Neutral | Negative | Mean stance |', '|---|---|---|---|---|---|'];
  for (const x of st.rounds) r.push(`| ${x.round} | ${x.event} | ${x.pos} | ${x.neu} | ${x.neg} | ${f1(x.mean)} |`);
  const s = ['| Segment | Agents | Final mean stance |', '|---|---|---|', ...st.segments.map((x) => `| ${x.name} | ${x.n} | ${f1(x.mean)} |`)];
  const o = st.objections.length ? ['| Objection | Mentions |', '|---|---|', ...st.objections.map((x) => `| ${x.name} | ${x.count} |`)] : ['No objections were tagged.'];
  const i = Object.entries(st.finalIntents).map(([k, v]) => `${k}: ${v}`).join(', ');
  return `${r.join('\n')}\n\n${s.join('\n')}\n\n${o.join('\n')}\n\nFinal-round intent mix: ${i}`;
}

// ---------- Projection: from the AI panel to the whole population ----------

function wilson(k, n, z = 1.645) {
  if (!n) return [0, 1];
  const p = k / n, d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d, m = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, c - m), Math.min(1, c + m)];
}
const POSITIVE = ['adopt', 'trial'], RESIST = ['reject', 'churn'];

export function project(world, agents, population, calibration) {
  const full = population <= agents.length;
  const present = world.segments.filter((s) => agents.some((a) => a.segment === s.name));
  const totShare = present.reduce((t, s) => t + s.share, 0) || 1;
  const rows = present.map((s) => {
    const ag = agents.filter((a) => a.segment === s.name), k = ag.length;
    const pos = ag.filter((a) => POSITIVE.includes(a.intent)).length;
    const commit = ag.filter((a) => a.intent === 'adopt').length;
    const resist = ag.filter((a) => RESIST.includes(a.intent)).length;
    const [lo, hi] = full ? [pos / k, pos / k] : wilson(pos, k);
    const share = s.share / totShare;
    return { name: s.name, share, k, people: Math.round(population * share), rate: pos / k, lo, hi, commit: commit / k, resist: resist / k };
  });
  const sum = (f) => rows.reduce((t, r) => t + r.share * f(r), 0);
  const overall = { rate: sum((r) => r.rate), lo: sum((r) => r.lo), hi: sum((r) => r.hi), commit: sum((r) => r.commit), resist: sum((r) => r.resist) };
  let calibrated = null;
  if (calibration && calibration.ratio && calibration.n > 0) {
    const c = (x) => Math.min(1, Math.max(0, x * calibration.ratio));
    calibrated = { rate: c(overall.rate), lo: c(overall.lo), hi: c(overall.hi), ratio: calibration.ratio, n: calibration.n };
  }
  return { population, panel: agents.length, full, rows, overall, calibrated };
}

const pct = (x) => `${Math.round(x * 100)}%`;
const big = (n) => n.toLocaleString('en-US');

export function projectionMarkdown(pr) {
  const head = pr.full
    ? `All ${big(pr.population)} people were simulated individually, so these are direct counts.`
    : `${big(pr.panel)} AI personas stand in for ${big(pr.population)} people. Each persona represents the share of the audience its segment holds, so the figures below scale the panel up. Segment shares are assumptions taken from your documents.`;
  const rows = ['| Segment | Share of audience | Panel | People represented | Positive intent (adopt or trial) | Likely range |', '|---|---|---|---|---|---|',
    ...pr.rows.map((r) => `| ${r.name} | ${pct(r.share)} | ${r.k} | ${big(r.people)} | ${pct(r.rate)} | ${pr.full ? 'exact' : `${pct(r.lo)} to ${pct(r.hi)}`} |`)];
  const o = pr.overall;
  const line = `**Overall: about ${pct(o.rate)} of ${big(pr.population)} people (${big(Math.round(o.rate * pr.population))}) show positive intent${pr.full ? '' : `, likely between ${pct(o.lo)} and ${pct(o.hi)}`}.** About ${pct(o.commit)} said they would adopt outright; about ${pct(o.resist)} lean to reject.`;
  const cal = pr.calibrated ? `\n\n**Calibrated with your real-world results:** earlier real outcomes (${pr.calibrated.n} recorded) were on average ${pr.calibrated.ratio.toFixed(2)} times the simulated rate, so the adjusted estimate is about ${pct(pr.calibrated.rate)} (range ${pct(pr.calibrated.lo)} to ${pct(pr.calibrated.hi)}). Few data points means low confidence.` : '';
  const why = pr.full ? '' : '\n\nThe range reflects only how few personas stand for each segment. It does not cover model bias or wrong segment shares, so treat it as a guide, not a guarantee.';
  return `${head}\n\n${rows.join('\n')}\n\n${line}${cal}${why}`;
}

// ---------- Stage 6: report ----------

const CHAPTERS = [
  ['Predicted market reaction', 'What happened across the rounds: how stances moved, which events mattered, what the simulated community converged on. Cite the computed statistics and quote the feed.'],
  ['How each buyer type responds', 'Walk through the segments and roles. Include a Markdown table: persona/segment | role | what makes them buy | what makes them leave. Quote interviews.'],
  ['Objections, risks and competitive pressure', 'The objections that appeared (use the objection tallies), how people compare this to the named competitors, and which vendor claims are at risk. Include a Markdown table: objection | severity (low/medium/high) | best rebuttal or proof needed.'],
  ['Six-month outlook', 'Three scenarios (Bear / Base / Bull) as a Markdown table: scenario | assumptions | leading indicators. Qualitative only; ground them in how the simulated stances moved. State that these extrapolate a small simulated panel.'],
  ['Recommended actions', 'Five prioritised experiments (what to test, how to judge success qualitatively), plus one line on cross-sell if other offerings appear in the source. If an outreach message was supplied: what works, what hurts, 3 alternative subject lines, and a tighter rewrite under 120 words.'],
];

function context(world, agents, sim, interviews, stats, outreach, projection) {
  const quotes = [...sim.feed].sort((a, b) => Math.abs(b.stance) - Math.abs(a.stance)).slice(0, 28)
    .map((p) => `[r${p.round}] ${p.name} (${p.segment}, ${p.role}, stance ${p.stance}, objection ${p.objection}): "${p.text}"`).join('\n');
  const iv = interviews.map((x) => `${x.agent.name} (${x.agent.segment}, ${x.agent.role}, stance ${x.agent.stance}):\n` + x.answers.map((a, i) => `  Q${i + 1} ${x.questions[i]} -> "${a}"`).join('\n')).join('\n');
  return `PRODUCT: ${JSON.stringify(world.product)}
SEGMENTS: ${JSON.stringify(world.segments)}
COMPETITORS: ${(world.competitors || []).join(', ')}
TENSIONS: ${(world.tensions || []).join(' | ')}
SIMULATION STATISTICS (computed from ${stats.n} simulated agents; stance -2 hostile .. +2 enthusiastic):
${statsMarkdown(stats)}

PROJECTION TO THE WHOLE AUDIENCE:
${projection ? projectionMarkdown(projection) : 'n/a'}

FEED QUOTES:
${quotes}

INTERVIEWS:
${iv}
${outreach ? `\nOUTREACH MESSAGE:\n${clip(outreach, 3000)}` : ''}`;
}

export function keyFacts(st) {
  const last = st.rounds[st.rounds.length - 1], first = st.rounds[0];
  const segs = [...st.segments].sort((a, b) => b.mean - a.mean);
  const top = st.objections[0];
  const dir = st.delta > 0.25 ? 'warmed up' : st.delta < -0.25 ? 'cooled down' : 'stayed about the same';
  const pct = (n) => `${n} of ${st.n}`;
  const lines = [
    `- **Mood over time:** the simulated community ${dir} (average stance ${f1(st.first)} to ${f1(st.final)} on a scale from -2 to +2).`,
    `- **At the end:** ${pct(last.pos)} agents were positive, ${pct(last.neu)} neutral, ${pct(last.neg)} negative.`,
    segs.length ? `- **Most receptive group:** ${segs[0].name} (${f1(segs[0].mean)}). **Least receptive:** ${segs[segs.length - 1].name} (${f1(segs[segs.length - 1].mean)}).` : '',
    top ? `- **Most common objection:** ${top.name} (${top.count} mentions across all rounds).` : '',
    `- **What people said they would do:** ${Object.entries(st.finalIntents).filter(([k]) => k !== 'none').map(([k, v]) => `${k} ${v}`).join(', ') || 'no clear intent'}.`,
  ];
  return lines.filter(Boolean).join('\n');
}

function appendix(interviews, sim) {
  const iv = interviews.filter((x) => x.answers.length).map((x) => `**${x.agent.name}** — ${x.agent.segment}, ${x.agent.role} (final stance ${f1(x.agent.stance)})\n\n` + x.answers.map((a, i) => `> *${x.questions[i]}* "${a}"`).join('\n>\n')).join('\n\n');
  const quotes = [...sim.feed].sort((a, b) => Math.abs(b.stance) - Math.abs(a.stance)).slice(0, 8)
    .map((p) => `> "${p.text}" — ${p.name}, ${p.segment} (round ${p.round}, stance ${p.stance > 0 ? '+' : ''}${p.stance})`).join('\n\n');
  return `## Appendix A: Interviews with simulated buyers\n\n${iv || '_No interviews were completed._'}\n\n## Appendix B: Strongest posts from the simulated feed\n\n${quotes}`;
}

// Verdict comes from the simulated numbers, not from the model's mood.
export function verdictFor(st) {
  const last = st.rounds[st.rounds.length - 1];
  const negShare = last ? last.neg / st.n : 0;
  if (st.final >= 0.75 && negShare < 0.25) return 'Go';
  if (st.final >= 0 && negShare < 0.5) return 'Go with changes';
  return 'Hold';
}

export async function writeReport(llm, world, agents, sim, interviews, stats, { outreach, language, params, projection }) {
  const ctx = context(world, agents, sim, interviews, stats, outreach, projection);
  const rules = `Write ONE chapter body in Markdown. No headings of any kind (use **bold lead-ins** and lists). Quote simulated people as blockquotes on their own paragraph, attributed like: > "…" — Name, segment (simulated). Use only the statistics supplied above and quote them accurately; never invent figures. Write for a smart non-expert: short sentences, plain words, define any jargon in a few words. Max 330 words.${language ? ` Write in ${language}.` : ''}`;

  const chapterJobs = CHAPTERS.map(([title, brief]) =>
    {
      const ask = () => llm([{ role: 'system', content: SYS }, { role: 'user', content: `${ctx}\n\nCHAPTER: ${title}\nBRIEF: ${brief}\n\n${rules}` }], { maxTokens: 1800, temperature: 0.5 });
      return ask().catch((e) => { if (e.fatal) throw e; return ask(); })
        .catch((e) => { if (e.fatal) throw e; return `_This chapter could not be generated: ${e.message}_`; });
    });

  const headJob = llm.json([{ role: 'system', content: SYS }, { role: 'user', content: `${ctx}\n\nReturn JSON: {"title":"a sharp finding as report title (not just the product name)","summary":"one sentence core predicted finding","verdict_reason":"one sentence tied to the statistics"}${language ? ` Write in ${language}.` : ''}` }], { maxTokens: 1200, temperature: 0.3 })
    .catch(() => ({ title: `${world.product?.name} — Simulation Report`, summary: '', verdict_reason: '' }));

  const summaryJob = llm.json([{ role: 'system', content: SYS }, { role: 'user', content: `${ctx}\n\nWrite the executive summary for a busy manager who will not read the rest. Plain, simple language, no jargon. Return JSON: {"bottom_line":"2-3 sentences: what the simulation suggests about this product","who_will_buy":["max 3 short items"],"who_will_resist":["max 3 short items, with the reason"],"biggest_risk":"one sentence","do_first":["exactly 3 concrete next steps"]}. No invented numbers.${language ? ` Write in ${language}.` : ''}` }], { maxTokens: 1800, temperature: 0.3 })
    .catch((e) => { if (e.fatal) throw e; return null; });

  const [head, summary, ...chapters] = await Promise.all([headJob, summaryJob, ...chapterJobs]);
  head.verdict = verdictFor(stats);
  const date = new Date().toISOString().slice(0, 10);
  const md = [
    `# ${head.title}`,
    head.summary ? `> ${head.summary}` : '',
    `**Product:** ${world.product?.name} | **Verdict:** ${head.verdict} | **Population:** ${big(projection?.population || stats.n)} | **AI panel:** ${stats.n} | **Rounds:** ${params.rounds} | **Generated:** ${date}`,
    head.verdict_reason ? `*${head.verdict_reason}*` : '',
    '## At a glance',
    summary ? `**Bottom line.** ${summary.bottom_line}` : '',
    summary ? `**Who is likely to buy**\n\n${(summary.who_will_buy || []).map((x) => `- ${x}`).join('\n')}` : '',
    summary ? `**Who is likely to resist**\n\n${(summary.who_will_resist || []).map((x) => `- ${x}`).join('\n')}` : '',
    summary ? `**Biggest risk.** ${summary.biggest_risk}` : '',
    summary ? `**Do these first**\n\n${(summary.do_first || []).map((x, i) => `${i + 1}. ${x}`).join('\n')}` : '',
    '## The numbers',
    'These figures are counted from the simulation itself (not written by the AI).',
    keyFacts(stats),
    statsMarkdown(stats),
    projection ? `## Projection to ${big(projection.population)} people` : '',
    projection ? projectionMarkdown(projection) : '',
    ...CHAPTERS.flatMap(([title], i) => [`## Chapter ${i + 1}: ${title}`, chapters[i]]),
    appendix(interviews, sim),
    '## How to read this report',
    '- **Stance** is how positive a simulated person feels about the product: -2 hostile, 0 neutral, +2 enthusiastic.\n- **Agents** are AI-generated people built from your documents (buyers, users, skeptics, decision-makers).\n- **Rounds** are time steps. Each round introduces an event (for example a price change or an outage) and agents react to it and to each other.\n- **Verdict** is calculated from the final stances: Go, Go with changes, or Hold.',
    '## Method and limits',
    `This report comes from a small social simulation: ${stats.n} AI-generated personas reacting over ${params.rounds} rounds to events drawn from your documents, then interviewed. Statistics describe that simulated panel only. They are scenarios to test, not market data, forecasts or proof of demand. Vendor claims in the source are treated as unverified claims.`,
  ].filter(Boolean).join('\n\n');
  return { markdown: md, head };
}

// ---------- Orchestration ----------

export async function runAll(llm, input, hooks = {}) {
  const p = { ...DEFAULTS, ...(input.params || {}) };
  p.population = Math.max(p.agents, Math.min(1000000, Math.round(Number(p.population) || p.agents)));
  if (!(input.params && input.params.batch)) p.batch = p.agents > 40 ? 6 : 4;
  const stage = (name, detail = '') => hooks.onStage && hooks.onStage(name, detail);
  const t0 = Date.now();

  stage('world', 'Reading your documents and building the world');
  const world = await buildWorld(llm, { docs: input.docs, scenario: input.scenario, outreach: input.outreach, rounds: p.rounds, language: input.language });
  hooks.onWorld && hooks.onWorld(world);

  stage('agents', `Creating ${p.agents} AI personas to represent ${p.population.toLocaleString('en-US')} people`);
  const agents = await buildAgents(llm, world, { count: p.agents, language: input.language });
  hooks.onAgents && hooks.onAgents(agents);

  stage('simulate', `Running ${p.rounds} rounds`);
  const sim = await runSimulation(llm, world, agents, { rounds: p.rounds, batch: p.batch, language: input.language, onRound: (r, h) => hooks.onRound && hooks.onRound(r, h), onPost: hooks.onPost });

  stage('interview', 'Interviewing key agents');
  const interviews = await interviewAgents(llm, world, agents, sim.feed, { language: input.language });
  hooks.onInterviews && hooks.onInterviews(interviews);

  const stats = computeStats(agents, sim.history);
  hooks.onStats && hooks.onStats(stats);

  stage('report', 'Writing the report');
  const projection = project(world, agents, p.population, input.calibration);
  hooks.onProjection && hooks.onProjection(projection);
  const report = await writeReport(llm, world, agents, sim, interviews, stats, { outreach: input.outreach, language: input.language, params: p, projection });

  stage('done', `Finished in ${Math.round((Date.now() - t0) / 1000)}s`);
  return { world, agents, sim, interviews, stats, projection, report, seconds: Math.round((Date.now() - t0) / 1000) };
}
