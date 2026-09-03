/* SENTINEL risk register — vanilla JS, hand-rolled SVG, no CDN. */
let D, CUR = null;
const $ = s => document.querySelector(s);
const BANDS = ['CRITICAL', 'HIGH', 'WATCH', 'HEALTHY'];
const C = { CRITICAL: 'var(--critical)', HIGH: 'var(--high)', WATCH: 'var(--watch)', HEALTHY: 'var(--healthy)' };
const VIEWS = [['portfolio', 'Portfolio'], ['watchlist', 'Watchlist'], ['merchant', 'Merchant file'],
['evidence', 'Evidence'], ['network', 'Network'], ['categories', 'Categories']];
const NOTES = {
  portfolio: 'Every merchant financed on the book, ranked by SENTINEL score. Colour is used only to carry risk.',
  watchlist: 'The review queue. Sort any column; click a row to open that merchant\u2019s file.',
  merchant: 'One merchant, its trajectory, the evidence behind its score and the binding financing action.',
  evidence: 'Whether the model earns its place \u2014 tested against two baselines under purged walk-forward.',
  network: 'Merchants sharing an abnormal number of customers. The fake-transaction ring signal.',
  categories: 'Where risk concentrates by product category and delivery region.'
};

const pct = (v, d = 1) => v == null ? '\u2014' : (v * 100).toFixed(d) + '%';
const brl = v => v == null ? '\u2014' : 'R$ ' + Math.round(v).toLocaleString('en-US');
const num = (v, d = 2) => v == null ? '\u2014' : (+v).toFixed(d);
const esc = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const md = s => esc(s).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  .split('\n\n').map(p => `<p>${p}</p>`).join('');

const TIP = $('#tip');
function tip(e, txt) { TIP.textContent = txt; TIP.classList.add('on'); move(e); }
function move(e) {
  const r = TIP.getBoundingClientRect();
  TIP.style.left = Math.min(e.clientX + 12, innerWidth - r.width - 8) + 'px';
  TIP.style.top = Math.max(e.clientY - r.height - 10, 8) + 'px';
}
const untip = () => TIP.classList.remove('on');

/* ------------------------------------------------------------ SVG utils */
function svgEl(w, h, cls = '') {
  return `<svg viewBox="0 0 ${w} ${h}" class="${cls}" preserveAspectRatio="none" style="width:100%;height:${h}px">`;
}
function path(pts, cls, extra = '') {
  if (!pts.length) return '';
  const d = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
  const len = pts.length * 9;
  return `<path d="${d}" class="${cls} draw" style="--len:${len}" ${extra}/>`;
}

/* --------------------------------------------------------------- tape */
function tape() {
  const ms = D.merchants, W = 1000, H = 56, gap = 1;
  const bw = Math.max((W - ms.length * gap) / ms.length, .8);
  let s = svgEl(W, H, 'tape') + `<line x1="0" y1="${H - 10}" x2="${W}" y2="${H - 10}" stroke="var(--rule)"/>`;
  ms.forEach((m, i) => {
    const h = 6 + (m.score || 0) * 38, x = i * (bw + gap);
    s += `<rect x="${x.toFixed(2)}" y="${(H - 10 - h).toFixed(1)}" width="${bw.toFixed(2)}" height="${h.toFixed(1)}"
      fill="${C[m.band]}" data-id="${m.id}" data-i="${i}"/>`;
  });
  s += '</svg>';
  return `<div class="tape-wrap">
    <div class="eyebrow">The book \u2014 ${ms.length} financed merchants, tallest is riskiest</div>
    ${s}
    <div class="tape-legend">${BANDS.map(b =>
    `<span><i class="swatch" style="background:${C[b]}"></i>${b.toLowerCase()} \u00b7 ${D.bands[b] || 0}</span>`).join('')}
      <span style="margin-left:auto">click any bar to open its file</span>
    </div></div>`;
}
function wireTape(root) {
  root.querySelectorAll('.tape rect').forEach(r => {
    const m = D.merchants[+r.dataset.i];
    r.onmouseenter = e => tip(e, `${m.short}\u2026\nscore ${num(m.score)}  ${m.band}\nbad rate ${pct(m.bad_rate)}  ${m.n_orders} orders\nexposure ${brl(m.exposure)}`);
    r.onmousemove = move; r.onmouseleave = untip;
    r.onclick = () => { untip(); open(m.id); };
  });
}

/* ---------------------------------------------------------- portfolio */
function portfolio() {
  const M = D.metrics, ms = D.merchants;
  const atRisk = ms.filter(m => m.band === 'CRITICAL' || m.band === 'HIGH');
  const expAtRisk = atRisk.reduce((a, m) => a + (m.exposure || 0), 0);
  const kpi = (l, v, s, cls = '') => `<div class="kpi ${cls}"><div class="lab">${l}</div>
    <div class="val">${v}</div><div class="sub">${s}</div></div>`;

  const T = D.trend, tw = 1000, th = 150, mx = Math.max(...T.map(t => t.bad));
  const tpts = T.map((t, i) => [i / (T.length - 1) * (tw - 40) + 30, th - 24 - t.bad / mx * (th - 44)]);
  const trend = svgEl(tw, th) +
    `<line x1="30" y1="${th - 24}" x2="${tw - 10}" y2="${th - 24}" stroke="var(--rule)"/>` +
    T.map((t, i) => i % 3 ? '' : `<text x="${tpts[i][0]}" y="${th - 8}" class="lab" text-anchor="middle"
      style="font-family:PlexMono;font-size:9px;fill:var(--muted)">${t.d.slice(2)}</text>`).join('') +
    path(tpts, 'trace') +
    tpts.map((p, i) => `<circle cx="${p[0]}" cy="${p[1]}" r="2.2" fill="var(--ink)" class="fade"/>`).join('') +
    `<text x="30" y="14" style="font-family:PlexMono;font-size:10px;fill:var(--muted)">${pct(mx)} peak</text></svg>`;

  const maxE = Math.max(...BANDS.map(b => D.band_exposure[b] || 0));
  const bars = BANDS.map(b => {
    const n = D.bands[b] || 0, e = D.band_exposure[b] || 0;
    return `<div class="bar-row"><span class="nm">${b.toLowerCase()}</span>
      <span class="bar" style="width:${Math.max(e / maxE * 190, 2)}px;background:${C[b]}"></span>
      <span class="vv">${brl(e)} \u00b7 ${n}</span></div>`;
  }).join('');

  const queue = ms.slice(0, 8).map(m => `<tr data-id="${m.id}">
    <td class="mono">${m.short}\u2026</td><td><span class="pill ${m.band}">${m.band}</span></td>
    <td class="n">${num(m.score)}</td><td class="n">${pct(m.bad_rate)}</td>
    <td class="n">${brl(m.exposure)}</td><td class="n">${brl(m.limit)}</td></tr>`).join('');

  return tape() + `<div class="kpis">
    ${kpi('Merchants scored', ms.length, 'financed, 90-day activity')}
    ${kpi('Critical + high', atRisk.length, 'need action this week', 'crit')}
    ${kpi('Exposure at risk', brl(expAtRisk), 'financed value in those merchants')}
    ${kpi('Median warning', M.median_lead_days + ' days', 'before deterioration', 'pos')}
    ${kpi('Caught early', pct(M.pct_caught_early, 0), `${M.ew_caught_early} of ${M.ew_eligible} that deteriorated`, 'pos')}
  </div>
  <div class="grid3">
    <div class="card"><h3>Portfolio bad-order rate</h3>
      <div class="hint">Share of orders cancelled, delivered late, or rated 1\u20132 stars.</div>${trend}</div>
    <div class="card"><h3>Exposure by band</h3>
      <div class="hint">Where the money sits, not where the merchants sit.</div>${bars}
      <div class="caption">Bands are positions in the book: critical is the top 5%, high the next 10%,
      watch the next 15%. Review capacity is fixed, so the queue should be too.</div></div>
  </div>
  <div class="card" style="margin-top:var(--s5)"><h3>Action queue</h3>
    <div class="hint">Highest-scoring merchants. Click a row for the full file.</div>
    <table><thead><tr><th>Merchant</th><th>Band</th><th>Score</th><th>Bad rate</th>
    <th>Exposure</th><th>Revised limit</th></tr></thead><tbody>${queue}</tbody></table></div>`;
}

/* ---------------------------------------------------------- watchlist */
let sortKey = 'score', sortDir = -1;
function watchlist() {
  const cols = [['short', 'Merchant', 0], ['band', 'Band', 0], ['score', 'Score', 1],
  ['bad_rate', 'Bad rate', 1], ['exp_bad', 'Expected', 1], ['late_rate', 'Late', 1],
  ['mean_review', 'Review', 1], ['n_orders', 'Orders', 1],
  ['exposure', 'Exposure', 1], ['limit', 'New limit', 1], ['tenure_cap', 'Tenor', 1]];
  const ms = [...D.merchants].sort((a, b) => {
    const x = a[sortKey], y = b[sortKey];
    return (typeof x === 'string' ? x.localeCompare(y) : (x || 0) - (y || 0)) * sortDir;
  });
  const fmt = { score: v => num(v), bad_rate: pct, exp_bad: pct, late_rate: pct, mean_review: v => num(v, 1), exposure: brl, limit: brl };
  const rows = ms.map(m => `<tr data-id="${m.id}">` + cols.map(([k, , n]) => {
    if (k === 'band') return `<td><span class="pill ${m.band}">${m.band}</span></td>`;
    if (k === 'short') return `<td class="mono">${m.short}\u2026</td>`;
    return `<td class="${n ? 'n' : ''}">${(fmt[k] || (v => v))(m[k])}</td>`;
  }).join('') + '</tr>').join('');
  return `<div class="card"><h3>Review queue \u00b7 ${ms.length} merchants</h3>
  <div class="hint">\u201cExpected\u201d is the bad rate this merchant\u2019s own product and region mix would
  predict. A merchant far above its own expectation is the real signal \u2014 not one that simply
  sells difficult categories.</div>
  <div class="scroll"><table><thead><tr>${cols.map(([k, l, n]) =>
    `<th data-k="${k}" style="${n ? 'text-align:right' : ''}">${l}${sortKey === k ? (sortDir < 0 ? ' \u25be' : ' \u25b4') : ''}</th>`).join('')}
  </tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

/* ------------------------------------------- merchant file + THE RIBBON */
function ribbon(m) {
  const W = 1000, H = 190, L = 42, R = 14, T = 16, B = 30;
  const n = D.snapshots.length;
  const X = i => L + i / (n - 1) * (W - L - R);
  const Y = v => T + (1 - v) * (H - T - B);
  let s = svgEl(W, H, 'ribbon');
  [[.95, 'critical'], [.85, 'high'], [.70, 'watch']].forEach(([v, lab]) => {
    s += `<line x1="${L}" y1="${Y(v)}" x2="${W - R}" y2="${Y(v)}" class="band-line"/>
      <text x="4" y="${Y(v) + 3}" class="lab">${lab}</text>`;
  });
  if (m.first_flag != null && m.first_bad != null && m.first_flag < m.first_bad)
    s += `<rect class="lead-band" x="${X(m.first_flag)}" y="${T}"
      width="${X(m.first_bad) - X(m.first_flag)}" height="${H - T - B}"/>`;
  s += path(m.traj.map(p => [X(p.t), Y(p.s)]), 'trace');
  m.traj.forEach(p => {
    if (p.y === 1) s += `<circle cx="${X(p.t)}" cy="${Y(p.s)}" r="3" fill="var(--critical)" class="fade"/>`;
  });
  if (m.first_flag != null) s += `<circle cx="${X(m.first_flag)}" cy="${Y(m.traj.find(p => p.t === m.first_flag).s)}"
    r="5" fill="none" stroke="var(--high)" stroke-width="2" class="fade"/>`;
  const t0 = m.traj[0], t1 = m.traj[m.traj.length - 1];
  s += `<text x="${X(t0.t)}" y="${H - 10}" class="lab">${t0.d}</text>
    <text x="${W - R}" y="${H - 10}" class="lab" text-anchor="end">${t1.d}</text></svg>`;
  const verdict = m.lead
    ? `SENTINEL flagged this merchant <strong>${m.lead} days</strong> before it deteriorated.
       The shaded band is that warning window \u2014 time the risk team had to cap exposure,
       shorten tenor, or hold back settlement.`
    : m.first_bad != null
      ? `This merchant deteriorated without an advance flag. Shown because honest evidence
         includes the misses \u2014 the model catches ${pct(D.metrics.pct_caught_early, 0)} of cases, not all.`
      : `No deterioration recorded in the observation window. The trace is the merchant\u2019s
         risk score across every fortnightly snapshot.`;
  return `<div class="ribbon-card">
    <div class="eyebrow">Lead-time trace \u00b7 hollow ring = first flag \u00b7 filled dot = deterioration</div>
    ${s}<div class="caption">${verdict}</div></div>`;
}

function merchant() {
  const m = CUR || D.merchants[0];
  const opts = D.merchants.map(x =>
    `<option value="${x.id}" ${x.id === m.id ? 'selected' : ''}>${x.short}\u2026 \u00b7 ${x.band} \u00b7 ${num(x.score)}</option>`).join('');
  const mx = Math.max(...m.drivers.map(d => Math.abs(d.v))) || 1;
  const drivers = m.drivers.map(d => `<div class="bar-row"><span class="nm">${d.f}</span>
    <span class="bar ${d.v > 0 ? 'up' : 'dn'}" style="width:${Math.abs(d.v) / mx * 150}px"></span>
    <span class="vv">${d.v > 0 ? '+' : ''}${num(d.v, 3)}</span></div>`).join('');
  const cxs = Object.entries(m.cx).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]);
  const cx = cxs.length ? cxs.map(([k, v]) => `<div class="bar-row"><span class="nm">${k.replace(/_/g, ' ')}</span>
    <span class="bar" style="width:${Math.min(v * 420, 150)}px"></span>
    <span class="vv">${pct(v, 0)}</span></div>`).join('') : '<div class="caption">No complaint keywords found in this merchant\u2019s reviews.</div>';
  const g = (l, v) => `<div><div class="l">${l}</div><div class="v">${v}</div></div>`;
  return `<div class="card" style="margin-bottom:var(--s5)">
    <div class="eyebrow">Open file</div>
    <select id="msel" style="width:100%;max-width:420px">${opts}</select></div>
  ${ribbon(m)}
  <div class="action ${m.band}">
    <div class="eyebrow" style="margin:0">Binding action \u00b7 ${m.band}</div>
    <div style="font-size:14px;margin-top:4px">${esc(m.rec)}</div>
    <div class="row">
      <div>Current exposure<b>${brl(m.exposure)}</b></div>
      <div>Revised limit<b>${brl(m.limit)}</b></div>
      <div>Max tenor<b>${m.tenure_cap} mo</b></div>
      <div>Holdback<b>${pct(m.holdback, 0)}</b></div>
      <div>Score<b>${num(m.score)}</b></div>
    </div></div>
  <div class="metrics-grid" style="margin:var(--s5) 0">
    ${g('Bad rate', pct(m.bad_rate))}${g('Expected', pct(m.exp_bad))}
    ${g('Late', pct(m.late_rate))}${g('Cancelled', pct(m.cancel_rate))}
    ${g('1\u20132 star', pct(m.badreview_rate))}${g('Avg review', num(m.mean_review, 1))}
    ${g('Orders 90d', m.n_orders)}${g('Products', m.n_products)}
    ${g('Tenure', m.tenure ? m.tenure + 'd' : '\u2014')}</div>
  <div class="grid2">
    <div class="card"><h3>Risk memo</h3>
      <div class="memo">${md(m.memo)}</div>
      <div class="caption">Written from model drivers and peer benchmarks. Deterministic, so every
      figure traces to the data \u2014 it cannot invent a number.</div></div>
    <div>
      <div class="card"><h3>Score drivers</h3>
        <div class="hint">SHAP contributions. Red raises risk, green lowers it.</div>${drivers}</div>
      <div class="card" style="margin-top:var(--s5)"><h3>Complaint mix</h3>
        <div class="hint">Portuguese review text, keyword taxonomy.</div>${cx}</div>
    </div></div>`;
}

/* ----------------------------------------------------------- evidence */
function evidence() {
  const M = D.metrics, W = D.walkforward;
  const w = 1000, h = 240, L = 46, R = 12, T = 18, B = 40;
  const series = [['sentinel', 'SENTINEL', 'var(--accent)', 2.2], ['eb', 'Shrunk rate only', 'var(--watch)', 1.3],
  ['naive', 'Raw bad rate', 'var(--high)', 1.3], ['model', 'Model only', 'var(--faint)', 1.3]];
  const all = W.flatMap(f => series.map(s => f[s[0]])).filter(v => v != null);
  const lo = Math.min(...all) - .02, hi = Math.max(...all) + .02;
  const X = i => L + i / (W.length - 1) * (w - L - R);
  const Y = v => T + (1 - (v - lo) / (hi - lo)) * (h - T - B);
  let g = svgEl(w, h);
  [lo, (lo + hi) / 2, hi].forEach(v => g += `<line x1="${L}" y1="${Y(v)}" x2="${w - R}" y2="${Y(v)}"
    stroke="var(--rule-soft)"/><text x="6" y="${Y(v) + 3}" style="font-family:PlexMono;font-size:9.5px;fill:var(--muted)">${v.toFixed(2)}</text>`);
  series.forEach(([k, , col, sw]) => {
    const pts = W.map((f, i) => [X(i), Y(f[k])]);
    g += `<path d="${pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ')}"
      fill="none" stroke="${col}" stroke-width="${sw}" class="draw" style="--len:${pts.length * 12}"/>`;
  });
  W.forEach((f, i) => { if (i % 4 === 0) g += `<text x="${X(i)}" y="${h - 14}" text-anchor="middle" style="font-family:PlexMono;font-size:9px;fill:var(--muted)">${f.fold.slice(2, 7)}</text>`; });
  g += '</svg>';
  const leg = series.map(([, l, c]) => `<span><i class="swatch" style="background:${c}"></i>${l}</span>`).join('');

  const cmp = [['SENTINEL (0.30 model + 0.70 shrunk rate)', M.wf_sentinel_auc, M.wf_sentinel_lift, true],
  ['Empirical-Bayes shrunk rate', M.wf_eb_auc, null, false],
  ['Raw bad rate (naive persistence)', M.wf_naive_auc, M.wf_naive_lift, false],
  ['Gradient-boosted model alone', M.wf_model_only_auc, null, false]]
    .map(([n, a, l, b]) => `<tr><td>${b ? '<strong>' + n + '</strong>' : n}</td>
      <td class="n">${num(a, 4)}</td><td class="n">${l ? num(l) + '\u00d7' : '\u2014'}</td></tr>`).join('');

  const imp = D.importance.slice(0, 12);
  const mi = Math.max(...imp.map(x => x.v));
  const impb = imp.map(x => `<div class="bar-row"><span class="nm">${x.f}</span>
    <span class="bar" style="width:${x.v / mi * 160}px"></span><span class="vv">${x.v}</span></div>`).join('');

  return `<div class="verdict">
    <strong>The question that decides this project:</strong> why not just rank merchants by their
    current bad rate? We tested exactly that. Under purged walk-forward across
    ${M.wf_folds} folds, SENTINEL scores <strong>${num(M.wf_sentinel_auc, 3)}</strong> against
    <strong>${num(M.wf_naive_auc, 3)}</strong> for raw persistence, winning
    <strong>${M.wf_auc_wins} of ${M.wf_folds}</strong> folds at <strong>p = ${num(M.wf_pvalue, 4)}</strong>.
    Note the fourth line below: the machine-learning model <em>alone</em> is the worst scorer of the
    four. Its value is as a 30% correction on top of a volume-shrunk rate, not as a replacement for it.
  </div>
  <div class="kpis">
    <div class="kpi pos"><div class="lab">SENTINEL AUC</div><div class="val">${num(M.wf_sentinel_auc, 3)}</div>
      <div class="sub">\u00b1${num(M.wf_sentinel_auc_std, 3)} across folds</div></div>
    <div class="kpi"><div class="lab">Naive baseline</div><div class="val">${num(M.wf_naive_auc, 3)}</div>
      <div class="sub">rank by raw bad rate</div></div>
    <div class="kpi"><div class="lab">Folds won</div><div class="val">${M.wf_auc_wins}/${M.wf_folds}</div>
      <div class="sub">expanding window, purged</div></div>
    <div class="kpi pos"><div class="lab">Significance</div><div class="val">${num(M.wf_pvalue, 4)}</div>
      <div class="sub">paired t-test vs naive</div></div>
  </div>
  <div class="card"><h3>Every fold, every scorer</h3>
    <div class="hint">Training snapshots whose outcome window overlaps the test window are removed.
    Without that purge the model looks far better than it is.</div>
    ${g}<div class="tape-legend">${leg}</div></div>
  <div class="grid2" style="margin-top:var(--s5)">
    <div class="card"><h3>Scorer comparison</h3>
      <table><thead><tr><th>Scorer</th><th style="text-align:right">Mean AUC</th>
      <th style="text-align:right">Lift @20%</th></tr></thead><tbody>${cmp}</tbody></table>
      <div class="caption">Shrinkage matters because a merchant with six orders at a 33% bad rate is
      mostly noise. Each rate is pulled toward what that merchant\u2019s own product and region mix
      predicts, in proportion to how little volume backs it.</div></div>
    <div class="card"><h3>Business case</h3>
      <div class="slider-row"><span>Consumer-durable book</span><span class="mono" id="pf-l">\u20b9500 Cr</span></div>
      <input type="range" id="pf" min="100" max="2000" step="50" value="500">
      <div class="slider-row" style="margin-top:12px"><span>Share of flagged loss actually avoided</span>
        <span class="mono" id="ef-l">35%</span></div>
      <input type="range" id="ef" min="10" max="60" step="5" value="35">
      <div class="metrics-grid" style="margin-top:var(--s4)">
        <div><div class="l">Gross bad-order loss</div><div class="v" id="roi-g">\u2014</div></div>
        <div><div class="l">Model-captured</div><div class="v" id="roi-a">\u2014</div></div>
        <div><div class="l">Annual saving</div><div class="v" id="roi-s">\u2014</div></div>
      </div>
      <div class="caption">Assumes the observed ${pct(M.bad_order_rate)} bad-order rate and the model\u2019s
      measured ${pct(M.pct_loss_captured, 0)} capture of financed value in bad orders.</div></div>
  </div>
  <div class="grid2" style="margin-top:var(--s5)">
    <div class="card"><h3>What drives the model</h3>${impb}</div>
    <div class="card"><h3>What this cannot do</h3>
      <div class="memo" style="font-size:12.5px">
      <p><strong>The label is constructed, not ground truth.</strong> Olist carries no fraud flag.
      A bad order is one cancelled, delivered more than five days late, or rated 1\u20132 stars.
      We do not claim it equals return fraud.</p>
      <p><strong>AUC near 0.66 is the ceiling here.</strong> We tested order-level models,
      autoregressive lags, five label definitions and an exposure-weighted target that scored 0.855
      \u2014 and rejected it, because merchant size alone scored 0.858 on it. A model that loses to
      one column is not a model.</p>
      <p><strong>Brazilian proxy data.</strong> Chosen as the only public source carrying merchant,
      fulfilment, review and EMI-installment fields in one relational schema. Field mapping to an
      Indian LMS is in the README.</p>
      <p><strong>No leakage.</strong> Every feature comes from a strictly past window, the split is
      temporal, and overlapping outcome windows are purged from training.</p></div></div>
  </div>`;
}
function wireROI() {
  const M = D.metrics, pf = $('#pf'), ef = $('#ef');
  if (!pf) return;
  const upd = () => {
    const p = +pf.value, e = +ef.value / 100;
    const gross = p * M.bad_order_rate, addr = gross * M.pct_loss_captured;
    $('#pf-l').textContent = '\u20b9' + p + ' Cr'; $('#ef-l').textContent = ef.value + '%';
    $('#roi-g').textContent = '\u20b9' + gross.toFixed(1) + ' Cr';
    $('#roi-a').textContent = '\u20b9' + addr.toFixed(1) + ' Cr';
    $('#roi-s').textContent = '\u20b9' + (addr * e).toFixed(2) + ' Cr';
  };
  pf.oninput = ef.oninput = upd; upd();
}

/* ------------------------------------------------------------ network */
function network() {
  const N = D.graph.nodes, E = D.graph.edges, W = 1000, H = 540;
  const cx = W / 2, cy = H / 2, r = 215;
  const pos = {};
  N.forEach((n, i) => {
    const a = i / N.length * Math.PI * 2 - Math.PI / 2;
    pos[n.id] = [cx + Math.cos(a) * r, cy + Math.sin(a) * r];
  });
  const mw = Math.max(...E.map(e => e.w));
  let s = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:${H}px">`;
  E.forEach(e => {
    const a = pos[e.a], b = pos[e.b]; if (!a || !b) return;
    s += `<path d="M${a[0]} ${a[1]} Q ${cx} ${cy} ${b[0]} ${b[1]}" fill="none"
      stroke="var(--ink)" stroke-opacity="${(.10 + e.w / mw * .40).toFixed(2)}"
      stroke-width="${(.6 + e.w / mw * 2).toFixed(1)}"/>`;
  });
  N.forEach(n => {
    const p = pos[n.id];
    const col = n.risk >= .95 ? C.CRITICAL : n.risk >= .85 ? C.HIGH : n.risk >= .70 ? C.WATCH : C.HEALTHY;
    s += `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="6" fill="${col}"
      stroke="var(--panel)" stroke-width="1.5" data-id="${n.id}" class="gnode" style="cursor:pointer"/>`;
  });
  s += `<text x="${cx}" y="${cy - 6}" text-anchor="middle" style="font-family:PlexCond;font-weight:700;font-size:30px;fill:var(--ink)">${E.length}</text>
    <text x="${cx}" y="${cy + 14}" text-anchor="middle" style="font-family:Plex;font-size:11px;fill:var(--muted)">merchant pairs sharing customers</text></svg>`;
  const rows = E.slice(0, 18).map(e => `<tr><td class="mono">${e.a.slice(0, 10)}\u2026</td>
    <td class="mono">${e.b.slice(0, 10)}\u2026</td><td class="n">${e.w}</td></tr>`).join('');
  return `<div class="card"><h3>Shared-customer network</h3>
    <div class="hint">Two merchants selling to the same handful of customers, repeatedly, is what a
    fake-transaction ring looks like from the lender\u2019s side. Node colour is that merchant\u2019s
    risk band; edge weight is the number of customers in common.</div>${s}</div>
  <div class="card" style="margin-top:var(--s5)"><h3>Strongest links</h3>
    <table><thead><tr><th>Merchant A</th><th>Merchant B</th>
    <th style="text-align:right">Shared customers</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

/* --------------------------------------------------------- categories */
function categories() {
  const K = D.categories, W = 1000, H = 340, L = 48, R = 16, T = 18, B = 40;
  const xs = K.map(k => Math.log10(k.orders)), ys = K.map(k => k.bad);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y1 = Math.max(...ys);
  const X = v => L + (Math.log10(v) - x0) / (x1 - x0) * (W - L - R);
  const Y = v => T + (1 - v / y1) * (H - T - B);
  const mf = Math.max(...K.map(k => k.financed));
  let s = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:${H}px">`;
  [0, y1 / 2, y1].forEach(v => s += `<line x1="${L}" y1="${Y(v)}" x2="${W - R}" y2="${Y(v)}" stroke="var(--rule-soft)"/>
    <text x="6" y="${Y(v) + 3}" style="font-family:PlexMono;font-size:9.5px;fill:var(--muted)">${pct(v, 0)}</text>`);
  K.forEach(k => {
    const col = k.bad > y1 * .75 ? C.CRITICAL : k.bad > y1 * .55 ? C.HIGH : k.bad > y1 * .38 ? C.WATCH : C.HEALTHY;
    s += `<circle cx="${X(k.orders).toFixed(1)}" cy="${Y(k.bad).toFixed(1)}"
      r="${(3 + Math.sqrt(k.financed / mf) * 17).toFixed(1)}" fill="${col}" fill-opacity=".62"
      stroke="${col}" data-cat="${esc(k.name)}" class="cnode"/>`;
  });
  s += `<text x="${W / 2}" y="${H - 8}" text-anchor="middle" style="font-size:10.5px;fill:var(--muted)">orders (log scale) \u2192 bubble size = financed value</text></svg>`;
  const rows = K.slice(0, 16).map(k => `<tr><td>${k.name}</td><td class="n">${k.orders}</td>
    <td class="n">${pct(k.bad)}</td><td class="n">${pct(k.cancel, 2)}</td>
    <td class="n">${pct(k.late)}</td><td class="n">${brl(k.financed)}</td></tr>`).join('');
  const S = D.states, ms = Math.max(...S.map(s => s.bad));
  const sb = S.slice(0, 14).map(st => `<div class="bar-row"><span class="nm">${st.s}</span>
    <span class="bar" style="width:${st.bad / ms * 150}px;background:${st.bad > ms * .8 ? C.CRITICAL : st.bad > ms * .6 ? C.HIGH : C.WATCH}"></span>
    <span class="vv">${pct(st.bad)} \u00b7 ${st.orders}</span></div>`).join('');
  return `<div class="card"><h3>Category risk</h3>
    <div class="hint">High and to the right is the dangerous quadrant: a category that fails often
    and is financed at volume.</div>${s}</div>
  <div class="grid2" style="margin-top:var(--s5)">
    <div class="card"><h3>Worst categories</h3><div class="scroll"><table>
      <thead><tr><th>Category</th><th style="text-align:right">Orders</th>
      <th style="text-align:right">Bad</th><th style="text-align:right">Cancel</th>
      <th style="text-align:right">Late</th><th style="text-align:right">Financed</th></tr></thead>
      <tbody>${rows}</tbody></table></div></div>
    <div class="card"><h3>Delivery region</h3>
      <div class="hint">Bad-order rate by destination state.</div>${sb}</div></div>`;
}

/* ---------------------------------------------------------------- app */
const R = { portfolio, watchlist, merchant, evidence, network, categories };
let view = 'portfolio';
function open(id) { CUR = D.merchants.find(m => m.id === id); go('merchant'); }
function go(v) {
  view = v;
  document.querySelectorAll('.view').forEach(s => s.classList.remove('on'));
  document.querySelectorAll('#nav button').forEach(b =>
    b.setAttribute('aria-current', b.dataset.v === v));
  const label = VIEWS.find(x => x[0] === v)[1];
  $('#crumb').textContent = label;
  $('#title').textContent = v === 'merchant' && CUR ? CUR.short + '\u2026' : label;
  $('#note').textContent = NOTES[v];
  const el = $('#v-' + v); el.innerHTML = R[v](); el.classList.add('on');
  if (v === 'portfolio') wireTape(el);
  if (v === 'evidence') wireROI();
  if (v === 'merchant') $('#msel').onchange = e => open(e.target.value);
  el.querySelectorAll('tbody tr[data-id]').forEach(tr => tr.onclick = () => open(tr.dataset.id));
  el.querySelectorAll('.gnode').forEach(n => {
    const m = D.merchants.find(x => x.id === n.dataset.id);
    n.onmouseenter = e => tip(e, m ? `${m.short}\u2026\n${m.band}  score ${num(m.score)}` : n.dataset.id.slice(0, 10));
    n.onmousemove = move; n.onmouseleave = untip;
    n.onclick = () => m && open(m.id);
  });
  el.querySelectorAll('.cnode').forEach(n => {
    const k = D.categories.find(c => c.name === n.dataset.cat);
    n.onmouseenter = e => tip(e, `${k.name}\n${k.orders} orders  bad ${pct(k.bad)}\nfinanced ${brl(k.financed)}`);
    n.onmousemove = move; n.onmouseleave = untip;
  });
  el.querySelectorAll('th[data-k]').forEach(th => th.onclick = () => {
    const k = th.dataset.k;
    sortDir = sortKey === k ? -sortDir : -1; sortKey = k; go('watchlist');
  });
  scrollTo(0, 0);
}

fetch('data.json').then(r => r.json()).then(d => {
  D = d;
  $('#nav').innerHTML = VIEWS.map(([v, l]) =>
    `<button data-v="${v}">${l}</button>`).join('');
  document.querySelectorAll('#nav button').forEach(b => b.onclick = () => go(b.dataset.v));
  const at = D.merchants.filter(m => m.band === 'CRITICAL' || m.band === 'HIGH')
    .reduce((a, m) => a + (m.exposure || 0), 0);
  $('#foot-exposure').textContent = brl(at);
  $('#foot-lead').textContent = D.metrics.median_lead_days + ' days';
  $('#foot-gen').textContent = D.metrics.n_orders.toLocaleString() + ' orders \u00b7 ' + D.generated;
  go('portfolio');
}).catch(e => { $('#title').textContent = 'Could not load data.json'; console.error(e); });
