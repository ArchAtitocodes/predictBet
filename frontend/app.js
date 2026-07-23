/* ============================================================
   MODEL LOGIC — mirrors analytics.py exactly (same formulas).
   Nothing here inflates confidence; shrinkage & decay dampen
   small-sample noise rather than amplifying it.
   ============================================================ */

function weightedShrunkRate(values, leagueAvg, decay, shrinkageK) {
    const n = values.length;
    if (n === 0) return leagueAvg;
    const weights = values.map((_, i) => Math.pow(decay, n - 1 - i));
    const weightSum = weights.reduce((a, b) => a + b, 0);
    const weightedAvg = values.reduce((acc, v, i) => acc + v * weights[i], 0) / weightSum;
    const shrinkage = n / (n + shrinkageK);
    return shrinkage * weightedAvg + (1 - shrinkage) * leagueAvg;
}

function poissonPmf(k, lam) {
    if (lam <= 0) return k === 0 ? 1.0 : 0.0;
    let fact = 1;
    for (let i = 2; i <= k; i++) fact *= i;
    return Math.pow(lam, k) * Math.exp(-lam) / fact;
}

function buildModel(home, away, leagueAvgHome, leagueAvgAway, homeAdvantage, decay, shrinkageK, maxGoals) {
    maxGoals = maxGoals || 8;
    const homeScoredRate = weightedShrunkRate(home.scored, leagueAvgHome, decay, shrinkageK);
    const homeConcededRate = weightedShrunkRate(home.conceded, leagueAvgAway, decay, shrinkageK);
    const awayScoredRate = weightedShrunkRate(away.scored, leagueAvgAway, decay, shrinkageK);
    const awayConcededRate = weightedShrunkRate(away.conceded, leagueAvgHome, decay, shrinkageK);

    const homeAttack = leagueAvgHome ? homeScoredRate / leagueAvgHome : 0;
    const homeDefense = leagueAvgAway ? homeConcededRate / leagueAvgAway : 0;
    const awayAttack = leagueAvgAway ? awayScoredRate / leagueAvgAway : 0;
    const awayDefense = leagueAvgHome ? awayConcededRate / leagueAvgHome : 0;

    const expHome = leagueAvgHome * homeAttack * awayDefense * homeAdvantage;
    const expAway = leagueAvgAway * awayAttack * homeDefense;

    const homeProbs = [], awayProbs = [];
    for (let g = 0; g <= maxGoals; g++) { homeProbs.push(poissonPmf(g, expHome)); awayProbs.push(poissonPmf(g, expAway)); }

    let homeWin = 0, draw = 0, awayWin = 0, over25 = 0, bttsYes = 0;
    for (let hg = 0; hg <= maxGoals; hg++) {
        for (let ag = 0; ag <= maxGoals; ag++) {
            const p = homeProbs[hg] * awayProbs[ag];
            if (hg > ag) homeWin += p; else if (hg === ag) draw += p; else awayWin += p;
            if (hg + ag > 2.5) over25 += p;
            if (hg >= 1 && ag >= 1) bttsYes += p;
        }
    }
    const total = homeWin + draw + awayWin;
    if (total > 0) { homeWin /= total; draw /= total; awayWin /= total; }

    const sampleHome = home.scored.length, sampleAway = away.scored.length;
    const smallest = Math.min(sampleHome, sampleAway);
    const confidence = Math.round(75 * (1 - Math.exp(-smallest / 8)));
    let qualityNote;
    if (smallest < 5) qualityNote = "LOW sample size (<5 matches) — probabilities are high-variance estimates, not reliable predictions.";
    else if (smallest < 10) qualityNote = "MODERATE sample size — treat probabilities as rough estimates only.";
    else qualityNote = "Reasonable sample size for a simple form-based model.";

    return {
        home_team: home.name, away_team: away.name,
        home_win_prob: homeWin, draw_prob: draw, away_win_prob: awayWin,
        over_2_5_prob: over25, under_2_5_prob: 1 - over25,
        btts_yes_prob: bttsYes, btts_no_prob: 1 - bttsYes,
        expected_home_goals: expHome, expected_away_goals: expAway,
        sample_size_home: sampleHome, sample_size_away: sampleAway,
        confidence_score: confidence, data_quality_note: qualityNote
    };
}

function devig1x2(oh, od, oa) {
    const raw = [1 / oh, 1 / od, 1 / oa];
    const overround = raw.reduce((a, b) => a + b, 0);
    return raw.map(r => r / overround);
}

function compareToMarket(model, oh, od, oa) {
    const [mh, md, ma] = devig1x2(oh, od, oa);
    const overroundPct = (1 / oh + 1 / od + 1 / oa - 1) * 100;
    const edge = (mp, mkp) => Math.round((mp - mkp) * 1000) / 10;
    return {
        bookmaker_overround_pct: Math.round(overroundPct * 100) / 100,
        home: { model_prob_pct: Math.round(model.home_win_prob * 10000) / 100, market_implied_pct: Math.round(mh * 10000) / 100, edge_pct_points: edge(model.home_win_prob, mh) },
        draw: { model_prob_pct: Math.round(model.draw_prob * 10000) / 100, market_implied_pct: Math.round(md * 10000) / 100, edge_pct_points: edge(model.draw_prob, md) },
        away: { model_prob_pct: Math.round(model.away_win_prob * 10000) / 100, market_implied_pct: Math.round(ma * 10000) / 100, edge_pct_points: edge(model.away_win_prob, ma) },
        note: "Edge is the raw difference between Poisson GLM form-model probabilities and de-vigged market odds. A positive edge does NOT mean a bet is profitable — it may reflect information asymmetry where the market prices in factors the model does not capture (injuries, lineups, tactical shifts, motivation). Always cross-reference with additional intelligence sources."
    };
}

/* ============================================================
   STATE + RENDERING
   ============================================================ */

let matches = [];
let selectedIndex = null;
let serverAvailable = false;

const $ = sel => document.querySelector(sel);

function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// Theme initialization
const currentTheme = localStorage.getItem('theme') || 'light';
if (currentTheme === 'dark') document.documentElement.setAttribute('data-theme', 'dark');

// Toast Notification System
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
    toast.innerHTML = `<span class="toast-icon">${icon}</span><span>${escapeHtml(message)}</span>`;
    container.appendChild(toast);

    // Trigger reflow to ensure animation plays
    void toast.offsetWidth;
    toast.classList.add('show');

    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
const matchListEl = $('#matchList');
const sidebarEmpty = $('#sidebarEmpty');
const matchCountEl = $('#matchCount');
const mainEl = $('#main');

function qualityChipClass(note) {
    if (!note) return 'ok';
    if (note.startsWith('LOW')) return 'low';
    if (note.startsWith('MODERATE')) return 'mod';
    return 'ok';
}
function qualityChipText(note) {
    if (!note) return 'OK SAMPLE';
    if (note.startsWith('LOW')) return 'LOW SAMPLE';
    if (note.startsWith('MODERATE')) return 'MODERATE';
    return 'OK SAMPLE';
}

function renderSidebar() {
    matchCountEl.textContent = matches.length;
    matchListEl.innerHTML = '';
    if (matches.length === 0) {
        matchListEl.appendChild(sidebarEmpty);
        return;
    }
    const filterText = ($('#sidebarSearch') ? $('#sidebarSearch').value.toLowerCase().trim() : '');
    let visibleCount = 0;

    matches.forEach((rec, i) => {
        const m = rec.model;
        if (filterText && !m.home_team.toLowerCase().includes(filterText) && !m.away_team.toLowerCase().includes(filterText)) {
            return; // Skip if it doesn't match filter
        }
        visibleCount++;
        const row = document.createElement('div');
        row.className = 'match-row' + (i === selectedIndex ? ' active' : '');
        row.style.position = 'relative';
        row.innerHTML = `
      <button class="match-delete" title="Remove fixture" aria-label="Remove ${escapeHtml(m.home_team)} vs ${escapeHtml(m.away_team)}">&times;</button>
      <div class="teams">${escapeHtml(m.home_team)} <span style="color:var(--text-dim); font-weight:400;">vs</span> ${escapeHtml(m.away_team)}</div>
      <div class="row-bars">
        <span style="width:${(m.home_win_prob * 100).toFixed(1)}%; background:var(--cyan);"></span>
        <span style="width:${(m.draw_prob * 100).toFixed(1)}%; background:var(--text-faint);"></span>
        <span style="width:${(m.away_win_prob * 100).toFixed(1)}%; background:var(--amber);"></span>
      </div>
      <div class="meta">
        <span>${escapeHtml(rec.match_label || (rec.match_date || 'Unlabeled fixture'))}</span>
        <span class="chip ${qualityChipClass(rec.data_quality_note)}">${escapeHtml(qualityChipText(rec.data_quality_note))}</span>
      </div>
    `;
        row.querySelector('.match-delete').addEventListener('click', (e) => {
            e.stopPropagation();
            if (confirm(`Remove ${escapeHtml(m.home_team)} vs ${escapeHtml(m.away_team)}?`)) {
                matches.splice(i, 1);
                if (selectedIndex === i) selectedIndex = matches.length > 0 ? 0 : null;
                else if (selectedIndex > i) selectedIndex--;
                renderSidebar();
                renderMain();
                showToast('Fixture removed', 'info');
            }
        });
        row.addEventListener('click', () => {
            selectedIndex = i;
            renderSidebar();
            renderMain();
            enterMobileDetailView();
        });
        matchListEl.appendChild(row);
    });

    if (visibleCount === 0 && filterText) {
        matchListEl.innerHTML = '<div class="empty-state">No matches match your search.</div>';
    }
}

if ($('#sidebarSearch')) {
    $('#sidebarSearch').addEventListener('input', renderSidebar);
}

function gaugeSvg(score) {
    const cx = 90, cy = 90, r = 70;
    const startAngle = 180, endAngle = 0;
    const clamped = Math.max(0, Math.min(100, score));
    const angle = startAngle - (startAngle - endAngle) * (clamped / 100);
    const rad = angle * Math.PI / 180;
    const nx = cx + (r - 14) * Math.cos(rad);
    const ny = cy - (r - 14) * Math.sin(rad);

    let ticks = '';
    for (let t = 0; t <= 100; t += 10) {
        const a = (startAngle - (startAngle - endAngle) * (t / 100)) * Math.PI / 180;
        const inner = r - 6, outer = r;
        const x1 = cx + inner * Math.cos(a), y1 = cy - inner * Math.sin(a);
        const x2 = cx + outer * Math.cos(a), y2 = cy - outer * Math.sin(a);
        ticks += `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="var(--border)" stroke-width="2"/>`;
    }
    const arcColor = clamped < 35 ? 'var(--red)' : clamped < 60 ? 'var(--amber)' : 'var(--cyan)';
    const trackPath = describeArc(cx, cy, r, 0, 180);
    const valuePath = describeArc(cx, cy, r, 180 - (180 * clamped / 100), 180);

    return `
  <svg width="180" height="110" viewBox="0 0 180 110" class="gauge-svg">
    <path d="${trackPath}" fill="none" stroke="var(--panel-alt)" stroke-width="10" stroke-linecap="round"/>
    <path d="${valuePath}" fill="none" stroke="${arcColor}" stroke-width="10" stroke-linecap="round" class="gauge-value-arc"/>
    ${ticks}
    <line x1="${cx}" y1="${cy}" x2="${nx.toFixed(1)}" y2="${ny.toFixed(1)}" stroke="var(--text)" stroke-width="2.5" stroke-linecap="round"/>
    <circle cx="${cx}" cy="${cy}" r="4" fill="var(--text)"/>
  </svg>`;
}

function polarToXY(cx, cy, r, angleDeg) {
    const rad = angleDeg * Math.PI / 180;
    return { x: cx + r * Math.cos(rad), y: cy - r * Math.sin(rad) };
}
function describeArc(cx, cy, r, startDeg, endDeg) {
    const start = polarToXY(cx, cy, r, startDeg);
    const end = polarToXY(cx, cy, r, endDeg);
    const largeArc = (endDeg - startDeg) > 180 ? 1 : 0;
    return `M ${start.x.toFixed(1)} ${start.y.toFixed(1)} A ${r} ${r} 0 ${largeArc} 0 ${end.x.toFixed(1)} ${end.y.toFixed(1)}`;
}

function barRow(label, modelPct, marketPct) {
    const hasMarket = marketPct !== undefined && marketPct !== null;
    return `
    <div class="bar-row">
      <div class="bl">${label}</div>
      <div class="bar-track">
        <div class="bar-fill model" style="width:${modelPct}%;"></div>
        ${hasMarket ? `<div class="bar-fill market" style="width:${marketPct}%; opacity:0.9;"></div>` : ''}
      </div>
      <div class="bar-vals">${modelPct.toFixed(1)}%${hasMarket ? ` &nbsp;/&nbsp; ${marketPct.toFixed(1)}%` : ''}</div>
    </div>`;
}

function renderFinancialCard(label, finData) {
    if (!finData || !finData.ticker) {
        return `
      <div class="card fin-card">
        <div class="card-label">Corporate Valuation — ${label}</div>
        <div style="font-family: var(--font-mono); font-size: 12px; color: var(--text-faint); padding: 12px 0;">Not a publicly traded club</div>
      </div>`;
    }
    const changeVal = finData.change_pct || 0;
    const changeClass = changeVal > 0 ? 'up' : changeVal < 0 ? 'down' : 'neutral';
    const changeSign = changeVal > 0 ? '+' : '';
    const tagClass = changeVal >= 0 ? '' : 'neg';
    return `
    <div class="card fin-card">
      <div class="card-label">Corporate Valuation — ${label}</div>
      <div class="fin-row">
        <span class="fin-label">Ticker</span>
        <span class="fin-val neutral">${escapeHtml(finData.ticker || 'N/A')}<span class="fin-tag ${tagClass}">${changeSign}${changeVal.toFixed(2)}%</span></span>
      </div>
      <div class="fin-row">
        <span class="fin-label">Price</span>
        <span class="fin-val ${changeClass}">${escapeHtml(finData.currency || '$')}${(finData.price || 0).toFixed(2)}</span>
      </div>
      <div class="fin-row">
        <span class="fin-label">Market Cap</span>
        <span class="fin-val neutral">${escapeHtml(finData.market_cap || 'N/A')}</span>
      </div>
      <div class="fin-row">
        <span class="fin-label">52W Range</span>
        <span style="font-family: var(--font-mono); font-size: 11px; color: var(--text-dim);">${escapeHtml(finData.fifty_two_week_range || 'N/A')}</span>
      </div>
    </div>`;
}

function renderMain() {
    if (selectedIndex === null || !matches[selectedIndex]) {
        mainEl.innerHTML = `<div class="main-empty">
      <div class="glyph">&#8942;&#8942;&#8942;</div>
      <div class="msg">Select a fixture from the left, or add one to see the full model breakdown — expected goals, 1X2 probabilities, goals markets, and market comparison if odds are supplied.</div>
    </div>`;
        return;
    }
    const rec = matches[selectedIndex];
    const m = rec.model;
    const mc = rec.market_comparison;
    const maxXg = Math.max(m.expected_home_goals, m.expected_away_goals, 0.5);

    let marketBlock = '';
    if (mc) {
        const edgeCell = v => `<td class="${v >= 0 ? 'pos' : 'neg'}">${v >= 0 ? '+' : ''}${v.toFixed(1)} pts</td>`;
        marketBlock = `
      <div class="card full-card">
        <div class="card-label">Model vs. Market (de-vigged, overround ${mc.bookmaker_overround_pct}%)</div>
        <div class="table-scroll">
        <table class="edge-table">
          <thead><tr><th>Outcome</th><th>Model</th><th>Market (fair)</th><th>Edge</th></tr></thead>
          <tbody>
            <tr><td>Home win</td><td>${mc.home.model_prob_pct.toFixed(1)}%</td><td>${mc.home.market_implied_pct.toFixed(1)}%</td>${edgeCell(mc.home.edge_pct_points)}</tr>
            <tr><td>Draw</td><td>${mc.draw.model_prob_pct.toFixed(1)}%</td><td>${mc.draw.market_implied_pct.toFixed(1)}%</td>${edgeCell(mc.draw.edge_pct_points)}</tr>
            <tr><td>Away win</td><td>${mc.away.model_prob_pct.toFixed(1)}%</td><td>${mc.away.market_implied_pct.toFixed(1)}%</td>${edgeCell(mc.away.edge_pct_points)}</tr>
          </tbody>
        </table>
        </div>
        <div class="note-banner">${escapeHtml(mc.note)}</div>
      </div>`;
    } else if (rec.odds && (rec.odds.home || rec.odds.draw || rec.odds.away)) {
        marketBlock = `
      <div class="card full-card">
        <div class="card-label">Scoreboard Market Odds</div>
        <div style="font-family: var(--font-mono); font-size: 13px;">
          Odds: 
          Home: <span style="color:var(--cyan)">${escapeHtml(String(rec.odds.home || 'N/A'))}</span> &nbsp;|&nbsp; 
          Draw: <span style="color:var(--text-dim)">${escapeHtml(String(rec.odds.draw || 'N/A'))}</span> &nbsp;|&nbsp; 
          Away: <span style="color:var(--amber)">${escapeHtml(String(rec.odds.away || 'N/A'))}</span>
        </div>
      </div>`;
    }

    // Heatmap — shows if scraper-built model generated one
    const heatmapSrc = rec.heatmap_file ? `/api/chart?file=${encodeURIComponent(rec.heatmap_file)}&t=${Date.now()}` : '/api/chart?t=' + Date.now();
    const heatmapBlock = rec.sources ? `
    <div class="card heatmap-card">
      <div class="card-label">Scoreline Probability Heatmap (Poisson Grid)</div>
      <img src="${heatmapSrc}" alt="Scoreline Heatmap" loading="lazy" onerror="this.parentElement.innerHTML='<div class=heatmap-placeholder>Heatmap not available for manual entries. Use Auto Scrape to generate.</div>'">
    </div>` : '';

    // Financial cards
    let financialBlock = '';
    if (rec.financials) {
        financialBlock = renderFinancialCard(m.home_team, rec.financials.home)
            + renderFinancialCard(m.away_team, rec.financials.away);
    }

    mainEl.innerHTML = `
    <button class="btn-back-mobile" id="btnBackMobile" aria-label="Back to fixture list">&larr; All fixtures</button>
    <div class="fixture-head">
      <div>
        <div class="fixture-title">${escapeHtml(m.home_team)}<span class="vs">vs</span>${escapeHtml(m.away_team)}</div>
        <div class="fixture-meta">${rec.match_label ? escapeHtml(rec.match_label) + ' &middot; ' : ''}${escapeHtml(rec.match_date || 'no date set')} &middot; ${m.sample_size_home} / ${m.sample_size_away} matches sampled
          ${rec.sources ? ' &middot; <span style="color:var(--cyan);">Scraped from ' + escapeHtml(rec.sources.team_form) + '</span>' : ''}
        </div>
      </div>
    </div>

    <div class="grid">
      <div class="card gauge-card">
        <div class="card-label">Data Sufficiency</div>
        ${gaugeSvg(rec.confidence_score)}
        <div class="gauge-readout">${rec.confidence_score}<span style="font-size:14px; color:var(--text-dim);">/100</span></div>
        <div class="gauge-caption">Sample-size based confidence metric, capped at 75.</div>
      </div>

      <div class="card xg-card">
        <div class="card-label">Model Expected Goals</div>
        <div class="xg-row"><span class="xg-team">${escapeHtml(m.home_team)}</span><span class="xg-val">${m.expected_home_goals.toFixed(2)}</span></div>
        <div class="xg-bar-track"><div class="xg-bar-fill" style="width:${(m.expected_home_goals / maxXg * 100).toFixed(1)}%;"></div></div>
        <div class="xg-row" style="margin-top:14px;"><span class="xg-team">${escapeHtml(m.away_team)}</span><span class="xg-val away">${m.expected_away_goals.toFixed(2)}</span></div>
        <div class="xg-bar-track"><div class="xg-bar-fill away" style="width:${(m.expected_away_goals / maxXg * 100).toFixed(1)}%;"></div></div>
      </div>

      <div class="card xg-card">
        <div class="card-label">Goals Markets</div>
        <div class="xg-row"><span class="xg-team">Over 2.5</span><span class="xg-val">${(m.over_2_5_prob * 100).toFixed(1)}%</span></div>
        <div class="xg-bar-track"><div class="xg-bar-fill" style="width:${(m.over_2_5_prob * 100).toFixed(1)}%;"></div></div>
        <div class="xg-row" style="margin-top:14px;"><span class="xg-team">BTTS Yes</span><span class="xg-val away">${(m.btts_yes_prob * 100).toFixed(1)}%</span></div>
        <div class="xg-bar-track"><div class="xg-bar-fill away" style="width:${(m.btts_yes_prob * 100).toFixed(1)}%;"></div></div>
      </div>

      <div class="card full-card">
        <div class="card-label">1X2 Probabilities ${mc ? '&mdash; model vs. de-vigged market' : ''}</div>
        <div class="bar-chart">
          ${barRow('Home win', m.home_win_prob * 100, mc ? mc.home.market_implied_pct : null)}
          ${barRow('Draw', m.draw_prob * 100, mc ? mc.draw.market_implied_pct : null)}
          ${barRow('Away win', m.away_win_prob * 100, mc ? mc.away.market_implied_pct : null)}
        </div>
        <div class="legend">
          <div class="legend-item"><span class="swatch model"></span>Model</div>
          ${mc ? '<div class="legend-item"><span class="swatch market"></span>Market (fair)</div>' : ''}
        </div>
      </div>

      ${marketBlock}

      ${heatmapBlock}

      ${financialBlock}

      ${m.elo_home && m.elo_away ? `
      <div class="card full-card">
        <div class="card-label">ClubELO Ratings</div>
        <div class="xg-row">
          <span class="xg-team">${escapeHtml(m.home_team)} ELO</span>
          <span class="xg-val" style="color:var(--accent);">${Math.round(m.elo_home)}</span>
        </div>
        <div class="xg-row" style="margin-top:8px;">
          <span class="xg-team">${escapeHtml(m.away_team)} ELO</span>
          <span class="xg-val away" style="color:var(--accent);">${Math.round(m.elo_away)}</span>
        </div>
        <div style="font-size: 13px; color:var(--text-muted); margin-top: 12px; border-top: 1px solid var(--border); padding-top: 8px;">
          ELO Differential: <b>${Math.round(m.elo_home - m.elo_away)}</b> pts
        </div>
      </div>
      ` : ''}

      <div class="card full-card">
        <div class="note-banner quality">
          <b>Data quality:</b> ${escapeHtml(rec.data_quality_note || 'N/A')} Model uses Poisson GLM with attack/defense ratings, recency-weighted form, and Bayesian shrinkage. 
          ${m.dixon_coles_applied ? '<b>Dixon-Coles adjustment is active</b> to correct low-scoring draws.' : ''}
          Additional data sources (ClubELO, financial, league-wide) are incorporated when available.
        </div>
      </div>
    </div>
  `;

    const backBtn = $('#btnBackMobile');
    if (backBtn) backBtn.addEventListener('click', exitMobileDetailView);
}

/* ============================================================
   MOBILE MASTER/DETAIL NAVIGATION
   ============================================================ */

function enterMobileDetailView() {
    if (window.innerWidth <= 768) {
        document.querySelector('.body')?.classList.add('show-detail');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}
function exitMobileDetailView() {
    document.querySelector('.body')?.classList.remove('show-detail');
}
window.addEventListener('resize', () => {
    if (window.innerWidth > 768) {
        document.querySelector('.body')?.classList.remove('show-detail');
    }
});

/* ============================================================
   SERVER STATUS CHECK
   ============================================================ */

async function checkServerStatus() {
    try {
        const res = await fetch('/api/search?query=test');
        if (res.ok) {
            serverAvailable = true;
            $('#serverStatus').className = 'server-status online';
            $('#serverStatusText').textContent = 'Online Scraper';
            $('#tabAuto').classList.remove('disabled');
            setTab('auto');
        } else {
            throw new Error();
        }
    } catch (e) {
        serverAvailable = false;
        $('#serverStatus').className = 'server-status';
        $('#serverStatusText').textContent = 'Offline Mode';
        $('#tabAuto').classList.add('disabled');
        setTab('manual');
    }
}

/* ============================================================
   MODAL / ADD MATCH TABS
   ============================================================ */

const modalBackdrop = $('#modalBackdrop');
let activeTab = 'auto';

function setTab(tabName) {
    if (tabName === 'auto' && !serverAvailable) return;

    activeTab = tabName;
    if (tabName === 'auto') {
        $('#tabAuto').classList.add('active');
        $('#tabManual').classList.remove('active');
        $('#contentAuto').classList.add('active');
        $('#contentManual').classList.remove('active');
    } else {
        $('#tabAuto').classList.remove('active');
        $('#tabManual').classList.add('active');
        $('#contentAuto').classList.remove('active');
        $('#contentManual').classList.add('active');
    }
}

$('#tabAuto').addEventListener('click', () => setTab('auto'));
$('#tabManual').addEventListener('click', () => setTab('manual'));

function openModal() {
    modalBackdrop.classList.add('open');
    document.body.classList.add('modal-open');
    $('#formError').style.display = 'none';
    $('#autoFormError').style.display = 'none';
    checkServerStatus();
}
function closeModal() {
    modalBackdrop.classList.remove('open');
    document.body.classList.remove('modal-open');
}

$('#btnAdd').addEventListener('click', openModal);
$('#btnAutoCancel').addEventListener('click', closeModal);
$('#btnManualCancel').addEventListener('click', closeModal);
modalBackdrop.addEventListener('click', e => { if (e.target === modalBackdrop) closeModal(); });

// Theme Toggle
const themeBtn = $('#btnThemeToggle');
if (themeBtn) {
    themeBtn.textContent = currentTheme === 'dark' ? '☀️' : '🌙';
    themeBtn.addEventListener('click', () => {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        if (isDark) {
            document.documentElement.removeAttribute('data-theme');
            localStorage.setItem('theme', 'light');
            themeBtn.textContent = '🌙';
            showToast('Switched to Light Mode', 'info');
        } else {
            document.documentElement.setAttribute('data-theme', 'dark');
            localStorage.setItem('theme', 'dark');
            themeBtn.textContent = '☀️';
            showToast('Switched to Dark Mode', 'info');
        }
    });
}

// Clear All
const clearBtn = $('#btnClearAll');
if (clearBtn) {
    clearBtn.addEventListener('click', () => {
        if (matches.length === 0) return;
        if (confirm('Are you sure you want to clear all loaded fixtures?')) {
            matches = [];
            selectedIndex = null;
            renderSidebar();
            renderMain();
            exitMobileDetailView();
            showToast('All fixtures cleared', 'info');
        }
    });
}

$('#configTrigger').addEventListener('click', () => {
    $('#configTrigger').classList.toggle('open');
    $('#configContent').classList.toggle('open');
});

/* ============================================================
   AUTO SCRAPE AUTOCOMPLETE & SEARCH
   ============================================================ */

let selectedHomeTeam = null;
let selectedAwayTeam = null;

function setupAutocomplete(inputEl, suggEl, onSelect) {
    let debounceTimeout = null;

    inputEl.addEventListener('input', () => {
        clearTimeout(debounceTimeout);
        const query = inputEl.value.trim();
        if (query.length < 2) {
            suggEl.innerHTML = '';
            suggEl.style.display = 'none';
            return;
        }

        debounceTimeout = setTimeout(async () => {
            try {
                const res = await fetch(`/api/search?query=${encodeURIComponent(query)}`);
                if (!res.ok) return;
                const list = await res.json();

                suggEl.innerHTML = '';
                if (list.length === 0) {
                    suggEl.style.display = 'none';
                    return;
                }

                list.forEach(team => {
                    const item = document.createElement('div');
                    item.className = 'suggestion-item';
                    item.innerHTML = `<span>${escapeHtml(team.name)}</span><span class="subtitle">${escapeHtml(team.subtitle)}</span>`;
                    item.addEventListener('mousedown', (e) => {
                        inputEl.value = team.name;
                        onSelect(team);
                        suggEl.innerHTML = '';
                        suggEl.style.display = 'none';
                    });
                    suggEl.appendChild(item);
                });
                suggEl.style.display = 'block';
            } catch (e) { }
        }, 250);
    });

    inputEl.addEventListener('blur', () => {
        setTimeout(() => {
            suggEl.style.display = 'none';
        }, 200);
    });
}

setupAutocomplete($('#auto_home'), $('#sugg_home'), (team) => {
    selectedHomeTeam = team;
});

setupAutocomplete($('#auto_away'), $('#sugg_away'), (team) => {
    selectedAwayTeam = team;
});

// Auto Scrape Form Submit
let isScraping = false;
$('#btnAutoSubmit').addEventListener('click', async () => {
    if (isScraping) return;
    const errEl = $('#autoFormError');
    errEl.style.display = 'none';

    if (!selectedHomeTeam || !selectedAwayTeam) {
        errEl.textContent = 'Please search and select both teams from the autocomplete suggestions.';
        errEl.style.display = 'block';
        return;
    }

    const homeName = $('#auto_home').value.trim();
    const awayName = $('#auto_away').value.trim();
    if (homeName !== selectedHomeTeam.name || awayName !== selectedAwayTeam.name) {
        errEl.textContent = 'Please select teams from the dropdown. Typed names must match resolved teams.';
        errEl.style.display = 'block';
        return;
    }

    isScraping = true;
    $('#btnAutoSubmit').disabled = true;
    $('#btnAutoText').innerHTML = `<span class="loader"></span>Scraping...`;

    const decay = parseFloat($('#f_auto_decay').value) || 0.92;
    const shrinkageK = parseFloat($('#f_auto_shrinkage').value) || 6.0;
    const homeAdvantage = parseFloat($('#f_auto_hadv').value) || 1.0;

    const leagueSlug = selectedHomeTeam.league || selectedAwayTeam.league || 'eng.1';

    try {
        const url = `/api/scrape?home_id=${selectedHomeTeam.id}&away_id=${selectedAwayTeam.id}&league_slug=${leagueSlug}&decay=${decay}&shrinkage_k=${shrinkageK}&home_advantage=${homeAdvantage}&match_date=${new Date().toISOString().slice(0, 10)}`;
        const res = await fetch(url);
        if (!res.ok) {
            const errText = await res.text();
            throw new Error(errText || 'Scraping request failed');
        }
        const record = await res.json();

        matches.push(record);
        selectedIndex = matches.length - 1;

        renderSidebar();
        renderMain();
        closeModal();
        showToast(`Model built for ${homeName} vs ${awayName}`, 'success');

        $('#auto_home').value = '';
        $('#auto_away').value = '';
        selectedHomeTeam = null;
        selectedAwayTeam = null;
    } catch (e) {
        errEl.textContent = 'Error: ' + e.message;
        errEl.style.display = 'block';
    } finally {
        isScraping = false;
        $('#btnAutoSubmit').disabled = false;
        $('#btnAutoText').textContent = 'Scrape & Build';
    }
});

/* ============================================================
   MANUAL FORM SUBMIT
   ============================================================ */

function parseGoalList(s) {
    return s.split(',').map(x => x.trim()).filter(x => x !== '').map(x => {
        const n = parseInt(x, 10);
        if (isNaN(n)) throw new Error(`"${x}" is not a valid integer goal count`);
        return n;
    });
}

$('#btnSubmit').addEventListener('click', () => {
    const errEl = $('#formError');
    try {
        const homeTeam = $('#f_home_team').value.trim() || 'Home Team';
        const awayTeam = $('#f_away_team').value.trim() || 'Away Team';
        const homeScored = parseGoalList($('#f_home_scored').value);
        const homeConceded = parseGoalList($('#f_home_conceded').value);
        const awayScored = parseGoalList($('#f_away_scored').value);
        const awayConceded = parseGoalList($('#f_away_conceded').value);

        if (homeScored.length === 0 || awayScored.length === 0) {
            throw new Error('Enter at least one match of goals scored for each team.');
        }
        if (homeScored.length !== homeConceded.length) {
            throw new Error('Home scored/conceded lists must be the same length.');
        }
        if (awayScored.length !== awayConceded.length) {
            throw new Error('Away scored/conceded lists must be the same length.');
        }

        const leagueHome = parseFloat($('#f_lg_home').value) || 1.45;
        const leagueAway = parseFloat($('#f_lg_away').value) || 1.15;
        const homeAdv = parseFloat($('#f_hadv').value) || 1.0;
        const label = $('#f_label').value.trim() || null;

        const model = buildModel(
            { name: homeTeam, scored: homeScored, conceded: homeConceded },
            { name: awayTeam, scored: awayScored, conceded: awayConceded },
            leagueHome, leagueAway, homeAdv, 0.92, 6.0, 8
        );

        let marketComparison = null;
        const oh = parseFloat($('#f_odds_home').value);
        const od = parseFloat($('#f_odds_draw').value);
        const oa = parseFloat($('#f_odds_away').value);
        if (oh && od && oa) {
            marketComparison = compareToMarket(model, oh, od, oa);
        }

        matches.push({
            match_label: label,
            match_date: new Date().toISOString().slice(0, 10),
            model: model,
            confidence_score: model.confidence_score,
            data_quality_note: model.data_quality_note,
            market_comparison: marketComparison
        });
        selectedIndex = matches.length - 1;
        renderSidebar();
        renderMain();
        closeModal();
        clearManualForm();
        showToast(`Manual model built successfully`, 'success');
    } catch (e) {
        errEl.textContent = e.message;
        errEl.style.display = 'block';
    }
});

function clearManualForm() {
    ['f_home_team', 'f_away_team', 'f_home_scored', 'f_home_conceded', 'f_away_scored', 'f_away_conceded', 'f_label', 'f_odds_home', 'f_odds_draw', 'f_odds_away']
        .forEach(id => $('#' + id).value = '');
    $('#f_lg_home').value = '1.45';
    $('#f_lg_away').value = '1.15';
    $('#f_hadv').value = '1.0';
}

/* ============================================================
   IMPORT / EXPORT
   ============================================================ */

$('#btnImport').addEventListener('click', () => $('#fileInput').click());
$('#fileInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
        try {
            const data = JSON.parse(evt.target.result);
            const incoming = Array.isArray(data) ? data : [data];
            let added = 0;
            incoming.forEach(rec => {
                if (rec && rec.model && rec.model.home_team) {
                    if (rec.confidence_score === undefined) {
                        const smallest = Math.min(rec.model.sample_size_home, rec.model.sample_size_away);
                        rec.confidence_score = Math.round(75 * (1 - Math.exp(-smallest / 8)));
                    }
                    if (!rec.data_quality_note) {
                        const smallest = Math.min(rec.model.sample_size_home, rec.model.sample_size_away);
                        rec.data_quality_note = smallest < 5 ? "LOW sample size (<5 matches) — probabilities are high-variance estimates, not reliable predictions."
                            : smallest < 10 ? "MODERATE sample size — treat probabilities as rough estimates only."
                                : "Reasonable sample size for a simple form-based model.";
                    }
                    matches.push(rec);
                    added++;
                }
            });
            if (added > 0) {
                selectedIndex = matches.length - 1;
                renderSidebar();
                renderMain();
                showToast(`Imported ${added} match record(s)`, 'success');
            } else {
                showToast('No valid match records found in that file.', 'error');
            }
        } catch (err) {
            showToast('Could not parse JSON: ' + err.message, 'error');
        }
        e.target.value = '';
    };
    reader.readAsText(file);
});

$('#btnExport').addEventListener('click', () => {
    if (matches.length === 0) { showToast('No matches in this session yet.', 'error'); return; }
    const blob = new Blob([JSON.stringify(matches, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'match_model_session.json';
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
});

/* ============================================================
   BETIKA FIXTURES BROWSER
   ============================================================ */

const betikaModal = $('#betikaModalBackdrop');
let allBetikaFixtures = [];
let betikaTab = 'upcoming';

function setBetikaTab(tab) {
    betikaTab = tab;
    if (tab === 'upcoming') {
        $('#tabUpcoming').classList.add('active');
        $('#tabLive').classList.remove('active');
        $('#contentUpcoming').classList.add('active');
        $('#contentLive').classList.remove('active');
    } else {
        $('#tabUpcoming').classList.remove('active');
        $('#tabLive').classList.add('active');
        $('#contentUpcoming').classList.remove('active');
        $('#contentLive').classList.add('active');
    }
}

$('#tabUpcoming').addEventListener('click', () => {
    if (!serverAvailable) return;
    setBetikaTab('upcoming');
    if (allBetikaFixtures.length === 0) loadBetikaFixtures();
});

$('#tabLive').addEventListener('click', () => {
    if (!serverAvailable) return;
    setBetikaTab('live');
    loadBetikaLive();
});

$('#btnBetika').addEventListener('click', () => {
    if (!serverAvailable) return;
    betikaModal.classList.add('open');
    document.body.classList.add('modal-open');
    if (allBetikaFixtures.length === 0) loadBetikaFixtures();
});

$('#btnBetikaClose').addEventListener('click', () => {
    betikaModal.classList.remove('open');
    document.body.classList.remove('modal-open');
});

betikaModal.addEventListener('click', e => {
    if (e.target === betikaModal) {
        betikaModal.classList.remove('open');
        document.body.classList.remove('modal-open');
    }
});

$('#btnRefreshFixtures').addEventListener('click', () => {
    allBetikaFixtures = [];
    loadBetikaFixtures();
});

$('#betikaSearch').addEventListener('input', () => {
    renderBetikaFixtures($('#betikaSearch').value.trim().toLowerCase());
});

$('#fixturesList').addEventListener('click', (e) => {
    const btn = e.target.closest('.btn-betika-build');
    if (!btn) return;
    const homeTeam = btn.getAttribute('data-home');
    const awayTeam = btn.getAttribute('data-away');
    const matchId = btn.getAttribute('data-match');
    if (homeTeam && awayTeam) {
        buildModelFromBetika(homeTeam, awayTeam, matchId);
    }
});

function renderFixtureCard(m, showActions) {
    const time = m.start_time ? m.start_time.replace(' ', ' &middot; ') : 'TBD';
    const odds = [m.home_odd, m.draw_odd, m.away_odd].filter(Boolean).join(' / ');
    const oddsHtml = odds ? `<span style="color: var(--cyan); font-family: var(--font-mono); font-size: 11px;">${escapeHtml(odds)}</span>` : '';
    const liveBadge = m.is_live ? '<span class="chip low" style="margin-left: 6px;">LIVE</span>' : '';
    const comp = m.competition_name || '';
    const category = m.category || '';

    let actionHtml = '';
    if (showActions) {
        actionHtml = `
      <button class="primary" style="padding:5px 10px; font-size:11px; margin-top:8px;"
              data-home="${escapeHtml(m.home_team)}" data-away="${escapeHtml(m.away_team)}" data-match="${escapeHtml(m.match_id || '')}" class="btn-betika-build">
        Build Model
      </button>
    `;
    }

    return `
    <div style="padding: 14px 16px; border-bottom: 1px solid var(--border); display: flex; flex-direction: column; gap: 6px;">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-family: var(--font-mono); font-size: 10.5px; color: var(--text-dim);">${escapeHtml(time)}${liveBadge}</span>
        ${oddsHtml}
      </div>
      <div style="font-family: var(--font-ui); font-weight: 600; font-size: 14px;">
        ${escapeHtml(m.home_team)} <span style="color: var(--text-dim); font-weight: 400;">vs</span> ${escapeHtml(m.away_team)}
      </div>
      <div style="font-family: var(--font-mono); font-size: 10px; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.06em;">
        ${escapeHtml(comp)} ${category ? '&middot; ' + escapeHtml(category) : ''}
      </div>
      ${actionHtml}
    </div>
  `;
}

function renderBetikaFixtures(filterText) {
    const container = $('#fixturesList');
    let fixtures = allBetikaFixtures;
    if (filterText) {
        fixtures = fixtures.filter(m =>
            m.home_team.toLowerCase().includes(filterText) ||
            m.away_team.toLowerCase().includes(filterText) ||
            (m.competition_name && m.competition_name.toLowerCase().includes(filterText))
        );
    }
    if (fixtures.length === 0) {
        container.innerHTML = '<div class="empty-state">No fixtures found.</div>';
        return;
    }
    container.innerHTML = fixtures.map(m => renderFixtureCard(m, true)).join('');
}

function renderBetikaLive() {
    const container = $('#liveList');
    // We reuse the same fixture data but mark live ones
    const liveFixtures = allBetikaFixtures.filter(m => m.is_live);
    if (liveFixtures.length === 0) {
        container.innerHTML = '<div class="empty-state">No live matches at this time.</div>';
        return;
    }
    container.innerHTML = liveFixtures.map(m => renderFixtureCard(m, true)).join('');
}

async function loadBetikaFixtures() {
    const container = $('#fixturesList');
    container.innerHTML = '<div class="empty-state"><span class="loader" style="margin-right:8px;"></span> Loading fixtures from Betika...</div>';
    try {
        const res = await fetch('/api/betika/fixtures?limit=100');
        if (!res.ok) throw new Error('Failed to load fixtures');
        const data = await res.json();
        allBetikaFixtures = data.data || [];
        renderBetikaFixtures($('#betikaSearch').value.trim().toLowerCase());
    } catch (e) {
        container.innerHTML = `<div class="empty-state" style="color: var(--red);">Error: ${escapeHtml(e.message)}</div>`;
    }
}

async function loadBetikaLive() {
    const container = $('#liveList');
    container.innerHTML = '<div class="empty-state"><span class="loader" style="margin-right:8px;"></span> Loading live matches...</div>';
    try {
        const res = await fetch('/api/betika/live');
        if (!res.ok) throw new Error('Failed to load live matches');
        const data = await res.json();
        // Merge with existing fixtures, marking live ones
        const liveMatches = data.data || [];
        const liveIds = new Set(liveMatches.map(m => m.match_id));
        allBetikaFixtures.forEach(m => {
            m.is_live = liveIds.has(m.match_id);
        });
        // Add any live matches not in the main list
        for (const lm of liveMatches) {
            if (!allBetikaFixtures.find(m => m.match_id === lm.match_id)) {
                allBetikaFixtures.push({ ...lm, is_live: true });
            }
        }
        renderBetikaLive();
    } catch (e) {
        container.innerHTML = `<div class="empty-state" style="color: var(--red);">Error: ${escapeHtml(e.message)}</div>`;
    }
}

async function buildModelFromBetika(homeTeam, awayTeam, matchId) {
    if (isScraping) return;
    const errEl = $('#autoFormError');
    errEl.style.display = 'none';

    isScraping = true;
    $('#btnAutoSubmit').disabled = true;
    $('#btnAutoText').innerHTML = '<span class="loader"></span>Building Model...';

    const decay = parseFloat($('#f_auto_decay').value) || 0.92;
    const shrinkageK = parseFloat($('#f_auto_shrinkage').value) || 6.0;
    const homeAdvantage = parseFloat($('#f_auto_hadv').value) || 1.0;

    try {
        const url = `/api/betika/scrape?home_id=${encodeURIComponent(homeTeam)}&away_id=${encodeURIComponent(awayTeam)}&match_id=${encodeURIComponent(matchId)}&decay=${decay}&shrinkage_k=${shrinkageK}&home_advantage=${homeAdvantage}&match_date=${new Date().toISOString().slice(0, 10)}`;
        const res = await fetch(url);
        if (!res.ok) {
            const errText = await res.text();
            throw new Error(errText || 'Scraping request failed');
        }
        const record = await res.json();

        matches.push(record);
        selectedIndex = matches.length - 1;

        renderSidebar();
        renderMain();
        betikaModal.classList.remove('open');
        document.body.classList.remove('modal-open');
        showToast(`Model built for ${homeTeam} vs ${awayTeam}`, 'success');
    } catch (e) {
        errEl.textContent = 'Error: ' + e.message;
        errEl.style.display = 'block';
    } finally {
        isScraping = false;
        $('#btnAutoSubmit').disabled = false;
        $('#btnAutoText').textContent = 'Scrape & Build';
    }
}

/* ============================================================
   TEAM INFO MODAL
   ============================================================ */

const teamInfoModal = $('#teamInfoModalBackdrop');
let selectedTeamInfo = null;

function setupTeamInfoAutocomplete(inputEl, suggEl) {
    let debounceTimeout = null;

    inputEl.addEventListener('input', () => {
        clearTimeout(debounceTimeout);
        const query = inputEl.value.trim();
        if (query.length < 2) {
            suggEl.innerHTML = '';
            suggEl.style.display = 'none';
            return;
        }

        debounceTimeout = setTimeout(async () => {
            try {
                const res = await fetch(`/api/team/search?query=${encodeURIComponent(query)}`);
                if (!res.ok) return;
                const list = await res.json();

                suggEl.innerHTML = '';
                if (list.length === 0) {
                    suggEl.style.display = 'none';
                    return;
                }

                list.forEach(team => {
                    const item = document.createElement('div');
                    item.className = 'suggestion-item';
                    item.innerHTML = `<span>${escapeHtml(team.name)}</span><span class="subtitle">${escapeHtml(team.subtitle)}</span>`;
                    item.addEventListener('mousedown', (e) => {
                        inputEl.value = team.name;
                        loadTeamInfo(team.name);
                        suggEl.innerHTML = '';
                        suggEl.style.display = 'none';
                    });
                    suggEl.appendChild(item);
                });
                suggEl.style.display = 'block';
            } catch (e) { }
        }, 300);
    });

    inputEl.addEventListener('blur', () => {
        setTimeout(() => {
            suggEl.style.display = 'none';
        }, 200);
    });
}

async function loadTeamInfo(teamName) {
    if (!teamName) return;

    const content = $('#teamInfoContent');
    content.style.display = 'none';
    content.innerHTML = '<div style="padding: 20px; text-align: center;"><span class="loader" style="margin-right:8px;"></span> Loading team info...</div>';
    content.style.display = 'block';

    try {
        const res = await fetch(`/api/team/info?name=${encodeURIComponent(teamName)}`);
        if (!res.ok) throw new Error('Failed to load team info');
        const data = await res.json();
        selectedTeamInfo = data;
        renderTeamInfo(data);
    } catch (e) {
        content.innerHTML = `<div style="padding: 20px; text-align: center; color: var(--red);">Error: ${escapeHtml(e.message)}</div>`;
    }
}

function renderTeamInfo(info) {
    const content = $('#teamInfoContent');
    content.style.display = 'none';

    let detailsHtml = '';
    const detailFields = [
        ['Stadium', 'stadium'],
        ['Manager', 'manager'],
        ['Founded', 'founded'],
        ['League', 'league'],
        ['Position', 'position'],
        ['Capacity', 'capacity'],
    ];

    detailFields.forEach(([label, key]) => {
        if (info[key]) {
            detailsHtml += `<div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border); padding: 6px 0;">
        <span style="color: var(--text-dim);">${escapeHtml(label)}</span>
        <span style="color: var(--text); text-align: right; max-width: 60%;">${escapeHtml(info[key])}</span>
      </div>`;
        }
    });

    if (!detailsHtml) {
        detailsHtml = '<div style="color: var(--text-faint);">No detailed info available</div>';
    }

    $('#teamInfoOverview').textContent = info.summary || 'No summary available.';
    $('#teamInfoDetails').innerHTML = detailsHtml;
    const wikiUrl = (info.url || '').trim();
    const safeWikiUrl = /^https?:\/\//i.test(wikiUrl) ? escapeHtml(wikiUrl) : '';
    $('#teamInfoSource').innerHTML = safeWikiUrl
        ? `<a href="${safeWikiUrl}" target="_blank" rel="noopener noreferrer" style="color: var(--cyan); text-decoration: none;">View on Wikipedia</a>`
        : 'Wikipedia';

    content.style.display = 'block';
}

// Keyboard shortcuts: Escape closes modals, Enter submits forms
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        if (modalBackdrop.classList.contains('open')) closeModal();
        if (betikaModal.classList.contains('open')) { betikaModal.classList.remove('open'); document.body.classList.remove('modal-open'); }
        if (teamInfoModal.classList.contains('open')) { teamInfoModal.classList.remove('open'); document.body.classList.remove('modal-open'); }
        if (document.querySelector('.body')?.classList.contains('show-detail')) exitMobileDetailView();
    }
});

// Focus trapping for modals
function trapFocus(modalEl, event) {
    if (!modalEl.classList.contains('open')) return;
    const focusable = modalEl.querySelectorAll('button, input, select, textarea, [tabindex]:not([tabindex="-1"])');
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.key === 'Tab') {
        if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first.focus();
        }
    }
}

modalBackdrop.addEventListener('keydown', (e) => trapFocus(modalBackdrop, e));
betikaModal.addEventListener('keydown', (e) => trapFocus(betikaModal, e));
teamInfoModal.addEventListener('keydown', (e) => trapFocus(teamInfoModal, e));

// Enter key submits active modal form
modalBackdrop.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && modalBackdrop.classList.contains('open')) {
        const activeTab = document.querySelector('.modal-tab.active');
        if (activeTab && activeTab.id === 'tabAuto') {
            $('#btnAutoSubmit').click();
        } else if (activeTab && activeTab.id === 'tabManual') {
            $('#btnSubmit').click();
        }
    }
});

// Auto-focus first input when Add Match modal opens
const originalOpenModal = openModal;
openModal = function () {
    originalOpenModal();
    setTimeout(() => {
        const firstInput = modalBackdrop.querySelector('input:not([type=hidden]):not([type=file])');
        if (firstInput) firstInput.focus();
    }, 100);
};

// Auto-focus team search when Team Info modal opens
const originalTeamInfoOpen = () => teamInfoModal.classList.add('open');
$('#btnTeamInfo').addEventListener('click', () => {
    teamInfoModal.classList.add('open');
    document.body.classList.add('modal-open');
    $('#teamInfoContent').style.display = 'none';
    $('#teamInfoSearch').value = '';
    selectedTeamInfo = null;
    setTimeout(() => $('#teamInfoSearch').focus(), 100);
});

$('#btnTeamInfoClose').addEventListener('click', () => {
    teamInfoModal.classList.remove('open');
    document.body.classList.remove('modal-open');
});

teamInfoModal.addEventListener('click', e => {
    if (e.target === teamInfoModal) {
        teamInfoModal.classList.remove('open');
        document.body.classList.remove('modal-open');
    }
});

setupTeamInfoAutocomplete($('#teamInfoSearch'), $('#teamInfoSuggestions'));

/* ============================================================
   PREDICTIONS FEED LOGIC
   ============================================================ */

$('#tabPredFeed').addEventListener('click', () => {
    $('#tabPredFeed').classList.add('active');
    $('#tabMatchAnalyzer').classList.remove('active');
    $('#viewPredFeed').classList.add('active');
    $('#viewMatchAnalyzer').classList.remove('active');
    $('.topbar-actions').classList.remove('force-show');
});

$('#tabMatchAnalyzer').addEventListener('click', () => {
    $('#tabMatchAnalyzer').classList.add('active');
    $('#tabPredFeed').classList.remove('active');
    $('#viewMatchAnalyzer').classList.add('active');
    $('#viewPredFeed').classList.remove('active');
    $('.topbar-actions').classList.add('force-show');
});

// Hide topbar actions on load since Predictions Feed is default
// (handled purely via CSS now — see .topbar-actions default state)

$('#btnAutoScan').addEventListener('click', async () => {
    const btn = $('#btnAutoScan');
    const grid = $('#predictionsGrid');
    if (btn.disabled) return;

    btn.disabled = true;
    btn.innerHTML = '<span class="loader"></span> Scanning...';
    grid.innerHTML = `
                <div class="skeleton-card">
                    <div class="skeleton skeleton-title"></div>
                    <div class="skeleton skeleton-text medium"></div>
                    <div class="skeleton skeleton-text short"></div>
                </div>
                <div class="skeleton-card">
                    <div class="skeleton skeleton-title"></div>
                    <div class="skeleton skeleton-text medium"></div>
                    <div class="skeleton skeleton-text short"></div>
                </div>
                <div class="skeleton-card">
                    <div class="skeleton skeleton-title"></div>
                    <div class="skeleton skeleton-text medium"></div>
                    <div class="skeleton skeleton-text short"></div>
                </div>
            `;

    try {
        const res = await fetch('/api/predictions');
        if (!res.ok) throw new Error('Scan failed');
        const result = await res.json();

        if (result.data && result.data.length > 0) {
            renderPredictions(result.data);
        } else {
            grid.innerHTML = '<div class="empty-state">Scan complete. No active predictions found right now.</div>';
        }
    } catch (e) {
        grid.innerHTML = `<div class="empty-state" style="color:var(--red)">Failed to scan fixtures: ${escapeHtml(e.message)}</div>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Auto-Scan All Fixtures';
    }
});

/* ============================================================
   PREDICTIONS FEED HELPERS & INIT
   ============================================================ */

/**
 * Create a prediction card DOM element.
 * @param {Object} p - Prediction data.
 * @returns {HTMLElement} - Card element.
 */
function createPredictionCard(p) {
    const card = document.createElement('div');
    card.className = `pred-card tier-${p.confidence_tier || 'LEAN'}`;

    const edge = p.edge_pct || 0;
    const edgeColor = edge > 0 ? 'var(--cyan)' : 'var(--red)';
    const sign = edge > 0 ? '+' : '';

    card.innerHTML = `
        <div class="pred-main">
            <div class="pred-header">
                <div class="pred-teams">${escapeHtml(p.match_label || 'Match')}</div>
                <div class="pred-tier">${escapeHtml(p.confidence_tier || 'LEAN')}</div>
            </div>
            <div style="font-size:12px; color:var(--text-dim); margin-top:-6px;">
                ${escapeHtml(p.competition || '')} ${p.match_date ? '&middot; ' + escapeHtml(p.match_date) : ''}
            </div>
            
            <div class="pred-bet">BET: ${escapeHtml(p.recommended_bet || 'N/A')}</div>
            
            <div class="pred-stats">
                <div class="pred-stat-item">
                    <span class="pred-stat-label">Model Prob</span>
                    <span class="pred-stat-val">${((p.model_probability || 0) * 100).toFixed(1)}%</span>
                </div>
                <div class="pred-stat-item">
                    <span class="pred-stat-label">Market Prob</span>
                    <span class="pred-stat-val">${((p.market_implied_probability || 0) * 100).toFixed(1)}%</span>
                </div>
                <div class="pred-stat-item">
                    <span class="pred-stat-label">Edge</span>
                    <span class="pred-stat-val" style="color:${edgeColor}">${sign}${edge.toFixed(1)}%</span>
                </div>
                <div class="pred-stat-item">
                    <span class="pred-stat-label">Suggest Stake</span>
                    <span class="pred-stat-val">${(p.stake_suggestion_pct || 0).toFixed(1)}%</span>
                </div>
            </div>
            
            <div class="pred-narrative">
                ${escapeHtml(p.scouting_narrative || '')}
            </div>
            <div class="risk-banner">
                <strong>Risk Disclaimer:</strong> Statistical estimates based on live form and model consensus. Always practice sound bankroll management.
            </div>
        </div>
    `;
    return card;
}

/**
 * Render predictions grid.
 * @param {Array} predictions - Array of prediction objects.
 */
function renderPredictions(predictions) {
    const grid = $('#predictionsGrid');
    if (!grid) return;

    grid.innerHTML = '';

    if (!predictions || predictions.length === 0) {
        grid.innerHTML = '<div class="empty-state">No active predictions found right now.</div>';
        return;
    }

    predictions.forEach(p => {
        grid.appendChild(createPredictionCard(p));
    });
}

// Initial load check
checkServerStatus();
renderSidebar();
renderMain();

