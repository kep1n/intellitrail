// ── State ──────────────────────────────────────────────────────────────────
let currentThreadId = null;
let selectedGpxFile = null;
let abortController = null;
let currentAiEl = null;      // the currently streaming AI message bubble
let hasReceivedTokens = false;

// ── GPX attachment ──────────────────────────────────────────────────────────
const gpxFileInput = document.getElementById('gpx-file-input');
const gpxChip      = document.getElementById('gpx-chip');
const gpxChipName  = document.getElementById('gpx-chip-name');

gpxFileInput.addEventListener('change', () => {
  const file = gpxFileInput.files[0];
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.gpx')) {
    appendErrorBubble('Please attach a .gpx file.');
    return;
  }
  selectedGpxFile = file;
  gpxChipName.textContent = file.name;
  gpxChip.hidden = false;
});

document.getElementById('gpx-chip-remove').addEventListener('click', () => {
  selectedGpxFile = null;
  gpxFileInput.value = '';
  gpxChip.hidden = true;
});

// ── Voice recording ─────────────────────────────────────────────────────────
let mediaRecorder = null;
let audioChunks = [];
const voiceBtn    = document.getElementById('voice-btn');
const voiceStatus = document.getElementById('voice-status');

voiceBtn.addEventListener('mousedown',  e => { e.preventDefault(); startRecording(); });
voiceBtn.addEventListener('touchstart', e => { e.preventDefault(); startRecording(); });
voiceBtn.addEventListener('mouseup',    e => { e.preventDefault(); stopAndTranscribe(); });
voiceBtn.addEventListener('touchend',   e => { e.preventDefault(); stopAndTranscribe(); });

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.start();
    voiceBtn.classList.add('recording');
    voiceStatus.textContent = 'Recording…';
  } catch {
    voiceStatus.textContent = 'Microphone access denied';
  }
}

async function stopAndTranscribe() {
  if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
  voiceBtn.classList.remove('recording');
  voiceStatus.textContent = 'Transcribing…';
  return new Promise(resolve => {
    mediaRecorder.onstop = async () => {
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      const fd = new FormData();
      fd.append('audio', blob, 'recording.webm');
      try {
        const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: 'Transcription failed' }));
          voiceStatus.textContent = err.detail || 'Transcription failed';
          resolve(); return;
        }
        const { text } = await res.json();
        document.getElementById('message-input').value = text;
        voiceStatus.textContent = '';
        autoResizeTextarea();
      } catch {
        voiceStatus.textContent = 'Transcription request failed';
      }
      resolve();
    };
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
  });
}

// ── Send message ────────────────────────────────────────────────────────────
const messageInput = document.getElementById('message-input');
const sendBtn      = document.getElementById('send-btn');

messageInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});
messageInput.addEventListener('input', autoResizeTextarea);

function autoResizeTextarea() {
  messageInput.style.height = 'auto';
  messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

sendBtn.addEventListener('click', () => {
  const text = messageInput.value.trim();
  if (!text && !selectedGpxFile) return;

  const fd = new FormData();
  fd.append('message', text || '(GPX file attached)');
  if (currentThreadId) fd.append('thread_id', currentThreadId);
  if (selectedGpxFile) fd.append('gpx_file', selectedGpxFile);

  // Show user message in chat
  appendUserMessage(text, selectedGpxFile ? selectedGpxFile.name : null);

  // Clear input
  messageInput.value = '';
  messageInput.style.height = 'auto';
  selectedGpxFile = null;
  gpxFileInput.value = '';
  gpxChip.hidden = true;

  sendChat(fd);
});

// ── Chat API call ────────────────────────────────────────────────────────────
async function sendChat(formData) {
  setInputEnabled(false);
  abortController = new AbortController();

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      body: formData,
      signal: abortController.signal,
    });
    await consumeSseStream(response);
  } catch (e) {
    if (e.name !== 'AbortError') {
      appendErrorBubble('Connection error. Please try again.');
    }
  } finally {
    finaliseAiBubble();
    setInputEnabled(true);
    abortController = null;
  }
}

// ── SSE consumption ──────────────────────────────────────────────────────────
async function consumeSseStream(response) {
  const reader  = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const { events, remaining } = parseSseBuffer(buffer);
      buffer = remaining;
      for (const ev of events) handleSseEvent(ev.event, ev.data);
    }
  } catch (e) {
    if (e.name !== 'AbortError') throw e;
  }
}

function parseSseBuffer(buffer) {
  const events = [];
  const blocks = buffer.split('\n\n');
  const remaining = blocks.pop();
  for (const block of blocks) {
    if (!block.trim()) continue;
    const lines = block.split('\n');
    let event = 'message', data = '';
    for (const line of lines) {
      if (line.startsWith('event: ')) event = line.slice(7).trim();
      if (line.startsWith('data: '))  data  = line.slice(6).trim();
    }
    try { events.push({ event, data: JSON.parse(data) }); } catch {}
  }
  return { events, remaining };
}

function handleSseEvent(event, data) {
  switch (event) {
    case 'thread_id':
      currentThreadId = data.thread_id;
      break;

    case 'tool_start':
      ensureAiBubble();
      appendToolIndicator(data.tool);
      break;

    case 'token':
      ensureAiBubble();
      if (!hasReceivedTokens) {
        // First real token — clear tool indicators (tools are done)
        currentAiEl.querySelectorAll('.tool-indicator').forEach(el => el.remove());
        hasReceivedTokens = true;
      }
      currentAiEl.querySelector('.bubble-text').textContent += data.text || '';
      scrollToBottom();
      break;

    case 'result':
      if (data.verdict || data.report) updateResultsPanel(data);
      break;

    case 'error':
      finaliseAiBubble();
      appendErrorBubble(data.message || 'An error occurred.');
      break;

    case 'done':
      finaliseAiBubble();
      break;
  }
}

// ── Message bubble helpers ───────────────────────────────────────────────────
const messagesEl = document.getElementById('messages');

function appendUserMessage(text, gpxName) {
  const div = document.createElement('div');
  div.className = 'message user';
  let html = `<div class="bubble"><div class="bubble-text">${escapeHtml(text)}</div>`;
  if (gpxName) html += `<div class="gpx-tag">📎 ${escapeHtml(gpxName)}</div>`;
  html += '</div>';
  div.innerHTML = html;
  messagesEl.appendChild(div);
  scrollToBottom();
}

function ensureAiBubble() {
  if (currentAiEl) return;
  currentAiEl = document.createElement('div');
  currentAiEl.className = 'message ai';
  currentAiEl.innerHTML = '<div class="bubble"><div class="bubble-text"></div></div>';
  messagesEl.appendChild(currentAiEl);
  scrollToBottom();
}

function appendToolIndicator(toolName) {
  const labels = {
    analyze_route: '🏔️ Analyzing route conditions…',
    search_routes: '🔍 Searching for routes…',
    rag_query:     '📚 Searching knowledge base…',
  };
  const div = document.createElement('div');
  div.className = 'tool-indicator';
  div.textContent = labels[toolName] || `Running ${toolName}…`;
  currentAiEl.querySelector('.bubble-text').appendChild(div);
  scrollToBottom();
}

function finaliseAiBubble() {
  if (!currentAiEl) return;
  const textEl = currentAiEl.querySelector('.bubble-text');
  const raw = textEl.textContent;
  if (raw.trim()) {
    textEl.innerHTML = typeof marked !== 'undefined' ? marked.parse(raw) : escapeHtml(raw);
  }
  currentAiEl = null;
  hasReceivedTokens = false;
  scrollToBottom();
}

function appendErrorBubble(msg) {
  const div = document.createElement('div');
  div.className = 'message error';
  div.innerHTML = `<div class="bubble"><div class="bubble-text">${escapeHtml(msg)}</div></div>`;
  messagesEl.appendChild(div);
  scrollToBottom();
}

// ── Results panel ────────────────────────────────────────────────────────────
function updateResultsPanel(data) {
  document.getElementById('results-placeholder').hidden = true;
  document.getElementById('results-content').hidden = false;
  document.getElementById('results-panel').scrollTop = 0;

  const badge   = document.getElementById('verdict-badge');
  const verdict = data.verdict || '';
  badge.textContent = verdict;
  badge.className = 'badge';
  if (verdict === 'GO')      badge.classList.add('badge-go');
  else if (verdict === 'NO-GO')   badge.classList.add('badge-no-go');
  else if (verdict === 'CAUTION') badge.classList.add('badge-caution');

  const report    = data.report    || {};
  const trailInfo = data.trail_info || null;

  // Trail data
  const trailEl = document.getElementById('content-trail-data');
  trailEl.innerHTML = '';
  if (trailInfo && Object.values(trailInfo).some(v => v != null)) {
    const rows = [];
    if (trailInfo.track_name)          rows.push(['Track',         trailInfo.track_name]);
    if (trailInfo.trail_type)          rows.push(['Type',          trailInfo.trail_type]);
    if (trailInfo.difficulty)          rows.push(['Difficulty',    trailInfo.difficulty]);
    if (trailInfo.distance_km != null) rows.push(['Distance',      `${trailInfo.distance_km.toFixed(2)} km`]);
    if (trailInfo.elevation_gain_m != null) rows.push(['Gain',     `${Math.round(trailInfo.elevation_gain_m)} m`]);
    if (trailInfo.elevation_loss_m != null) rows.push(['Loss',     `${Math.round(trailInfo.elevation_loss_m)} m`]);
    if (trailInfo.max_elevation_m != null)  rows.push(['Max alt',  `${Math.round(trailInfo.max_elevation_m)} m`]);
    if (trailInfo.min_elevation_m != null)  rows.push(['Min alt',  `${Math.round(trailInfo.min_elevation_m)} m`]);
    if (trailInfo.moving_time)              rows.push(['Moving time', trailInfo.moving_time]);
    trailEl.innerHTML = rows.map(([label, value]) =>
      `<div class="trail-stat"><span class="trail-label">${label}</span><span class="trail-value">${value}</span></div>`
    ).join('');
    document.getElementById('section-trail-data').open = true;
  }

  // Risk factors
  const rfEl = document.getElementById('content-risk-factors');
  rfEl.innerHTML = '';
  if (report.risk_factors && report.risk_factors.length) {
    rfEl.innerHTML = '<ul>' + report.risk_factors.map(r => `<li>${escapeHtml(r)}</li>`).join('') + '</ul>';
    document.getElementById('section-risk-factors').open = true;
  }

  // Reasoning
  if (report.reasoning) {
    renderMarkdown(document.getElementById('content-reasoning'), report.reasoning);
    document.getElementById('section-reasoning').open = true;
  }

  // Time windows
  if (report.time_windows) renderMarkdown(document.getElementById('content-time-windows'), report.time_windows);

  // Elevation context
  if (report.elevation_context) renderMarkdown(document.getElementById('content-elevation'), report.elevation_context);

  // Alternatives
  const altEl   = document.getElementById('content-alternatives');
  const alts    = report.alternatives || [];
  altEl.innerHTML = '';
  if (alts.length) {
    const md = alts.map(a => {
      const dist = a.distance_km ? ` — ${a.distance_km.toFixed(1)} km` : '';
      return `- ${a.name}${dist}`;
    }).join('\n');
    renderMarkdown(altEl, md);
  } else if (verdict === 'CAUTION' || verdict === 'NO-GO') {
    altEl.textContent = 'No alternatives found within 10 km.';
  }

  // Physical difficulty
  const diff = report.physical_difficulty || {};
  if (diff.level) {
    document.getElementById('content-difficulty').textContent = `${diff.level} — ${diff.description}`;
    document.getElementById('section-difficulty').open = true;
  }

  // Estimated time
  const ht = report.hiking_time || {};
  if (ht.estimated_time_str) {
    let htText = `Estimated: ${ht.estimated_time_str}`;
    if (ht.sunset_time) htText += `\nSunset: ${ht.sunset_time}  |  Latest start: ${ht.latest_start_time}`;
    document.getElementById('content-hiking-time').textContent = htText;
    document.getElementById('section-hiking-time').open = true;
  }

  // Refuges & shelters
  const refs = report.refuges || [];
  const refEl = document.getElementById('content-refuges');
  refEl.innerHTML = '';
  if (refs.length) {
    const md = refs.map(r => `- **${r.name}** (${r.type.replace('_', ' ')}) — ${r.distance_km} km`).join('\n');
    renderMarkdown(refEl, md);
    document.getElementById('section-refuges').open = true;
  }
}

// ── Utilities ────────────────────────────────────────────────────────────────
function setInputEnabled(enabled) {
  sendBtn.disabled      = !enabled;
  messageInput.disabled = !enabled;
  voiceBtn.disabled     = !enabled;
  gpxFileInput.disabled = !enabled;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderMarkdown(el, text) {
  if (el && text) {
    el.innerHTML = typeof marked !== 'undefined' ? marked.parse(text) : escapeHtml(text);
  }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
