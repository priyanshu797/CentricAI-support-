// =====================================================================
// CentrixSupport — Chat + Welcome UI
// Two-state design:
//   1. Welcome state  — cards, quick-start chips, bottom input bar
//   2. Chat state     — full chat UI (appears after first message)
// =====================================================================

// ── DOM refs ──────────────────────────────────────────────────────────
const welcomeState    = document.getElementById("welcomeState");
const welcomeInputBar = document.getElementById("welcomeInputBar");
const welcomeForm     = document.getElementById("welcomeForm");
const welcomeInput    = document.getElementById("welcomeInput");
const micBtnWelcome   = document.getElementById("micBtnWelcome");
const micIconWelcome  = document.getElementById("micIconWelcome");

const chatMain        = document.getElementById("chatMain");
const chatBox         = document.getElementById("chatBox");
const chatForm        = document.getElementById("chatForm");
const userInput       = document.getElementById("userInput");
const micBtn          = document.getElementById("micBtn");
const micIcon         = document.getElementById("micIcon");

// upload – welcome mode
const uploadForm      = document.getElementById("uploadForm");
const fileInput       = document.getElementById("fileInput");
const uploadStatus    = document.getElementById("uploadStatus");
const fileList        = document.getElementById("fileList");

// upload – chat mode
const uploadFormChat  = document.getElementById("uploadFormChat");
const fileInputChat   = document.getElementById("fileInputChat");
const uploadStatusChat= document.getElementById("uploadStatusChat");
const fileListChat    = document.getElementById("fileListChat");

const historyList     = document.getElementById("historyList");
const historySidebar  = document.getElementById("historySidebar");

// ── State ─────────────────────────────────────────────────────────────
let uploadedFilePaths = [];
let activeDocumentId  = null;
let isSearching       = false;
let chatModeActive    = false;
let currentSessionId  = Date.now().toString();
let sessions          = JSON.parse(localStorage.getItem("chatSessions") || "{}");

// ── Emotion config ────────────────────────────────────────────────────
const EMOTION_CONFIG = {
  stressed:    { emoji: "😤", label: "Stressed",    class: "emotion-anxious"     },
  overwhelmed: { emoji: "😰", label: "Overwhelmed", class: "emotion-overwhelmed" },
  sad:         { emoji: "😢", label: "Sad",         class: "emotion-sad"         },
  angry:       { emoji: "😠", label: "Angry",       class: "emotion-angry"       },
  anxious:     { emoji: "😨", label: "Anxious",     class: "emotion-anxious"     },
  neutral:     { emoji: "😌", label: "Neutral",     class: "emotion-neutral"     },
  happy:       { emoji: "😊", label: "Happy",       class: "emotion-happy"       },
};

// Pill CSS class per emotion
const EMOTION_PILL_CLASS = {
  stressed:    "epill-stressed",
  anxious:     "epill-anxious",
  sad:         "epill-sad",
  angry:       "epill-angry",
  overwhelmed: "epill-overwhelmed",
  happy:       "epill-happy",
  neutral:     "epill-neutral",
};

// ── Two-state transition ──────────────────────────────────────────────
function activateChatMode() {
  if (chatModeActive) return;
  chatModeActive = true;

  // Hide welcome UI
  welcomeState.classList.add("hidden");
  welcomeInputBar.classList.add("hidden");

  // Show chat UI
  chatMain.classList.remove("chat-hidden");

  // Body overflow back to hidden (chat manages its own scroll)
  document.body.classList.remove("welcome-mode");

  // Sync upload state from welcome uploads into chat upload status
  if (uploadedFilePaths.length) {
    _renderChatUploadReady();
  }
}

function _renderChatUploadReady() {
  if (!uploadStatusChat) return;
  uploadStatusChat.innerHTML = `<div class="upload-ready"><span class="upload-complete-ring" aria-hidden="true">✓</span><span><strong>File ready.</strong> ${uploadedFilePaths.length} file(s) indexed.</span></div>`;
}

// ── Speech synthesis ─────────────────────────────────────────────────
let currentUtterance = null;
let isSpeaking       = false;
let femaleVoice      = null;
let currentSpeakBtn  = null;

function waitForVoices(ms = 2000) {
  return new Promise((resolve) => {
    const start = performance.now();
    const check = () => {
      const v = window.speechSynthesis.getVoices();
      if (v && v.length) return resolve(v);
      if (performance.now() - start > ms) return resolve(v || []);
      setTimeout(check, 50);
    };
    check();
  });
}

function pickFemaleVoice(voices) {
  const n = (s) => (s ? s.toLowerCase() : "");
  const cands = voices.filter(v =>
    n(v.name).includes("female") ||
    n(v.name).includes("woman") ||
    (v.lang && v.lang.toLowerCase().startsWith("en") && n(v.name).includes("google"))
  );
  if (cands.length) return cands[0];
  return voices.find(v => v.lang && v.lang.toLowerCase().startsWith("en")) || voices[0] || null;
}

async function initVoices() {
  femaleVoice = pickFemaleVoice(await waitForVoices());
}
if (window.speechSynthesis) {
  window.speechSynthesis.onvoiceschanged = initVoices;
  initVoices();
}

function resetSpeakUI() {
  isSpeaking = false;
  if (currentSpeakBtn) currentSpeakBtn.innerHTML = "🔊";
  currentSpeakBtn = null;
  currentUtterance = null;
}

function stopAllSpeech() {
  try { window.speechSynthesis.cancel(); } catch (_) {}
  resetSpeakUI();
}

function toggleSpeech(text, btn) {
  if (isSpeaking) { stopAllSpeech(); return; }
  stopAllSpeech();
  currentUtterance = new SpeechSynthesisUtterance(text);
  currentUtterance.voice = femaleVoice || null;
  currentUtterance.pitch = 1.1;
  currentUtterance.rate  = 1;
  currentUtterance.onend   = resetSpeakUI;
  currentUtterance.onerror = resetSpeakUI;
  try {
    window.speechSynthesis.speak(currentUtterance);
    isSpeaking = true;
    currentSpeakBtn = btn || null;
    if (currentSpeakBtn) currentSpeakBtn.innerHTML = "⏹️";
  } catch (_) { resetSpeakUI(); }
}
window.addEventListener("beforeunload", stopAllSpeech);

// ── Emotion display ───────────────────────────────────────────────────
/**
 * Appends a small inline emotion pill below a bot bubble.
 * Shown only for non-neutral emotions.
 */
function appendEmotionPill(parentEl, emotion) {
  const cfg = EMOTION_CONFIG[emotion];
  if (!cfg || emotion === "neutral") return;
  const pillCls = EMOTION_PILL_CLASS[emotion] || "epill-neutral";
  const pill = document.createElement("div");
  pill.className = `emotion-inline-pill ${pillCls}`;
  pill.innerHTML = `<span>${cfg.emoji}</span><span>${cfg.label}</span>`;
  parentEl.appendChild(pill);
}

// ── Utilities ─────────────────────────────────────────────────────────
function addTimestamp() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function focusInput() {
  const el = chatModeActive ? userInput : welcomeInput;
  if (!el || el.disabled) return;
  setTimeout(() => { el.focus(); const end = el.value.length; el.setSelectionRange(end, end); }, 0);
}

function escapeHtml(v) {
  return String(v ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function compactFileName(v, max = 38) {
  const full = String(v || "Document").split(/[\\/]/).pop();
  const clean = full.replace(/_\d{13,}(?=\.[^.]+$)/, "");
  if (clean.length <= max) return clean;
  const dot = clean.lastIndexOf(".");
  const ext  = dot > 0 ? clean.slice(dot) : "";
  const base = dot > 0 ? clean.slice(0, dot) : clean;
  const avail = Math.max(12, max - ext.length - 1);
  return `${base.slice(0, avail)}…${ext}`;
}

function formatInlineMarkdown(v) {
  return escapeHtml(v)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_]+)__/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*(?=$|[\s).,!?:;])/g, "$1<em>$2</em>");
}

// ── Markdown formatter ────────────────────────────────────────────────
function formatBotResponse(text) {
  // Normalise line endings and unescape literal \n sequences
  const raw = String(text ?? "")
    .replace(/\\n/g, "\n")
    .replace(/\r\n?/g, "\n")
    .trim();

  // ── Pass 1: collect raw lines, group consecutive table lines ──────
  // We do a single pre-pass to handle <br> inside table cells and ---
  const lines = raw.split("\n");
  const html  = [];
  let paragraph   = [];
  let listType    = null;
  let inCodeBlock = false;
  let codeLines   = [];
  let tableLines  = [];   // buffer for consecutive | … | lines

  // ── Helpers ────────────────────────────────────────────────────────
  const flushP = () => {
    if (!paragraph.length) return;
    html.push(`<p>${paragraph.map(formatInlineMarkdown).join("<br>")}</p>`);
    paragraph = [];
  };
  const closeL = () => {
    if (!listType) return;
    html.push(`</${listType}>`);
    listType = null;
  };
  const openL = (t) => {
    flushP();
    if (listType === t) return;
    closeL();
    listType = t;
    html.push(`<${t}>`);
  };

  // Render a buffered group of Markdown table lines as an HTML <table>
  const flushTable = () => {
    if (!tableLines.length) return;
    const rows = tableLines;
    tableLines = [];

    // Split a pipe-delimited row into cells, trimming surrounding |
    const splitRow = (r) =>
      r.replace(/^[|]/, "").replace(/[|]$/, "").split("|").map(c => c.trim());

    // Detect separator row: cells look like ---, :---, :---:, etc.
    const isSep = (r) => splitRow(r).every(c => /^:?-+:?$/.test(c) || c === "");

    let headerRow = null;
    const bodyRows = [];
    let seenSep = false;

    for (const r of rows) {
      if (isSep(r)) { seenSep = true; continue; }
      if (!seenSep && headerRow === null) { headerRow = r; continue; }
      bodyRows.push(r);
    }
    // If no separator was found, treat all rows as body
    if (!seenSep) {
      if (headerRow !== null) bodyRows.unshift(headerRow);
      headerRow = null;
    }

    // Cell content: convert literal <br> / <br/> to newlines rendered
    // as <br> inside the cell, and apply inline markdown
    const cellHtml = (c) => {
      // Replace LLM-emitted literal <br> tags with a real line-break marker
      const cleaned = c
        .replace(/<br\s*\/?>/gi, "\n")  // literal <br> → real \n
        .trim();
      // Format each sub-line with inline markdown and join with <br>
      return cleaned
        .split("\n")
        .map(l => formatInlineMarkdown(l.trim()))
        .filter(l => l)
        .join("<br>");
    };

    let t = '<div class="table-wrap"><table>';
    if (headerRow !== null) {
      t += "<thead><tr>";
      for (const c of splitRow(headerRow)) t += `<th>${cellHtml(c)}</th>`;
      t += "</tr></thead>";
    }
    if (bodyRows.length) {
      t += "<tbody>";
      for (const r of bodyRows) {
        t += "<tr>";
        for (const c of splitRow(r)) t += `<td>${cellHtml(c)}</td>`;
        t += "</tr>";
      }
      t += "</tbody>";
    }
    t += "</table></div>";
    html.push(t);
  };

  // ── Pass 2: line-by-line processing ───────────────────────────────
  for (const rawLine of lines) {
    const line    = rawLine.trimEnd();
    const trimmed = line.trim();

    // ── Code fence ────────────────────────────────────────────────────
    if (/^```/.test(trimmed)) {
      flushTable(); flushP(); closeL();
      if (inCodeBlock) {
        html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = [];
      }
      inCodeBlock = !inCodeBlock;
      continue;
    }
    if (inCodeBlock) { codeLines.push(rawLine); continue; }

    // ── Table row ─────────────────────────────────────────────────────
    if (/^\|/.test(trimmed)) {
      flushP(); closeL();
      tableLines.push(trimmed);
      continue;
    }
    // Leaving a table block
    if (tableLines.length) flushTable();

    // ── Empty line ────────────────────────────────────────────────────
    if (!trimmed) { flushP(); closeL(); continue; }

    // ── Horizontal rule: ---, ***, ___  ──────────────────────────────
    if (/^[-*_]{3,}$/.test(trimmed)) {
      flushP(); closeL();
      html.push("<hr>");
      continue;
    }

    // ── ATX Headings ─────────────────────────────────────────────────
    const hd = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (hd) {
      flushP(); closeL();
      const lv = Math.min(hd[1].length + 1, 4);
      html.push(`<h${lv}>${formatInlineMarkdown(hd[2])}</h${lv}>`);
      continue;
    }

    // ── Implicit heading: short text ending in colon ──────────────────
    if (/^[^.!?]{2,64}:$/.test(trimmed) && !/^https?:/i.test(trimmed)) {
      flushP(); closeL();
      html.push(`<h3>${formatInlineMarkdown(trimmed.slice(0, -1))}</h3>`);
      continue;
    }

    // ── Numbered heading: "1. Title (context)" ───────────────────────
    const numHd = trimmed.match(/^(\d+)\.\s+(.{4,80})$/);
    if (numHd && trimmed === trimmed && !trimmed.match(/^\d+\.\s+\S.*\s\S/)) {
      // Only treat as heading if it's a short title-like line
    }

    // ── Unordered list ────────────────────────────────────────────────
    const ul = trimmed.match(/^[-*•]\s+(.+)$/);
    if (ul) { openL("ul"); html.push(`<li>${formatInlineMarkdown(ul[1])}</li>`); continue; }

    // ── Ordered list ─────────────────────────────────────────────────
    const ol = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (ol) { openL("ol"); html.push(`<li>${formatInlineMarkdown(ol[1])}</li>`); continue; }

    // ── Blockquote ───────────────────────────────────────────────────
    const bq = trimmed.match(/^>\s?(.+)$/);
    if (bq) { flushP(); closeL(); html.push(`<blockquote>${formatInlineMarkdown(bq[1])}</blockquote>`); continue; }

    // ── Plain paragraph text ─────────────────────────────────────────
    closeL();
    // Replace literal <br> the LLM sometimes emits in prose
    const cleaned = trimmed.replace(/<br\s*\/?>/gi, "\n");
    if (cleaned.includes("\n")) {
      flushP();
      cleaned.split("\n").forEach(l => { if (l.trim()) paragraph.push(l.trim()); });
    } else {
      paragraph.push(cleaned);
    }
  }

  // Flush any remaining buffers
  if (tableLines.length) flushTable();
  flushP();
  closeL();
  if (inCodeBlock && codeLines.length) {
    html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  }

  return html.join("");
}

// ── Session helpers ───────────────────────────────────────────────────
function saveMessage(sessionId, role, text, emotion = null) {
  if (!sessions[sessionId]) sessions[sessionId] = { messages: [], created: new Date().toISOString() };
  sessions[sessionId].messages.push({ role, text, time: addTimestamp(), emotion });
  localStorage.setItem("chatSessions", JSON.stringify(sessions));
  localStorage.setItem("currentSessionId", sessionId);
  renderHistory();
}

function renderHistory() {
  if (!historyList) return;
  historyList.innerHTML = "";
  const sorted = Object.entries(sessions).sort((a, b) => new Date(b[1].created || 0) - new Date(a[1].created || 0));
  sorted.forEach(([sid, session]) => {
    const container = document.createElement("div");
    container.className = "relative group";
    const first = session.messages.find(m => m.role === "user");
    const label = first ? `${first.text.slice(0, 30)}…` : "New Chat";
    const div = document.createElement("div");
    div.style.cssText = "font-size:13px;background:#0f766e;color:white;padding:8px 32px 8px 8px;border-radius:8px;margin-bottom:8px;cursor:pointer;";
    div.textContent = `${label} [${session.messages[0]?.time || "now"}]`;
    div.onclick = () => loadSession(sid);
    const del = document.createElement("button");
    del.innerHTML = "🗑️";
    del.style.cssText = "position:absolute;right:6px;top:50%;transform:translateY(-50%);background:rgba(239,68,68,0.9);border:none;border-radius:4px;padding:3px 7px;cursor:pointer;";
    del.onclick = (e) => {
      e.stopPropagation();
      if (!confirm("Delete this chat?")) return;
      delete sessions[sid];
      localStorage.setItem("chatSessions", JSON.stringify(sessions));
      if (currentSessionId === sid) { currentSessionId = Date.now().toString(); sessions[currentSessionId] = { messages: [], created: new Date().toISOString() }; localStorage.setItem("chatSessions", JSON.stringify(sessions)); chatBox.innerHTML = ""; addBotMessage("Hi there! 👋 How can I support you today?"); }
      renderHistory();
    };
    container.appendChild(div); container.appendChild(del); historyList.appendChild(container);
  });
}

function loadSession(sid) {
  currentSessionId = sid;
  localStorage.setItem("currentSessionId", sid);
  activateChatMode();
  chatBox.innerHTML = "";
  const session = sessions[sid];
  if (session?.messages?.length) {
    session.messages.forEach(m => addMessage(m.text, m.role === "user", m.time, m.emotion));
  } else {
    addBotMessage("Hi there! 👋 How can I support you today?");
  }
  focusInput();
}

// ── Message rendering ─────────────────────────────────────────────────
function addMessage(text, isUser = false, time = null, emotion = null, hasFiles = false) {
  const wrapper = document.createElement("div");
  wrapper.className = isUser ? "text-right" : "text-left";
  const bubble = document.createElement("div");
  const isCrisis = text.includes("🚨") || text.toLowerCase().includes("crisis");
  bubble.className = `inline-block ${isUser ? "user-message" : "bot-message"} ${isCrisis ? "crisis-alert" : ""} animate-fade-in`;
  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";
  if (isUser) contentDiv.textContent = text;
  else        contentDiv.innerHTML = formatBotResponse(text);
  bubble.appendChild(contentDiv);
  if (hasFiles && !isUser) {
    const fi = document.createElement("div");
    fi.className = "file-indicator";
    fi.innerHTML = "🔍 Response based on uploaded files";
    bubble.appendChild(fi);
  }
  if (!isUser && emotion) appendEmotionPill(bubble, emotion);
  const footer = document.createElement("div");
  footer.className = "message-footer";
  const ts = document.createElement("span"); ts.className = "timestamp"; ts.textContent = time || addTimestamp();
  const cpBtn = document.createElement("button"); cpBtn.className = "icon-btn"; cpBtn.innerHTML = "📋"; cpBtn.title = "Copy";
  cpBtn.onclick = () => { navigator.clipboard.writeText(text.replace(/<[^>]*>/g, "")).then(() => { cpBtn.innerHTML = "✅"; setTimeout(() => (cpBtn.innerHTML = "📋"), 1000); }); };
  const spBtn = document.createElement("button"); spBtn.className = "icon-btn"; spBtn.innerHTML = "🔊"; spBtn.title = "Read aloud";
  spBtn.onclick = () => toggleSpeech(text.replace(/<[^>]*>/g, ""), spBtn);
  footer.append(ts, cpBtn, spBtn);
  bubble.appendChild(footer);
  wrapper.appendChild(bubble);
  chatBox.appendChild(wrapper);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function addBotMessage(text) { addMessage(text, false); }

function addTypingBubble() {
  const w = document.createElement("div"); w.className = "text-left"; w.id = "typingIndicator";
  const b = document.createElement("div"); b.className = "inline-block bot-message animate-fade-in";
  const l = document.createElement("div"); l.className = "typing-loader mt-1"; l.innerHTML = "<span></span><span></span><span></span>";
  b.appendChild(l); w.appendChild(b); chatBox.appendChild(w); chatBox.scrollTop = chatBox.scrollHeight;
  return w;
}

function removeTypingBubble() { const t = document.getElementById("typingIndicator"); if (t) t.remove(); }

// ── Streaming bubble ──────────────────────────────────────────────────
function responseTypeLabel(toolUsed, ragUsed) {
  const map = { calculator:"Calculated answer", web_search:"Web-assisted answer", "llm+web":"Web-assisted answer", docs:"Document answer", cache:"Saved answer", crisis:"Priority support" };
  return map[toolUsed] || (ragUsed ? "Document answer" : "AI answer");
}

function createStreamingBubble(toolUsed, ragUsed) {
  const wrapper = document.createElement("div"); wrapper.className = "text-left streaming-wrapper";
  const bubble  = document.createElement("div"); bubble.className = "inline-block bot-message animate-fade-in streaming-active";
  const label   = document.createElement("div"); label.className = "response-label"; label.textContent = responseTypeLabel(toolUsed, ragUsed);
  const contentEl = document.createElement("div"); contentEl.className = "message-content stream-content";
  const cursor  = document.createElement("span"); cursor.className = "stream-cursor"; cursor.setAttribute("aria-hidden","true");
  bubble.append(label, contentEl, cursor); wrapper.appendChild(bubble); chatBox.appendChild(wrapper); chatBox.scrollTop = chatBox.scrollHeight;

  let _plain = "", _displayed = "", tokenQueue = [], pumpTimer = null, drainResolvers = [];

  function renderBuf() { contentEl.innerHTML = formatBotResponse(_displayed); bubble.appendChild(cursor); chatBox.scrollTop = chatBox.scrollHeight; }
  function resolveDrain() { if (tokenQueue.length || pumpTimer) return; const r = drainResolvers.splice(0); r.forEach(fn => fn()); }
  function pumpTokens() {
    pumpTimer = null;
    const burst = tokenQueue.length > 120 ? 8 : tokenQueue.length > 40 ? 4 : 1;
    for (let i = 0; i < burst && tokenQueue.length; i++) _displayed += tokenQueue.shift();
    renderBuf();
    if (tokenQueue.length) pumpTimer = setTimeout(pumpTokens, 18); else resolveDrain();
  }
  function appendChunk(text) {
    if (!text) return;
    _plain += text;
    tokenQueue.push(...(text.match(/\s+|[^\s]+\s*/g) || [text]));
    if (!pumpTimer) pumpTimer = setTimeout(pumpTokens, 0);
  }
  function waitForDrain() { if (!tokenQueue.length && !pumpTimer) return Promise.resolve(); return new Promise(r => drainResolvers.push(r)); }

  async function finalize(sources, emotion, hasFiles, intensity, suggestions, confidence) {
    await waitForDrain();
    _displayed = _plain; renderBuf(); cursor.remove(); bubble.classList.remove("streaming-active");

    if (hasFiles) {
      const fi = document.createElement("div");
      fi.className = "file-indicator";
      fi.innerHTML = "🔍 Response based on uploaded files";
      bubble.appendChild(fi);
    }

    // Small inline emotion pill — appended after response content
    if (emotion && emotion !== "neutral") appendEmotionPill(bubble, emotion);

    const footer = document.createElement("div"); footer.className = "message-footer";
    const ts = document.createElement("span"); ts.className = "timestamp"; ts.textContent = addTimestamp();
    const cpBtn = document.createElement("button"); cpBtn.className = "icon-btn"; cpBtn.innerHTML = "📋"; cpBtn.title = "Copy";
    cpBtn.onclick = () => navigator.clipboard.writeText(_plain).then(() => { cpBtn.innerHTML = "✅"; setTimeout(() => (cpBtn.innerHTML = "📋"), 1000); });
    const spBtn = document.createElement("button"); spBtn.className = "icon-btn"; spBtn.innerHTML = "🔊"; spBtn.title = "Read aloud";
    spBtn.onclick = () => toggleSpeech(_plain, spBtn);
    footer.append(ts, cpBtn, spBtn); bubble.appendChild(footer);
    chatBox.scrollTop = chatBox.scrollHeight;
    return _plain;
  }
  return { wrapper, contentEl, cursor, appendChunk, finalize, bubbleEl: bubble };
}

// ── Streaming pipeline ────────────────────────────────────────────────
function setInputDisabled(disabled) {
  if (chatForm)  { const sb = chatForm.querySelector("button[type='submit']"); if (sb) sb.disabled = disabled; }
  if (userInput) userInput.disabled = disabled;
  if (!disabled) focusInput();
}

function startStream(query, attachFiles) {
  isSearching = true;
  setInputDisabled(true);
  const typingBubble = addTypingBubble();
  let firstToken      = true;
  let streamBubble    = null;
  let metaEmotion     = null;
  let metaIntensity   = null;
  let metaSuggestions = null;
  let metaConfidence  = null;
  let metaRagUsed     = false;
  let metaToolUsed    = "llm";
  let latestInsights  = null;   // populated by the "insights" SSE frame
  let finished        = false;

  const params = new URLSearchParams({ q: query, session_name: currentSessionId });
  if (activeDocumentId) params.set("document_id", activeDocumentId);
  const es = new EventSource(`/search/stream?${params}`);

  function cleanup() { if (finished) return; finished = true; es.close(); isSearching = false; setInputDisabled(false); }

  es.onmessage = async (event) => {
    let frame; try { frame = JSON.parse(event.data); } catch { return; }

    if (frame.type === "meta") {
      metaEmotion     = frame.emotion_detected   || null;
      metaConfidence  = frame.emotion_confidence != null ? frame.emotion_confidence : null;
      metaIntensity   = frame.emotion_intensity  || null;
      metaSuggestions = frame.emotion_suggestions || null;
      metaRagUsed     = frame.rag_used   || false;
      metaToolUsed    = frame.tool_used  || "llm";
      return;
    }

    // ping keeps the SSE connection alive during long multi-tool agent runs
    if (frame.type === "ping") { return; }

    // tool_progress shows which tool the agent is currently running —
    // creates the bubble early so the user sees activity, not a frozen spinner
    if (frame.type === "tool_progress") {
      if (firstToken) {
        removeTypingBubble();
        streamBubble = createStreamingBubble(metaToolUsed, metaRagUsed);
        firstToken = false;
      }
      const toolLabels = {
        analyze_emotion:       "Analysing emotion…",
        search_knowledge:      "Searching documents…",
        web_search:            "Searching the web…",
        create_self_care_plan: "Building your plan…",
        get_tasks:             "Fetching tasks…",
        create_task:           "Creating task…",
        log_mood:              "Logging mood…",
        get_mood_history:      "Loading mood history…",
        start_breathing_exercise: "Preparing exercise…",
        summarize_document:    "Summarising document…",
      };
      const label = toolLabels[frame.tool] || `${frame.tool}…`;
      streamBubble.appendChunk(`\n_${label}_\n`);
      return;
    }

    if (frame.type === "chunk" || frame.type === "token") {
      if (firstToken) {
        removeTypingBubble();
        streamBubble = createStreamingBubble(metaToolUsed, metaRagUsed);
        firstToken = false;
      }
      streamBubble.appendChunk(frame.text || "");
      return;
    }

    // ── insights frame (new) — arrives after done ────────────────────
    if (frame.type === "insights") {
      latestInsights = frame.data || null;
      // If bubble already finalised, attach the button now
      if (streamBubble && streamBubble.bubbleEl) {
        attachInsightsButton(streamBubble.bubbleEl, latestInsights);
      }
      return;
    }

    if (frame.type === "done") {
      es.close(); removeTypingBubble();
      if (streamBubble) {
        const ft = await streamBubble.finalize(
          frame.sources || [], metaEmotion, attachFiles && metaRagUsed,
          metaIntensity, metaSuggestions, metaConfidence
        );
        // Attach insights button if data already arrived; otherwise it will
        // be attached when the insights frame arrives (above).
        if (latestInsights && streamBubble.bubbleEl) {
          attachInsightsButton(streamBubble.bubbleEl, latestInsights);
        }
        saveMessage(currentSessionId, "bot", ft, metaEmotion);
      } else {
        addBotMessage("⚠️ No response received. Please try again.");
      }
      cleanup();
      return;
    }

    if (frame.type === "error") {
      es.close(); removeTypingBubble();
      if (streamBubble) {
        const pt = await streamBubble.finalize([], metaEmotion, false, null, null, null);
        if (pt) saveMessage(currentSessionId, "bot", pt, metaEmotion);
        addBotMessage("⚠️ Response was interrupted. Please try again.");
      } else {
        addBotMessage(`❌ ${frame.message || "An error occurred"}`);
      }
      cleanup();
    }
  };

  es.onerror = async () => {
    if (finished) return; removeTypingBubble();
    if (!streamBubble) {
      addBotMessage("❌ Connection lost. Please try again.");
    } else {
      const pt = await streamBubble.finalize([], metaEmotion, false, null, null, null);
      if (pt) saveMessage(currentSessionId, "bot", pt, metaEmotion);
      addBotMessage("⚠️ Connection lost. Please try again.");
    }
    cleanup();
  };
}

// ── Shared send logic ─────────────────────────────────────────────────
function handleSend(query) {
  if (!query || isSearching) return;
  activateChatMode();

  addMessage(query, true);
  saveMessage(currentSessionId, "user", query);

  startStream(query, Boolean(activeDocumentId));
}

// ── Welcome form (bottom input bar) ──────────────────────────────────
if (welcomeForm) {
  welcomeForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = welcomeInput.value.trim();
    welcomeInput.value = "";
    handleSend(q);
  });
}

// ── Chat form ─────────────────────────────────────────────────────────
if (chatForm) {
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = userInput.value.trim();
    userInput.value = "";
    handleSend(q);
  });
}

// ── Category card clicks ──────────────────────────────────────────────
document.querySelectorAll(".cat-card").forEach(btn => {
  btn.addEventListener("click", () => {
    const prompt = btn.dataset.prompt;
    if (prompt) handleSend(prompt);
  });
});

// ── Quick-start chip clicks ───────────────────────────────────────────
document.querySelectorAll(".qs-chip").forEach(btn => {
  btn.addEventListener("click", () => {
    const prompt = btn.dataset.prompt;
    if (prompt) {
      // Pre-fill the welcome input for user to see/edit, then send
      if (welcomeInput) welcomeInput.value = prompt;
      handleSend(prompt);
    }
  });
});

// ── New chat button ───────────────────────────────────────────────────
const newChatBtn = document.getElementById("newChat");
if (newChatBtn) {
  newChatBtn.addEventListener("click", () => {
    currentSessionId = Date.now().toString();
    sessions[currentSessionId] = { messages: [], created: new Date().toISOString() };
    localStorage.setItem("chatSessions", JSON.stringify(sessions));
    chatBox.innerHTML = "";
    addBotMessage("Hi there! 👋 How can I support you today?");
    uploadedFilePaths = []; activeDocumentId = null;
    if (fileInput) fileInput.value = "";
    if (uploadStatus) uploadStatus.innerHTML = "";
    if (fileList) fileList.innerHTML = "";
    if (fileInputChat) fileInputChat.value = "";
    if (uploadStatusChat) uploadStatusChat.innerHTML = "";
    if (fileListChat) fileListChat.innerHTML = "";
    renderHistory(); focusInput();
  });
}

// ── Sidebar toggle ────────────────────────────────────────────────────
document.querySelectorAll("#toggleHistory, #toggleHistoryBtn").forEach(btn => {
  btn.addEventListener("click", () => {
    const hidden = historySidebar.classList.toggle("hidden");
    chatMain.classList.toggle("with-sidebar", !hidden);
  });
});

// ── Upload helper (shared between welcome and chat upload forms) ───────
function _buildUploadHandler(form, fileInputEl, statusEl, listEl, isChatMode) {
  if (!form) return;
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const files = fileInputEl.files;
    if (!files.length) { alert("Please select a file first."); return; }

    statusEl.innerHTML = `<div class="upload-progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><span>0%</span></div><span class="upload-progress-label">Uploading…</span>`;
    const formData = new FormData();
    for (const f of files) formData.append("file", f);

    const submitBtn = form.querySelector("button[type='submit']");
    const chatSendBtn = chatForm ? chatForm.querySelector("button[type='submit']") : null;
    const disable = () => { if (submitBtn) submitBtn.disabled = true; if (userInput) userInput.disabled = true; if (welcomeInput) welcomeInput.disabled = true; if (chatSendBtn) chatSendBtn.disabled = true; };
    const enable  = () => { if (submitBtn) submitBtn.disabled = false; if (userInput) userInput.disabled = false; if (welcomeInput) welcomeInput.disabled = false; if (chatSendBtn) chatSendBtn.disabled = false; focusInput(); };
    const showFail = (msg) => { enable(); statusEl.innerHTML = `<p style="font-size:13px;color:#dc2626;">❌ ${escapeHtml(msg)}</p>`; };

    disable();
    const progressBar  = statusEl.querySelector(".upload-progress");
    const progressText = progressBar.querySelector("span");

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload", true);
    xhr.upload.onprogress = (ev) => {
      if (!ev.lengthComputable) return;
      const pct = Math.round(ev.loaded / ev.total * 100);
      progressBar.style.setProperty("--upload-progress", `${pct * 3.6}deg`);
      progressBar.setAttribute("aria-valuenow", String(pct));
      progressText.textContent = `${pct}%`;
    };
    xhr.upload.onload = () => { progressBar.style.setProperty("--upload-progress","360deg"); progressText.textContent = "…"; const lb = statusEl.querySelector(".upload-progress-label"); if (lb) lb.textContent = "Processing…"; };

    xhr.onload = function () {
      if (xhr.status !== 202) { let msg = `Upload failed (${xhr.status}).`; try { const r = JSON.parse(xhr.responseText || "{}"); if (r.error) msg = r.error; } catch (_) {} showFail(msg); return; }
      let result = {}; try { result = JSON.parse(xhr.responseText || "{}"); } catch (_) { showFail("Invalid server response."); return; }
      const pendingPaths = result.filepaths || [], pendingFiles = result.files || [], docId = result.document_id || "";
      if (!docId) { showFail("No indexing job ID returned."); return; }

      const poll = async () => {
        try {
          const res = await fetch(`/documents/${encodeURIComponent(docId)}/status`, { cache: "no-store" });
          const st  = await res.json();
          if (!res.ok || !st.success) throw new Error(st.error || "Status check failed");
          const pct = Math.max(0, Math.min(100, Number(st.progress) || 0));
          progressBar.style.setProperty("--upload-progress", `${pct * 3.6}deg`);
          progressBar.setAttribute("aria-valuenow", String(pct));
          progressText.textContent = `${pct}%`;
          const lb = statusEl.querySelector(".upload-progress-label"); if (lb) lb.textContent = `${st.status}…`;
          if (st.status === "Indexed") {
            uploadedFilePaths = pendingPaths; activeDocumentId = docId;
            enable(); fileInputEl.value = "";
            statusEl.innerHTML = `<div class="upload-ready"><span class="upload-complete-ring" aria-hidden="true">✓</span><span><strong>Upload and indexing complete.</strong><br>${pendingPaths.length} file(s) are ready for fast retrieval.</span></div>`;
            const af = pendingFiles.length ? pendingFiles : pendingPaths.map(p => ({ name: p, url: "" }));
            listEl.innerHTML = af.map(f => {
              const fullName = String(f.original_name || f.name || "Document").split(/[\\/]/).pop();
              const shortName = compactFileName(fullName);
              const inner = `<span aria-hidden="true">📎</span><span><strong>Current:</strong> ${escapeHtml(shortName)}</span>`;
              const content = f.url ? `<a href="${escapeHtml(f.url)}" target="_blank" rel="noopener">${inner}</a>` : inner;
              return `<li class="active-document-item" title="${escapeHtml(fullName)}">${content}</li>`;
            }).join("");
            // If in chat mode, also sync the other status display
            if (isChatMode && uploadStatus) uploadStatus.innerHTML = statusEl.innerHTML;
            if (!isChatMode && uploadStatusChat) _renderChatUploadReady();
            return;
          }
          if (st.status === "Failed") { showFail(st.error || "Indexing failed."); return; }
          setTimeout(poll, 500);
        } catch (err) { showFail(err.message || "Status check failed."); }
      };
      poll();
    };
    xhr.onerror = () => { enable(); statusEl.innerHTML = "<p style='font-size:13px;color:#dc2626;'>❌ Network error.</p>"; };
    xhr.send(formData);
  });
}

_buildUploadHandler(uploadForm,     fileInput,     uploadStatus,     fileList,     false);
_buildUploadHandler(uploadFormChat, fileInputChat, uploadStatusChat, fileListChat, true);

// ── Speech recognition ─────────────────────────────────────────────────
function _buildMicHandler(btn, iconEl, inputEl) {
  if (!btn || !("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) { if (btn) btn.style.display = "none"; return; }
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const rec = new SR(); rec.lang = "en-US"; rec.continuous = false; rec.interimResults = false;
  let listening = false;
  btn.addEventListener("click", () => {
    if (!listening) {
      try { rec.start(); if (iconEl) iconEl.src = "https://img.icons8.com/fluency/24/stop-squared.png"; btn.style.background = "#dc2626"; listening = true; }
      catch (e) { console.error(e); }
    } else { rec.stop(); if (iconEl) iconEl.src = "https://img.icons8.com/material-sharp/24/microphone--v1.png"; btn.style.background = "transparent"; listening = false; }
  });
  rec.onresult = (ev) => { if (inputEl) inputEl.value = ev.results[0][0].transcript; };
  rec.onend = () => { if (iconEl) iconEl.src = "https://img.icons8.com/material-sharp/24/microphone--v1.png"; btn.style.background = "transparent"; listening = false; };
  rec.onerror = () => { if (iconEl) iconEl.src = "https://img.icons8.com/material-sharp/24/microphone--v1.png"; btn.style.background = "transparent"; listening = false; };
}

_buildMicHandler(micBtn,        micIcon,        userInput);
_buildMicHandler(micBtnWelcome, micIconWelcome, welcomeInput);

// ── Global onclick handlers (static HTML messages) ────────────────────
window.copyMessage = function(btn) {
  const bub = btn.closest(".bot-message, .user-message");
  const el  = bub ? bub.querySelector(".message-content") : null;
  const text = (el || bub) ? (el || bub).innerText : "";
  navigator.clipboard.writeText(text).then(() => { btn.innerHTML = "✅"; setTimeout(() => (btn.innerHTML = "📋"), 1000); }).catch(console.error);
};
window.speakMessage = function(btn) {
  const bub = btn.closest(".bot-message, .user-message");
  const el  = bub ? bub.querySelector(".message-content") : null;
  toggleSpeech((el || bub) ? (el || bub).innerText : "", btn);
};

// ── Page Init ─────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  if (!localStorage.getItem("seenDisclaimer")) {
    localStorage.setItem("seenDisclaimer", "yes");
    window.location.href = "/disclaimer";
    return;
  }

  // Always start fresh session; history available via sidebar
  currentSessionId = Date.now().toString();
  localStorage.setItem("currentSessionId", currentSessionId);
  uploadedFilePaths = []; activeDocumentId = null;

  // Welcome state: allow body scroll
  document.body.classList.add("welcome-mode");

  renderHistory();
  focusInput();
});


// =====================================================================
// Daily Mind Check widget
// =====================================================================
(function () {
  const moodBtns   = document.querySelectorAll(".mood-emoji-btn");
  const checkBtn   = document.getElementById("moodCheckBtn");
  let selectedMood = null;

  const moodPrompts = {
    1: "I'm feeling very bad today. I could really use some support.",
    2: "I'm not feeling great today. Can you help me feel a bit better?",
    3: "I'm feeling okay today. Any tips to stay balanced?",
    4: "I'm feeling pretty good today! How can I make the most of it?",
    5: "I'm feeling great today! I'd love to share some positivity.",
  };

  const moodLabels = {
    1: "Very Bad 😢", 2: "Bad 😟", 3: "Okay 😐", 4: "Good 🙂", 5: "Great 😄",
  };

  moodBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      moodBtns.forEach((b) => b.classList.remove("selected"));
      btn.classList.add("selected");
      selectedMood = Number(btn.dataset.mood);
    });
  });

  if (checkBtn) {
    checkBtn.addEventListener("click", () => {
      if (!selectedMood) {
        showToast("Please select a mood first 👆");
        return;
      }
      const prompt = moodPrompts[selectedMood];
      // show brief toast then send to chat
      showToast(`Mood: ${moodLabels[selectedMood]} — starting chat...`);
      setTimeout(() => handleSend(prompt), 700);
    });
  }

  function showToast(msg) {
    const existing = document.querySelector(".mood-toast");
    if (existing) existing.remove();
    const toast = document.createElement("div");
    toast.className = "mood-toast";
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transition = "opacity 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, 2200);
  }
})();

// =====================================================================
// Today's Focus — persist checkbox state in localStorage
// =====================================================================
(function () {
  const checkboxes = document.querySelectorAll(".focus-checkbox");
  const today      = new Date().toDateString();
  const storageKey = "focusChecks_" + today;
  const saved      = JSON.parse(localStorage.getItem(storageKey) || "{}");

  checkboxes.forEach((cb) => {
    if (saved[cb.id]) cb.checked = true;
    cb.addEventListener("change", () => {
      saved[cb.id] = cb.checked;
      localStorage.setItem(storageKey, JSON.stringify(saved));
    });
  });
})();


// =====================================================================
// RESPONSE INSPECTOR
// Attaches a "View Insights" button to each agent bot bubble.
// Opens a slide-in modal with real runtime telemetry from the
// backend "insights" SSE frame.
// =====================================================================

// ── Attach "View Insights" button to a bot bubble ──────────────────
function attachInsightsButton(bubbleEl, insights) {
  if (!bubbleEl || !insights) return;
  // Avoid double-adding
  if (bubbleEl.querySelector(".insights-btn")) return;

  const btn = document.createElement("button");
  btn.className = "insights-btn";
  btn.innerHTML = "🔍 View Response Insights";
  btn.setAttribute("aria-label", "View response insights");
  btn.addEventListener("click", () => openInsightsModal(insights));

  // Insert before the message-footer so it sits above timestamps
  const footer = bubbleEl.querySelector(".message-footer");
  if (footer) {
    bubbleEl.insertBefore(btn, footer);
  } else {
    bubbleEl.appendChild(btn);
  }
}

// ── Modal open/close ───────────────────────────────────────────────
function openInsightsModal(insights) {
  // Remove any existing modal
  const existing = document.getElementById("insightsModal");
  if (existing) existing.remove();

  const modal = document.createElement("div");
  modal.id = "insightsModal";
  modal.className = "insights-modal-overlay";
  modal.setAttribute("role", "dialog");
  modal.setAttribute("aria-modal", "true");
  modal.setAttribute("aria-label", "Response Inspector");

  modal.innerHTML = buildInsightsHTML(insights);
  document.body.appendChild(modal);

  // Close on backdrop click
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeInsightsModal();
  });

  // Focus trap — close on Escape
  modal.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeInsightsModal();
  });

  // Animate in
  requestAnimationFrame(() => modal.classList.add("insights-modal-open"));

  // Focus the close button
  const closeBtn = modal.querySelector(".insights-close-btn");
  if (closeBtn) closeBtn.focus();
}

function closeInsightsModal() {
  const modal = document.getElementById("insightsModal");
  if (!modal) return;
  modal.classList.remove("insights-modal-open");
  modal.addEventListener("transitionend", () => modal.remove(), { once: true });
}

// ── Build Inspector HTML from insights dict ────────────────────────
function buildInsightsHTML(d) {
  if (!d) return "<p>No insights available.</p>";

  const fmt  = (v, fallback = "—") => (v !== null && v !== undefined ? v : fallback);
  const pct  = (v) => (v !== null && v !== undefined ? `${Math.round(v * 100)}%` : "—");
  const ms   = (v) => (v ? `${v} ms` : "—");
  const bool = (v) => v
    ? `<span class="ins-badge ins-badge-yes">Yes</span>`
    : `<span class="ins-badge ins-badge-no">No</span>`;

  const em     = d.emotion  || {};
  const agent  = d.agent    || {};
  const rag    = d.rag      || {};
  const llm    = d.llm      || {};
  const cache  = d.cache    || {};
  const fb     = d.fallback || {};
  const val    = d.validation || {};
  const mcp = d.mcp || {calls: []};
  const web = d.web_search || {};

  // ── Agent tools list ──────────────────────────────────────────────
  const toolList = (agent.tool_details || []);
  const toolsHTML = toolList.length
    ? `<ul class="ins-tool-list">
        ${toolList.map(t => `
          <li class="ins-tool-item">
            <span class="ins-tool-check">✓</span>
            <span class="ins-tool-name">${escapeHtml(t.tool)}</span>
            <span class="ins-tool-summary">${escapeHtml(t.result_summary || "")}</span>
            <span class="ins-tool-latency">${ms(t.latency_ms)}</span>
          </li>`).join("")}
       </ul>`
    : `<p class="ins-none">No tools used — direct response</p>`;

  // ── Emotion confidence bar ────────────────────────────────────────
  const confPct = em.confidence != null ? Math.round(em.confidence * 100) : null;
  const confBar = confPct != null
    ? `<div class="ins-bar-wrap">
         <div class="ins-bar" style="width:${confPct}%" aria-valuenow="${confPct}"></div>
       </div>`
    : "";

  return `
    <div class="insights-modal-panel">
      <div class="ins-header">
        <div class="ins-header-left">
          <span class="ins-icon">🔬</span>
          <span class="ins-title">Response Inspector</span>
        </div>
        <button class="insights-close-btn" onclick="closeInsightsModal()" aria-label="Close">✕</button>
      </div>

      <div class="ins-body">

        <!-- Row 1: Emotion + Intent -->
        <div class="ins-row">
          <div class="ins-card">
            <div class="ins-card-label">Emotion</div>
            <div class="ins-card-value ins-capitalize">${fmt(em.label)}</div>
            ${confBar}
            ${confPct != null ? `<div class="ins-card-sub">Confidence ${confPct}%</div>` : ""}
          </div>
          <div class="ins-card">
            <div class="ins-card-label">Intent</div>
            <div class="ins-card-value ins-capitalize">${fmt(d.intent)}</div>
          </div>
          <div class="ins-card">
            <div class="ins-card-label">Total Latency</div>
            <div class="ins-card-value">${ms(d.total_latency_ms)}</div>
          </div>
        </div>

        <!-- Agent Actions -->
        <div class="ins-section">
          <div class="ins-section-title">
            🤖 Agent Actions
            <span class="ins-badge ins-badge-count">${agent.tool_count || 0} tool${agent.tool_count !== 1 ? "s" : ""}</span>
          </div>
          ${toolsHTML}
          <div class="ins-card-sub" style="margin-top:6px">
            Planning latency: ${ms(agent.planning_latency_ms)}
          </div>
        </div>

        <div class="ins-section">
          <div class="ins-section-title">MCP Activity</div>
          <div class="ins-kv"><span>Used</span>${bool(mcp.used)}</div>
          <ul class="ins-tool-list">${(mcp.calls || []).map(c => `
            <li class="ins-tool-item">
              <span>${c.status === "success" ? "✓" : "✕"}</span>
              <span>${escapeHtml(c.tool || "")}</span>
              <span>${escapeHtml(c.server || "")}</span>
              <span>${escapeHtml(c.status || "")}</span>
              <span>${ms(c.latency_ms)}</span>
            </li>`).join("")}</ul>
        </div>
        <div class="ins-section">
          <div class="ins-section-title">Web Search</div>
          <div class="ins-row ins-row-sm">
            <div class="ins-kv"><span>Used</span>${bool(web.used)}</div>
            <div class="ins-kv"><span>Provider</span><strong>${escapeHtml(web.provider || "—")}</strong></div>
            <div class="ins-kv"><span>Results</span><strong>${fmt(web.results)}</strong></div>
            <div class="ins-kv"><span>Latency</span><strong>${ms(web.latency_ms)}</strong></div>
          </div>
        </div>
        <!-- RAG -->
        <div class="ins-section">
          <div class="ins-section-title">📚 RAG Retrieval</div>
          <div class="ins-row ins-row-sm">
            <div class="ins-kv"><span>Used</span>${bool(rag.used)}</div>
            <div class="ins-kv"><span>Chunks retrieved</span><strong>${fmt(rag.chunks_retrieved, "0")}</strong></div>
            <div class="ins-kv"><span>Top similarity</span><strong>${rag.top_similarity != null ? rag.top_similarity.toFixed(3) : "—"}</strong></div>
            <div class="ins-kv"><span>Groundedness</span><strong>${rag.groundedness_score != null ? pct(rag.groundedness_score) : "—"}</strong></div>
          </div>
        </div>

        <!-- LLM -->
        <div class="ins-section">
          <div class="ins-section-title">💬 LLM</div>
          <div class="ins-row ins-row-sm">
            <div class="ins-kv"><span>Provider</span><strong class="ins-capitalize">${fmt(llm.provider)}</strong></div>
            <div class="ins-kv"><span>Model</span><strong>${fmt(llm.model)}</strong></div>
            <div class="ins-kv"><span>Input tokens</span><strong>${fmt(llm.input_tokens)}</strong></div>
            <div class="ins-kv"><span>Output tokens</span><strong>${fmt(llm.output_tokens)}</strong></div>
            <div class="ins-kv"><span>LLM latency</span><strong>${ms(llm.latency_ms)}</strong></div>
          </div>
        </div>

        <!-- Cache + Fallback + Validation -->
        <div class="ins-row">
          <div class="ins-card">
            <div class="ins-card-label">Cache</div>
            <div class="ins-card-value">${cache.hit ? "HIT" : "MISS"}</div>
            ${cache.layer ? `<div class="ins-card-sub">${cache.layer}</div>` : ""}
          </div>
          <div class="ins-card">
            <div class="ins-card-label">Fallback</div>
            <div class="ins-card-value">${bool(fb.used)}</div>
            ${fb.provider ? `<div class="ins-card-sub">${fb.provider}</div>` : ""}
          </div>
          <div class="ins-card">
            <div class="ins-card-label">Validation</div>
            <div class="ins-card-value">${val.passed === true
              ? `<span class="ins-badge ins-badge-yes">Passed</span>`
              : `<span class="ins-badge ins-badge-no">Failed</span>`}</div>
          </div>
        </div>

      </div>
    </div>`;
}
