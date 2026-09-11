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
  overwhelmed: { emoji: "😰", label: "Overwhelmed", class: "emotion-overwhelmed" },
  sad:         { emoji: "😢", label: "Sad",         class: "emotion-sad"         },
  angry:       { emoji: "😠", label: "Angry",       class: "emotion-angry"       },
  anxious:     { emoji: "😨", label: "Anxious",     class: "emotion-anxious"     },
  neutral:     { emoji: "😌", label: "Neutral",     class: "emotion-neutral"     },
  happy:       { emoji: "😊", label: "Happy",       class: "emotion-happy"       },
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
function showEmotionBadge(emotion) {
  const cfg = EMOTION_CONFIG[emotion];
  if (!cfg || emotion === "neutral") return;
  const existing = document.querySelector(".emotion-float");
  if (existing) existing.remove();
  const badge = document.createElement("div");
  badge.className = `emotion-badge ${cfg.class} emotion-float`;
  badge.innerHTML = `<span style="font-size:20px">${cfg.emoji}</span><span>Emotion detected: ${cfg.label}</span>`;
  document.body.appendChild(badge);
  setTimeout(() => {
    badge.style.opacity = "0";
    badge.style.transition = "opacity 0.3s ease-out";
    setTimeout(() => badge.remove(), 300);
  }, 6000);
}

function createEmotionBadge(emotion) {
  const cfg = EMOTION_CONFIG[emotion];
  if (!cfg || emotion === "neutral") return null;
  const badge = document.createElement("div");
  badge.className = `emotion-badge ${cfg.class}`;
  badge.innerHTML = `<span>${cfg.emoji}</span><span>${cfg.label}</span>`;
  return badge;
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
  const lines = String(text ?? "").replace(/\\n/g, "\n").replace(/\r\n?/g, "\n").trim().split("\n");
  const html = [];
  let paragraph = [], listType = null, inCodeBlock = false, codeLines = [];

  const flushP   = () => { if (!paragraph.length) return; html.push(`<p>${paragraph.map(formatInlineMarkdown).join("<br>")}</p>`); paragraph = []; };
  const closeL   = () => { if (!listType) return; html.push(`</${listType}>`); listType = null; };
  const openL    = (t) => { flushP(); if (listType === t) return; closeL(); listType = t; html.push(`<${t}>`); };

  for (const rawLine of lines) {
    const line = rawLine.trimEnd(), trimmed = line.trim();
    if (/^```/.test(trimmed)) {
      flushP(); closeL();
      if (inCodeBlock) { html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`); codeLines = []; }
      inCodeBlock = !inCodeBlock; continue;
    }
    if (inCodeBlock) { codeLines.push(rawLine); continue; }
    if (!trimmed) { flushP(); closeL(); continue; }
    const hd = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (hd) { flushP(); closeL(); const lv = Math.min(hd[1].length + 1, 4); html.push(`<h${lv}>${formatInlineMarkdown(hd[2])}</h${lv}>`); continue; }
    if (/^[^.!?]{2,64}:$/.test(trimmed) && !/^https?:/i.test(trimmed)) { flushP(); closeL(); html.push(`<h3>${formatInlineMarkdown(trimmed.slice(0,-1))}</h3>`); continue; }
    const ul = trimmed.match(/^[-*•]\s+(.+)$/);
    if (ul) { openL("ul"); html.push(`<li>${formatInlineMarkdown(ul[1])}</li>`); continue; }
    const ol = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (ol) { openL("ol"); html.push(`<li>${formatInlineMarkdown(ol[1])}</li>`); continue; }
    const bq = trimmed.match(/^>\s?(.+)$/);
    if (bq) { flushP(); closeL(); html.push(`<blockquote>${formatInlineMarkdown(bq[1])}</blockquote>`); continue; }
    closeL(); paragraph.push(trimmed);
  }
  flushP(); closeL();
  if (inCodeBlock && codeLines.length) html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
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
  if (!isUser && emotion) {
    const eb = createEmotionBadge(emotion);
    if (eb) { wrapper.appendChild(eb); showEmotionBadge(emotion); }
  }
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

  async function finalize(sources, emotion, hasFiles) {
    await waitForDrain();
    _displayed = _plain; renderBuf(); cursor.remove(); bubble.classList.remove("streaming-active");
    if (emotion) { const eb = createEmotionBadge(emotion); if (eb) { wrapper.insertBefore(eb, bubble); showEmotionBadge(emotion); } }
    if (hasFiles) { const fi = document.createElement("div"); fi.className = "file-indicator"; fi.innerHTML = "🔍 Response based on uploaded files"; bubble.appendChild(fi); }
    const footer = document.createElement("div"); footer.className = "message-footer";
    const ts = document.createElement("span"); ts.className = "timestamp"; ts.textContent = addTimestamp();
    const cpBtn = document.createElement("button"); cpBtn.className = "icon-btn"; cpBtn.innerHTML = "📋"; cpBtn.title = "Copy";
    cpBtn.onclick = () => navigator.clipboard.writeText(_plain).then(() => { cpBtn.innerHTML = "✅"; setTimeout(() => (cpBtn.innerHTML = "📋"), 1000); });
    const spBtn = document.createElement("button"); spBtn.className = "icon-btn"; spBtn.innerHTML = "🔊"; spBtn.title = "Read aloud";
    spBtn.onclick = () => toggleSpeech(_plain, spBtn);
    footer.append(ts, cpBtn, spBtn); bubble.appendChild(footer); chatBox.scrollTop = chatBox.scrollHeight;
    return _plain;
  }
  return { wrapper, contentEl, cursor, appendChunk, finalize };
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
  let firstToken = true, streamBubble = null, metaEmotion = null, metaRagUsed = false, metaToolUsed = "llm", finished = false;

  const params = new URLSearchParams({ q: query, session_name: currentSessionId });
  if (activeDocumentId) params.set("document_id", activeDocumentId);
  const es = new EventSource(`/search/stream?${params}`);

  function cleanup() { if (finished) return; finished = true; es.close(); isSearching = false; setInputDisabled(false); }

  es.onmessage = async (event) => {
    let frame; try { frame = JSON.parse(event.data); } catch { return; }
    if (frame.type === "meta")  { metaEmotion = frame.emotion_detected || null; metaRagUsed = frame.rag_used || false; metaToolUsed = frame.tool_used || "llm"; return; }
    // ping keeps the SSE connection alive during long multi-tool agent runs
    if (frame.type === "ping") { return; }
    // tool_progress shows which tool the agent is currently running
    if (frame.type === "tool_progress") {
      if (firstToken) { removeTypingBubble(); streamBubble = createStreamingBubble(metaToolUsed, metaRagUsed); firstToken = false; }
      const label = { analyze_emotion: "Analysing emotion…", search_knowledge: "Searching documents…", web_search: "Searching the web…", create_self_care_plan: "Building your plan…", get_tasks: "Fetching tasks…", create_task: "Creating task…", log_mood: "Logging mood…", get_mood_history: "Loading mood history…", start_breathing_exercise: "Preparing exercise…", summarize_document: "Summarising document…" };
      streamBubble.appendChunk(`\n_${label[frame.tool] || frame.tool}_\n`); return;
    }
    if (frame.type === "chunk" || frame.type === "token") {
      if (firstToken) { removeTypingBubble(); streamBubble = createStreamingBubble(metaToolUsed, metaRagUsed); firstToken = false; }
      streamBubble.appendChunk(frame.text || ""); return;
    }
    if (frame.type === "done") {
      es.close(); removeTypingBubble();
      if (streamBubble) { const ft = await streamBubble.finalize(frame.sources || [], metaEmotion, attachFiles && metaRagUsed); saveMessage(currentSessionId, "bot", ft, metaEmotion); }
      else addBotMessage("⚠️ No response received. Please try again.");
      cleanup(); return;
    }
    if (frame.type === "error") {
      es.close(); removeTypingBubble();
      if (streamBubble) { const pt = await streamBubble.finalize([], metaEmotion, false); if (pt) saveMessage(currentSessionId, "bot", pt, metaEmotion); addBotMessage("⚠️ Response was interrupted. Please try again."); }
      else addBotMessage(`❌ ${frame.message || "An error occurred"}`);
      cleanup();
    }
  };
  es.onerror = async () => {
    if (finished) return; removeTypingBubble();
    if (!streamBubble) addBotMessage("❌ Connection lost. Please try again.");
    else { const pt = await streamBubble.finalize([], metaEmotion, false); if (pt) saveMessage(currentSessionId, "bot", pt, metaEmotion); addBotMessage("⚠️ Connection lost. Please try again."); }
    cleanup();
  };
}

// ── Shared send logic ─────────────────────────────────────────────────
function handleSend(query) {
  if (!query || isSearching) return;
  // Transition to chat mode on first real message
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
