// ══════════════════════════════════════════
// AlzhAgent Frontend
// ══════════════════════════════════════════

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const chatMessages = $("#chat-messages");
const chatInput = $("#chat-input");
const sendBtn = $("#send-btn");
const apiKeyInput = $("#api-key");
const modelSelect = $("#model-select");
const tempSlider = $("#temperature");
const tempValue = $("#temp-value");
const sidebarToggle = $("#sidebar-toggle");
const sidebar = $("#sidebar");

let chatHistory = [];
let currentMode = "chat";

// ── Init ──
async function init() {
    // Load models
    try {
        const res = await fetch("/api/models");
        const data = await res.json();
        data.models.forEach((m) => {
            const opt = document.createElement("option");
            opt.value = m.id;
            opt.textContent = m.name;
            modelSelect.appendChild(opt);
        });
    } catch (e) { console.error("Model load error:", e); }

    // Load DB stats
    refreshDBStats();

    // Restore API key
    const saved = localStorage.getItem("alzh_api_key");
    if (saved) apiKeyInput.value = saved;

    // Events
    tempSlider.addEventListener("input", () => { tempValue.textContent = tempSlider.value; });
    sidebarToggle.addEventListener("click", () => { sidebar.classList.toggle("visible"); });
    apiKeyInput.addEventListener("change", () => { localStorage.setItem("alzh_api_key", apiKeyInput.value); });
}
init();

// ── Mode Switching ──
function switchMode(mode) {
    currentMode = mode;
    $$(".mode-panel").forEach((p) => p.classList.remove("active"));
    $(`#mode-${mode}`).classList.add("active");
    $$(".mode-btn").forEach((b) => b.classList.remove("active"));
    $(`.mode-btn[data-mode="${mode}"]`).classList.add("active");
}

// ── DB Stats ──
async function refreshDBStats() {
    try {
        const res = await fetch("/api/db/stats");
        const data = await res.json();
        $("#lit-count").textContent = data.literature || 0;
        $("#res-count").textContent = data.research || 0;
    } catch (e) { /* ignore */ }
}

// ── Helpers ──
function getApiKey() {
    const k = apiKeyInput.value.trim();
    if (!k) { showError("사이드바에서 OpenRouter API Key를 입력해주세요."); return null; }
    return k;
}

function showError(msg) {
    const t = document.createElement("div");
    t.className = "error-toast";
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 4000);
}

function addMessage(role, content, extras) {
    const welcome = chatMessages.querySelector(".welcome-message");
    if (welcome) welcome.remove();

    const div = document.createElement("div");
    div.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "👤" : "🧬";

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";

    if (role === "assistant" && content) {
        contentDiv.innerHTML = marked.parse(content);
    } else {
        contentDiv.textContent = content || "";
    }

    div.appendChild(avatar);
    div.appendChild(contentDiv);

    // Source/entity tags
    if (extras) {
        if (extras.sources && extras.sources.length > 0) {
            const srcDiv = document.createElement("div");
            srcDiv.className = "msg-sources";
            extras.sources.slice(0, 5).forEach((s) => {
                const tag = document.createElement("span");
                tag.className = "source-tag";
                tag.textContent = s;
                srcDiv.appendChild(tag);
            });
            contentDiv.appendChild(srcDiv);
        }
        if (extras.entities && extras.entities.length > 0) {
            const entDiv = document.createElement("div");
            entDiv.className = "msg-sources";
            extras.entities.slice(0, 6).forEach((e) => {
                const tag = document.createElement("span");
                tag.className = "entity-tag";
                tag.textContent = `${e.label}: ${e.normalized}`;
                entDiv.appendChild(tag);
            });
            contentDiv.appendChild(entDiv);
        }
    }

    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return contentDiv;
}

// ══════════════════════════════════════════
// 1) CHAT MODE — 듀얼 RAG 챗봇
// ══════════════════════════════════════════
async function sendChat(e) {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (!text) return;
    const apiKey = getApiKey();
    if (!apiKey) return;

    chatHistory.push({ role: "user", content: text });
    addMessage("user", text);
    chatInput.value = "";
    sendBtn.disabled = true;

    const contentDiv = addMessage("assistant", "");

    try {
        // RAG 챗봇 엔드포인트 사용
        const res = await fetch("/api/rag-chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                api_key: apiKey,
                question: text,
                history: chatHistory.slice(-10),
            }),
        });

        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();

        contentDiv.innerHTML = marked.parse(data.answer || "응답 없음");
        chatHistory.push({ role: "assistant", content: data.answer });

        // 출처 + 엔티티 태그
        const tagsDiv = document.createElement("div");

        if (data.sources && data.sources.length > 0) {
            const srcDiv = document.createElement("div");
            srcDiv.className = "msg-sources";
            data.sources.slice(0, 5).forEach((s) => {
                const tag = document.createElement("span");
                tag.className = "source-tag";
                tag.textContent = s;
                srcDiv.appendChild(tag);
            });
            contentDiv.appendChild(srcDiv);
        }

        if (data.entities && data.entities.length > 0) {
            const entDiv = document.createElement("div");
            entDiv.className = "msg-sources";
            data.entities.slice(0, 6).forEach((e) => {
                const tag = document.createElement("span");
                tag.className = "entity-tag";
                tag.textContent = `${e.label}: ${e.normalized}`;
                entDiv.appendChild(tag);
            });
            contentDiv.appendChild(entDiv);
        }

    } catch (err) {
        // 폴백: 일반 스트리밍 챗
        try {
            contentDiv.textContent = "";
            const res2 = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    api_key: apiKey,
                    model: modelSelect.value,
                    temperature: parseFloat(tempSlider.value),
                    messages: chatHistory,
                    use_rag: true,
                }),
            });
            const reader = res2.body.getReader();
            const decoder = new TextDecoder();
            let full = "";
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                full += decoder.decode(value, { stream: true });
                contentDiv.innerHTML = marked.parse(full + "▌");
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            contentDiv.innerHTML = marked.parse(full);
            chatHistory.push({ role: "assistant", content: full });
        } catch (err2) {
            contentDiv.textContent = "오류: " + err2.message;
            showError("API 오류: " + err2.message);
        }
    } finally {
        sendBtn.disabled = false;
        chatInput.focus();
        refreshDBStats();
    }
}

function sendSuggestion(text) {
    chatInput.value = text;
    sendChat(new Event("submit"));
}

function clearChat() {
    chatHistory = [];
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">🧬</div>
            <h3>AlzhChat에 오신 것을 환영합니다</h3>
            <p>에이전트 리서치 결과와 문헌 DB를 참조하여 답변합니다.</p>
            <div class="suggestion-chips">
                <button onclick="sendSuggestion('현재 DB에 어떤 알츠하이머 타깃 정보가 있나요?')">📋 DB 현황 보기</button>
                <button onclick="sendSuggestion('Tau 타깃 기반 알츠하이머 치료제 개발 동향은?')">🎯 Tau 타깃 동향</button>
                <button onclick="sendSuggestion('TREM2 작용제 최신 연구 요약해줘')">🧪 TREM2 연구</button>
            </div>
        </div>`;
}

// ══════════════════════════════════════════
// 2) RESEARCH MODE — 에이전트 파이프라인
// ══════════════════════════════════════════
let researchPollId = null;

async function startResearch() {
    const apiKey = getApiKey();
    if (!apiKey) return;

    const query = $("#research-query").value.trim();
    const n = parseInt($("#research-n").value) || 5;
    if (!query) { showError("리서치 주제를 입력하세요."); return; }

    $("#research-start-btn").disabled = true;
    $("#research-progress").classList.remove("hidden");
    $("#research-results").classList.add("hidden");
    $("#progress-fill").style.width = "5%";
    $("#progress-msg").textContent = "시작 중...";

    try {
        const res = await fetch("/api/research/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ api_key: apiKey, query, max_compounds: n }),
        });
        const data = await res.json();
        if (data.error) { showError(data.error); return; }

        // 폴링
        researchPollId = setInterval(pollResearch, 2000);
    } catch (err) {
        showError("리서치 시작 실패: " + err.message);
        $("#research-start-btn").disabled = false;
    }
}

async function pollResearch() {
    try {
        const res = await fetch("/api/research/status");
        const s = await res.json();

        const pct = s.running ? Math.max(10, (s.step / s.total) * 90) : 100;
        $("#progress-fill").style.width = pct + "%";
        $("#progress-msg").textContent = s.message || "진행 중...";

        if (!s.running) {
            clearInterval(researchPollId);
            $("#research-start-btn").disabled = false;

            if (s.result) {
                renderResearchResults(s.result);
            }
            refreshDBStats();
        }
    } catch (e) { /* retry */ }
}

function renderResearchResults(r) {
    const div = $("#research-results");
    div.classList.remove("hidden");

    const targets = r.literature?.targets || [];
    const design = r.design || {};
    const critic = r.critic || {};
    const conf = critic.confidence || {};
    const topCpds = design.top_compounds || [];

    let html = `
    <div class="result-card">
        <h3>📊 요약</h3>
        <div class="metrics">
            <div class="metric-box"><div class="val">${targets.length}</div><div class="lbl">타깃</div></div>
            <div class="metric-box"><div class="val">${design.total || 0}</div><div class="lbl">생성 화합물</div></div>
            <div class="metric-box"><div class="val">${design.total_drug_like || 0}</div><div class="lbl">약물성 통과</div></div>
            <div class="metric-box"><div class="val">${r.elapsed || 0}s</div><div class="lbl">소요 시간</div></div>
            <div class="metric-box"><div class="val">${conf.overall || '?'}%</div><div class="lbl">신뢰도</div></div>
        </div>
    </div>`;

    // Targets
    if (targets.length > 0) {
        html += `<div class="result-card"><h3>🎯 식별된 타깃</h3>`;
        targets.forEach((t) => {
            html += `<div class="compound-row">
                <strong>${t.name || '?'}</strong> — Evidence: ${t.evidence || '?'}
                <div class="cpd-metrics">Mechanism: ${t.mechanism || 'N/A'} | Strategy: ${t.strategy || 'N/A'}</div>
            </div>`;
        });
        html += `</div>`;
    }

    // Compounds
    if (topCpds.length > 0) {
        html += `<div class="result-card"><h3>💊 상위 후보 화합물</h3>`;
        topCpds.forEach((c, i) => {
            const lip = c.lipinski ? '✅' : '❌';
            const tox = (c.tox_alerts?.length || 0) > 0 ? `⚠️ ${c.tox_alerts.join(', ')}` : '✅ Clean';
            html += `<div class="compound-row">
                <strong>#${i + 1}</strong> QED: <strong>${(c.qed || 0).toFixed(3)}</strong>
                <code>${c.smiles || '?'}</code>
                <div class="cpd-metrics">
                    LogP: ${(c.logp || 0).toFixed(2)} | MW: ${(c.mw || 0).toFixed(0)} | Lipinski: ${lip} | Tox: ${tox}
                </div>
            </div>`;
        });
        html += `</div>`;
    }

    // Critic Report
    if (critic.report) {
        html += `<div class="result-card">
            <h3 class="collapsible" onclick="toggleCollapse(this)">📋 검증 리포트</h3>
            <div class="collapse-body">${marked.parse(critic.report)}</div>
        </div>`;
    }

    div.innerHTML = html;
}

function toggleCollapse(el) {
    el.classList.toggle("open");
    const body = el.nextElementSibling;
    body.classList.toggle("open");
}

// ══════════════════════════════════════════
// 3) EXPLORE MODE — DB 검색
// ══════════════════════════════════════════
async function searchDB() {
    const q = $("#explore-query").value.trim();
    if (!q) return;
    const target = $("#explore-target").value;
    const div = $("#explore-results");
    div.innerHTML = '<p style="color:var(--text-muted)">검색 중...</p>';

    try {
        const res = await fetch("/api/db/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: q, target, top_k: 10 }),
        });
        const data = await res.json();

        // Flatten results
        let items = [];
        if (data.results) {
            items = data.results;
        } else {
            if (data.research) items = items.concat(data.research.map((r) => ({ ...r, _src: "🔬 Research" })));
            if (data.literature) items = items.concat(data.literature.map((r) => ({ ...r, _src: "📚 Literature" })));
        }

        if (items.length === 0) {
            div.innerHTML = '<p style="color:var(--text-muted)">결과가 없습니다.</p>';
            return;
        }

        let html = "";
        items.forEach((item) => {
            const m = item.metadata || {};
            const d = item.distance;
            const rel = d != null ? (1 - d).toFixed(2) : "?";
            const src = item._src || (m.source === "pubmed" ? "📚 Literature" : "🔬 Research");
            let header = `${src} | Relevance: ${rel}`;
            if (m.type) header += ` | ${m.type}`;
            if (m.target) header += ` | ${m.target}`;
            if (m.pmid) header += ` | PMID:${m.pmid}`;

            html += `<div class="result-item">
                <div class="result-header">${header}</div>
                <div class="result-text">${item.text || ''}</div>
            </div>`;
        });
        div.innerHTML = html;
    } catch (e) {
        div.innerHTML = `<p style="color:var(--err)">검색 오류: ${e.message}</p>`;
    }
}