const imageInput = document.getElementById("imageInput");
const voiceInput = document.getElementById("voiceInput");
const previewImage = document.getElementById("previewImage");
const dropZone = document.getElementById("dropZone");
const identifyBtn = document.getElementById("identifyBtn");
const healthBtn = document.getElementById("healthBtn");
const resultText = document.getElementById("resultText");
const runState = document.getElementById("runState");
const workerState = document.getElementById("workerState");
const speciesCount = document.getElementById("speciesCount");
const rescueCount = document.getElementById("rescueCount");
const pointCount = document.getElementById("pointCount");
const journalMeter = document.getElementById("journalMeter");
const journalFound = document.getElementById("journalFound");
const recentList = document.getElementById("recentList");
const voicePlayer = document.getElementById("voicePlayer");

let currentAudioParts = [];

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (!file) return;
  previewImage.src = URL.createObjectURL(file);
  dropZone.classList.add("has-image");
});

identifyBtn.addEventListener("click", () => runQuest("identify"));
healthBtn.addEventListener("click", () => runQuest("health"));

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    const readyCount = Object.values(status.workers || {}).filter((worker) => worker.native_ready).length;
    workerState.textContent = readyCount ? `${readyCount} native worker ready` : "Demo scout mode";
  } catch {
    workerState.textContent = "Camp signal weak";
  }
}

async function loadJournal() {
  try {
    const response = await fetch("/api/journal");
    const data = await response.json();
    const stats = data.stats || {};
    speciesCount.textContent = stats.species || 0;
    rescueCount.textContent = stats.rescues || 0;
    pointCount.textContent = (stats.total || 0) * 20;
    journalFound.textContent = stats.species || 0;
    journalMeter.style.width = `${Math.min(100, (stats.total || 0) * 8)}%`;
    recentList.innerHTML = "";
    const recent = data.recent || [];
    if (!recent.length) {
      recentList.innerHTML = `<div class="recent-card"><strong>Undiscovered</strong><span>The journal is waiting.</span></div>`;
      return;
    }
    for (const item of recent) {
      const title = item.species_common || item.health_status || item.mode || "Discovery";
      const card = document.createElement("article");
      card.className = "recent-card";
      card.innerHTML = `<strong>${escapeHtml(title)}</strong><span>${escapeHtml(item.status)} · ${escapeHtml(item.mode)}</span>`;
      recentList.appendChild(card);
    }
  } catch {
    recentList.innerHTML = `<div class="recent-card"><strong>Journal unavailable</strong><span>Try again shortly.</span></div>`;
  }
}

async function runQuest(mode) {
  const image = imageInput.files?.[0];
  if (!image) {
    runState.textContent = "Image needed";
    resultText.textContent = "Choose a plant photo before starting the expedition.";
    return;
  }

  identifyBtn.disabled = true;
  healthBtn.disabled = true;
  runState.textContent = mode === "identify" ? "Discovering" : "Rescuing";
  resultText.textContent = "";
  currentAudioParts = [];
  voicePlayer.removeAttribute("src");

  const form = new FormData();
  form.append("mode", mode);
  form.append("image", image);
  const voice = voiceInput.files?.[0];
  if (voice) form.append("voice_sample", voice);

  try {
    const response = await fetch("/api/run", { method: "POST", body: form });
    if (!response.ok || !response.body) {
      throw new Error(`Request failed with ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        handleEvent(JSON.parse(line));
      }
    }
    if (buffer.trim()) handleEvent(JSON.parse(buffer));
  } catch (error) {
    runState.textContent = "Trail blocked";
    resultText.textContent += `\n\n${error.message}`;
  } finally {
    identifyBtn.disabled = false;
    healthBtn.disabled = false;
    await loadStatus();
    await loadJournal();
  }
}

function handleEvent(event) {
  if (event.type === "status") {
    runState.textContent = event.message;
  }
  if (event.type === "text_delta") {
    resultText.textContent += event.delta;
  }
  if (event.type === "audio_chunk") {
    const bytes = base64ToBytes(event.data);
    currentAudioParts.push(bytes);
    const blob = new Blob(currentAudioParts, { type: event.mime_type || "audio/wav" });
    voicePlayer.src = URL.createObjectURL(blob);
  }
  if (event.type === "record_saved") {
    runState.textContent = "Journal saved";
  }
  if (event.type === "done") {
    runState.textContent = "Complete";
  }
  if (event.type === "error") {
    runState.textContent = "Error";
    resultText.textContent += `\n\n${event.message}`;
  }
}

function base64ToBytes(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

loadStatus();
loadJournal();
