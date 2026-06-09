const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const previewWrap = document.getElementById("previewWrap");
const dropZone = document.getElementById("dropZone");
const clearImageBtn = document.getElementById("clearImageBtn");
const identifyBtn = document.getElementById("identifyBtn");
const healthBtn = document.getElementById("healthBtn");
const resultText = document.getElementById("resultText");
const runState = document.getElementById("runState");
const journalFound = document.getElementById("journalFound");
const recentList = document.getElementById("recentList");

let currentAudioParts = [];

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (!file) return;
  const objectUrl = URL.createObjectURL(file);
  previewImage.onload = () => {
    sizePreviewToImage();
    URL.revokeObjectURL(objectUrl);
  };
  previewImage.src = objectUrl;
  dropZone.classList.add("has-image");
});

dropZone.addEventListener("click", (event) => {
  if (dropZone.classList.contains("has-image")) {
    return;
  }
  event.preventDefault();
  imageInput.click();
});

dropZone.addEventListener("keydown", (event) => {
  if (dropZone.classList.contains("has-image")) {
    return;
  }
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    imageInput.click();
  }
});

clearImageBtn.addEventListener("click", (event) => {
  event.preventDefault();
  event.stopPropagation();
  imageInput.value = "";
  previewImage.removeAttribute("src");
  previewWrap.style.removeProperty("--preview-width");
  previewWrap.style.removeProperty("--preview-height");
  dropZone.classList.remove("has-image");
  runState.textContent = "Ready";
  resultText.textContent = "Upload a plant image to begin.";
});

window.addEventListener("resize", () => {
  if (dropZone.classList.contains("has-image") && previewImage.naturalWidth && previewImage.naturalHeight) {
    sizePreviewToImage();
  }
});

function sizePreviewToImage() {
  const ratio = previewImage.naturalWidth / previewImage.naturalHeight;
  const maxWidth = Math.min(dropZone.clientWidth * 0.92, 660);
  const maxHeight = dropZone.clientHeight * 0.68;
  let width = maxWidth;
  let height = width / ratio;

  if (height > maxHeight) {
    height = maxHeight;
    width = height * ratio;
  }

  previewWrap.style.setProperty("--preview-width", `${Math.round(width)}px`);
  previewWrap.style.setProperty("--preview-height", `${Math.round(height)}px`);
}

identifyBtn.addEventListener("click", () => runQuest("identify"));
healthBtn.addEventListener("click", () => runQuest("health"));

async function loadJournal() {
  try {
    const response = await fetch("/api/journal");
    const data = await response.json();
    const stats = data.stats || {};
    journalFound.textContent = stats.species || 0;

    const recent = data.recent || [];
    recentList.innerHTML = "";
    if (!recent.length) {
      recentList.innerHTML = `<div class="recent-card"><strong>Undiscovered</strong><span>The journal is waiting.</span></div>`;
      return;
    }

    for (const item of recent.slice(0, 6)) {
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

  const form = new FormData();
  form.append("mode", mode);
  form.append("image", image);

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

loadJournal();
