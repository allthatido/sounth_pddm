const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const previewWrap = document.getElementById("previewWrap");
const dropZone = document.getElementById("dropZone");
const clearImageBtn = document.getElementById("clearImageBtn");
const identifyBtn = document.getElementById("identifyBtn");
const healthBtn = document.getElementById("healthBtn");
const resultText = document.getElementById("resultText");
const runState = document.getElementById("runState");
const journalText = document.getElementById("journalText");
const imageLoader = document.getElementById("imageLoader");
const speciesFoundCount = document.getElementById("speciesFoundCount");
const plantsRescuedCount = document.getElementById("plantsRescuedCount");
const nameForm = document.getElementById("nameForm");
const nameInput = document.getElementById("nameInput");
const nameSaveBtn = document.getElementById("nameSaveBtn");
const namePlate = document.getElementById("namePlate");
const leaderboardBtn = document.getElementById("leaderboardBtn");
const leaderboardModal = document.getElementById("leaderboardModal");
const leaderboardCloseBtn = document.getElementById("leaderboardCloseBtn");
const recordsTab = document.getElementById("recordsTab");
const rankingsTab = document.getElementById("rankingsTab");
const recordsPanel = document.getElementById("recordsPanel");
const rankingsPanel = document.getElementById("rankingsPanel");
const recordDetailPanel = document.getElementById("recordDetailPanel");
const recordBackBtn = document.getElementById("recordBackBtn");
const recordDetailImage = document.getElementById("recordDetailImage");
const recordDetailMode = document.getElementById("recordDetailMode");
const recordDetailTitle = document.getElementById("recordDetailTitle");
const recordDetailDate = document.getElementById("recordDetailDate");
const recordDetailText = document.getElementById("recordDetailText");
const myRank = document.getElementById("myRank");
const myTotal = document.getElementById("myTotal");
const myDiscover = document.getElementById("myDiscover");
const myRescue = document.getElementById("myRescue");
const myLastActivity = document.getElementById("myLastActivity");
const recentRuns = document.getElementById("recentRuns");
const leaderboardList = document.getElementById("leaderboardList");

const LEADERBOARD_STORAGE_KEY = "aranya_hf_leaderboard_id";

let isRunning = false;
let leaderboardIdentity = loadLeaderboardIdentity();
let hasTextDelta = false;
let audioQueue = [];
let audioPlaying = false;
let audioContext = null;
let segmentText = new Map();
let revealedSegments = new Set();
let pendingTextAnimations = new Set();
let audioIdleResolvers = [];
let audioStarted = false;
let audioStreamDone = false;
let prebufferTimer = null;
let recentRecordCache = new Map();

let balancedTextDelayMs = 600;
let audioPrebufferChunks = 2;
let audioPrebufferMaxMs = 2200;
let audioPlaybackRate = 0.92;

function loadLeaderboardIdentity() {
  const fallback = createLeaderboardIdentity();
  try {
    const stored = JSON.parse(window.localStorage.getItem(LEADERBOARD_STORAGE_KEY) || "null");
    if (stored && typeof stored === "object" && typeof stored.id === "string" && stored.id.trim()) {
      return {
        version: 1,
        id: stored.id.trim(),
        name: typeof stored.name === "string" ? stored.name.trim().slice(0, 80) : "",
        created_at: typeof stored.created_at === "string" ? stored.created_at : fallback.created_at,
      };
    }
  } catch {
    // Replace malformed identity data below.
  }
  saveLeaderboardIdentity(fallback);
  return fallback;
}

function createLeaderboardIdentity() {
  const bytes = new Uint8Array(16);
  if (window.crypto?.getRandomValues) {
    window.crypto.getRandomValues(bytes);
  } else {
    for (let index = 0; index < bytes.length; index += 1) {
      bytes[index] = Math.floor(Math.random() * 256);
    }
  }
  const id = Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
  return {
    version: 1,
    id,
    name: "",
    created_at: new Date().toISOString(),
  };
}

function saveLeaderboardIdentity(identity) {
  window.localStorage.setItem(LEADERBOARD_STORAGE_KEY, JSON.stringify(identity));
}

function hasSavedName() {
  return Boolean(leaderboardIdentity.name && leaderboardIdentity.name.trim());
}

function syncNameUi() {
  if (hasSavedName()) {
    nameForm.hidden = true;
    namePlate.hidden = false;
    namePlate.textContent = leaderboardIdentity.name;
  } else {
    nameForm.hidden = false;
    namePlate.hidden = true;
    nameInput.value = "";
  }
  updateActionAvailability();
}

function updateActionAvailability() {
  const disabled = isRunning || !hasSavedName();
  identifyBtn.disabled = disabled;
  healthBtn.disabled = disabled;
  nameInput.disabled = isRunning;
  nameSaveBtn.disabled = isRunning;
}

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
  if (isRunning || dropZone.classList.contains("has-image")) {
    return;
  }
  event.preventDefault();
  imageInput.click();
});

dropZone.addEventListener("keydown", (event) => {
  if (isRunning || dropZone.classList.contains("has-image")) {
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
  if (isRunning) return;
  imageInput.value = "";
  previewImage.removeAttribute("src");
  previewWrap.style.removeProperty("--preview-width");
  previewWrap.style.removeProperty("--preview-height");
  dropZone.classList.remove("has-image");
  runState.textContent = "Ready";
  setJournalText("Upload a plant image to begin.");
});

window.addEventListener("resize", () => {
  if (dropZone.classList.contains("has-image") && previewImage.naturalWidth && previewImage.naturalHeight) {
    sizePreviewToImage();
  }
});

nameForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const name = nameInput.value.trim().replace(/\s+/g, " ").slice(0, 80);
  if (!name) {
    nameInput.focus();
    runState.textContent = "Name needed";
    return;
  }
  leaderboardIdentity = { ...leaderboardIdentity, name };
  saveLeaderboardIdentity(leaderboardIdentity);
  syncNameUi();
  runState.textContent = "Ready";
  setJournalText("Upload a plant image to begin.");
});

leaderboardBtn.addEventListener("click", () => {
  openLeaderboard();
});

leaderboardCloseBtn.addEventListener("click", () => {
  closeLeaderboard();
});

leaderboardModal.addEventListener("click", (event) => {
  if (event.target?.hasAttribute("data-close-leaderboard")) {
    closeLeaderboard();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !leaderboardModal.hidden) {
    closeLeaderboard();
  }
});

recordsTab.addEventListener("click", () => {
  setLeaderboardTab("records");
});

rankingsTab.addEventListener("click", () => {
  setLeaderboardTab("rankings");
});

recordBackBtn.addEventListener("click", () => {
  setLeaderboardTab("records");
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

async function loadJournalStats() {
  try {
    const response = await fetch("/api/journal");
    const data = await response.json();
    const stats = data.stats || {};
    const species = stats.species || 0;
    const rescues = stats.rescues || 0;
    speciesFoundCount.textContent = species;
    plantsRescuedCount.textContent = rescues;
  } catch {
    speciesFoundCount.textContent = "0";
    plantsRescuedCount.textContent = "0";
  }
}

async function openLeaderboard() {
  leaderboardModal.hidden = false;
  setLeaderboardTab("records");
  await loadLeaderboard();
}

function closeLeaderboard() {
  leaderboardModal.hidden = true;
}

function setLeaderboardTab(tab) {
  const showRecords = tab === "records";
  const showRankings = tab === "rankings";
  const showDetail = tab === "detail";
  recordsTab.classList.toggle("is-active", showRecords || showDetail);
  rankingsTab.classList.toggle("is-active", showRankings);
  recordsTab.setAttribute("aria-selected", String(showRecords));
  rankingsTab.setAttribute("aria-selected", String(showRankings));
  recordsPanel.classList.toggle("is-active", showRecords);
  rankingsPanel.classList.toggle("is-active", showRankings);
  recordDetailPanel.classList.toggle("is-active", showDetail);
  recordsPanel.hidden = !showRecords;
  rankingsPanel.hidden = !showRankings;
  recordDetailPanel.hidden = !showDetail;
}

async function loadLeaderboard() {
  renderLeaderboardLoading();
  try {
    const params = new URLSearchParams({ leaderboard_id: leaderboardIdentity.id });
    const response = await fetch(`/api/leaderboard?${params.toString()}`);
    if (!response.ok) {
      throw new Error(`Leaderboard failed with ${response.status}`);
    }
    renderLeaderboard(await response.json());
  } catch {
    myLastActivity.textContent = "Records are unavailable right now.";
    recentRuns.replaceChildren();
    leaderboardList.replaceChildren(emptyListItem("Leaderboard unavailable."));
  }
}

function renderLeaderboardLoading() {
  recentRecordCache = new Map();
  myRank.textContent = "-";
  myTotal.textContent = "0";
  myDiscover.textContent = "0";
  myRescue.textContent = "0";
  myLastActivity.textContent = "Loading records...";
  recentRuns.replaceChildren();
  leaderboardList.replaceChildren(emptyListItem("Loading leaderboard..."));
}

function renderLeaderboard(data) {
  const me = data.me || {};
  myRank.textContent = me.rank ? `#${me.rank}` : "-";
  myTotal.textContent = me.total || 0;
  myDiscover.textContent = me.discover || 0;
  myRescue.textContent = me.rescue || 0;
  myLastActivity.textContent = me.last_activity ? `Last activity ${formatDate(me.last_activity)}` : "No completed requests yet.";

  const recentItems = (data.recent || []).map((run) => {
    recentRecordCache.set(run.id, run);
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.className = "record-row-button";
    button.type = "button";
    button.addEventListener("click", () => {
      void openRecordDetail(run.id);
    });
    const thumb = document.createElement("img");
    thumb.src = run.thumb_url;
    thumb.alt = "";
    const text = document.createElement("span");
    text.className = "record-row-copy";
    const mode = document.createElement("b");
    mode.textContent = run.mode === "health" ? "Rescue" : "Discover";
    const date = document.createElement("span");
    date.textContent = formatDate(run.created_at);
    text.append(mode, date);
    button.append(thumb, text);
    item.append(button);
    return item;
  });
  recentRuns.replaceChildren(...(recentItems.length ? recentItems : [emptyListItem("No completed requests yet.")]));

  const leaderboardItems = (data.leaderboard || []).map((entry) => {
    const item = document.createElement("li");
    if (entry.leaderboard_id === leaderboardIdentity.id) {
      item.classList.add("is-me");
    }
    const rank = document.createElement("b");
    rank.textContent = `#${entry.rank}`;
    const name = document.createElement("span");
    name.textContent = entry.display_name || "Wildkeeper";
    const counts = document.createElement("small");
    counts.textContent = `${entry.total || 0} total | ${entry.discover || 0} discover | ${entry.rescue || 0} rescue`;
    item.append(rank, name, counts);
    return item;
  });
  leaderboardList.replaceChildren(...(leaderboardItems.length ? leaderboardItems : [emptyListItem("No completed requests yet.")]));
}

async function openRecordDetail(runId) {
  setLeaderboardTab("detail");
  const cachedRecord = recentRecordCache.get(runId);
  if (cachedRecord) {
    renderRecordDetail(cachedRecord);
  } else {
    renderRecordDetailLoading();
  }
  try {
    const params = new URLSearchParams({ leaderboard_id: leaderboardIdentity.id });
    const response = await fetch(`/api/leaderboard/runs/${encodeURIComponent(runId)}?${params.toString()}`);
    if (!response.ok) {
      throw new Error(`Record failed with ${response.status}`);
    }
    renderRecordDetail(await response.json());
  } catch {
    if (cachedRecord) {
      return;
    }
    recordDetailTitle.textContent = "Record unavailable";
    recordDetailMode.textContent = "";
    recordDetailDate.textContent = "";
    recordDetailImage.removeAttribute("src");
    recordDetailText.textContent = "This record could not be opened.";
  }
}

function renderRecordDetailLoading() {
  recordDetailTitle.textContent = "Opening record...";
  recordDetailMode.textContent = "";
  recordDetailDate.textContent = "";
  recordDetailImage.removeAttribute("src");
  recordDetailText.textContent = "";
}

function renderRecordDetail(record) {
  const label = record.mode === "health" ? "Plant Rescue" : "Plant Discovery";
  recordDetailMode.textContent = label;
  recordDetailTitle.textContent = record.leaderboard_name || leaderboardIdentity.name || "Wildkeeper";
  recordDetailDate.textContent = formatDate(record.created_at);
  recordDetailImage.src = record.thumb_url;
  recordDetailImage.alt = label;
  recordDetailText.textContent = record.final_text || "No response text was saved for this record.";
}

function emptyListItem(message) {
  const item = document.createElement("li");
  item.className = "empty-row";
  item.textContent = message;
  return item;
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

async function runQuest(mode) {
  if (!hasSavedName()) {
    runState.textContent = "Name needed";
    nameInput.focus();
    return;
  }
  const image = imageInput.files?.[0];
  if (!image) {
    runState.textContent = "Image needed";
    setJournalText("Choose a plant photo before starting the expedition.");
    return;
  }

  setBusy(true);
  try {
    await unlockAudio();
  } catch {
    audioContext = null;
  }
  hasTextDelta = false;
  audioQueue = [];
  audioPlaying = false;
  audioStarted = false;
  audioStreamDone = false;
  clearPrebufferTimer();
  clearPendingText();
  segmentText = new Map();
  revealedSegments = new Set();
  runState.textContent = mode === "identify" ? "Discovering" : "Rescuing";
  setJournalText("");
  journalText.classList.add("is-waiting");

  const form = new FormData();
  form.append("mode", mode);
  form.append("image", image);
  form.append("leaderboard_id", leaderboardIdentity.id);
  form.append("leaderboard_name", leaderboardIdentity.name);

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
    journalText.classList.remove("is-waiting");
    appendJournalText(`\n\n${error.message}`);
  } finally {
    await waitForPlaybackIdle();
    await waitForTextIdle();
    setBusy(false);
    await loadJournalStats();
    if (!leaderboardModal.hidden) {
      await loadLeaderboard();
    }
  }
}

function handleEvent(event) {
  if (event.type === "status") {
    runState.textContent = event.message;
    if (Number.isFinite(event.text_delay_ms)) {
      balancedTextDelayMs = event.text_delay_ms;
    }
    if (Number.isFinite(event.audio_prebuffer_chunks)) {
      audioPrebufferChunks = Math.max(1, event.audio_prebuffer_chunks);
    }
    if (Number.isFinite(event.audio_prebuffer_max_ms)) {
      audioPrebufferMaxMs = Math.max(0, event.audio_prebuffer_max_ms);
    }
    if (Number.isFinite(event.audio_playback_rate)) {
      audioPlaybackRate = Math.max(0.75, Math.min(1.05, event.audio_playback_rate));
    }
  }
  if (event.type === "text_delta") {
    if (!hasTextDelta) {
      setJournalText("");
      journalText.classList.remove("is-waiting");
      hasTextDelta = true;
    }
    bufferSegmentText(event.segment_id ?? 0, event.delta);
  }
  if (event.type === "audio_chunk") {
    enqueueAudio({
      data: event.data,
      mimeType: event.mime_type || "audio/wav",
      audioFormat: event.audio_format || "wav",
      sampleRate: event.sample_rate || 24000,
      segmentId: event.segment_id,
    });
  }
  if (event.type === "record_saved") {
    runState.textContent = "Journal saved";
  }
  if (event.type === "done") {
    audioStreamDone = true;
    void playNextAudio(true);
    journalText.classList.remove("is-waiting");
    runState.textContent = "Complete";
  }
  if (event.type === "error") {
    audioStreamDone = true;
    void playNextAudio(true);
    journalText.classList.remove("is-waiting");
    runState.textContent = "Error";
    appendJournalText(`\n\n${event.message}`);
  }
}

function setBusy(value) {
  isRunning = value;
  imageInput.disabled = value;
  clearImageBtn.disabled = value;
  updateActionAvailability();
  dropZone.setAttribute("aria-busy", String(value));
  dropZone.classList.toggle("is-disabled", value);
  dropZone.classList.toggle("is-loading", value);
  if (imageLoader) {
    imageLoader.hidden = !value;
  }
}

function setJournalText(value) {
  journalText.textContent = value;
  resultText.textContent = value;
}

function appendJournalText(value) {
  journalText.textContent += value;
  resultText.textContent = journalText.textContent;
  journalText.scrollTop = journalText.scrollHeight;
}

function bufferSegmentText(segmentId, value) {
  const current = segmentText.get(segmentId) || "";
  segmentText.set(segmentId, current + value);
}

function clearPendingText() {
  for (const animation of pendingTextAnimations) {
    window.clearTimeout(animation.timer);
    animation.resolve();
  }
  pendingTextAnimations.clear();
}

function waitForTextIdle() {
  if (!pendingTextAnimations.size) return Promise.resolve();
  return new Promise((resolve) => {
    const check = () => {
      if (!pendingTextAnimations.size) {
        resolve();
      } else {
        window.setTimeout(check, 40);
      }
    };
    check();
  });
}

function enqueueAudio(item) {
  const bytes = base64ToBytes(item.data);
  audioQueue.push({ ...item, bytes });
  void playNextAudio();
}

async function playNextAudio(force = false) {
  if (audioPlaying || !audioQueue.length) return;
  if (!force && !audioStarted && !audioStreamDone && audioQueue.length < audioPrebufferChunks) {
    schedulePrebufferPlayback();
    return;
  }
  clearPrebufferTimer();
  audioStarted = true;
  audioPlaying = true;
  const item = audioQueue.shift();
  try {
    void revealSegmentDuringPlayback(item.segmentId, estimateAudioDurationMs(item));
    if (audioContext) {
      if (item.audioFormat === "pcm") {
        await playPcmWithAudioContext(item.bytes, item.sampleRate);
      } else {
        await playWithAudioContext(item.bytes);
      }
    } else {
      await playWithAudioElement(item.bytes, item.mimeType);
    }
  } catch {
    runState.textContent = "Audio unavailable";
  } finally {
    audioPlaying = false;
    if (audioQueue.length) {
      void playNextAudio();
    } else {
      resolveAudioIdle();
    }
  }
}

function schedulePrebufferPlayback() {
  if (prebufferTimer || audioPrebufferMaxMs <= 0) return;
  prebufferTimer = window.setTimeout(() => {
    prebufferTimer = null;
    void playNextAudio(true);
  }, audioPrebufferMaxMs);
}

function clearPrebufferTimer() {
  if (!prebufferTimer) return;
  window.clearTimeout(prebufferTimer);
  prebufferTimer = null;
}

function estimateAudioDurationMs(item) {
  if (item.audioFormat === "pcm") {
    const frames = Math.floor(item.bytes.byteLength / 2);
    return Math.max(600, Math.round((frames / item.sampleRate / audioPlaybackRate) * 1000));
  }
  return Math.max(1200, ((segmentText.get(item.segmentId) || "").length * 55) / audioPlaybackRate);
}

function revealSegmentDuringPlayback(segmentId, durationMs) {
  if (revealedSegments.has(segmentId)) return Promise.resolve();
  const text = segmentText.get(segmentId) || "";
  revealedSegments.add(segmentId);
  if (!text) return Promise.resolve();

  return new Promise((resolve) => {
    const chars = Array.from(text);
    const stepMs = Math.max(18, Math.min(55, Math.floor((durationMs + balancedTextDelayMs) / Math.max(chars.length, 1))));
    let index = 0;
    const animation = { timer: 0, resolve };
    const tick = () => {
      if (index >= chars.length) {
        pendingTextAnimations.delete(animation);
        resolve();
        return;
      }
      appendJournalText(chars[index]);
      index += 1;
      animation.timer = window.setTimeout(tick, stepMs);
    };
    pendingTextAnimations.add(animation);
    tick();
  });
}

async function unlockAudio() {
  const Context = window.AudioContext || window.webkitAudioContext;
  if (!Context) return;
  audioContext = audioContext || new Context();
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }
  const source = audioContext.createBufferSource();
  source.buffer = audioContext.createBuffer(1, 1, 24000);
  source.connect(audioContext.destination);
  source.start();
}

async function playWithAudioContext(bytes) {
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }
  const arrayBuffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const decoded = await audioContext.decodeAudioData(arrayBuffer);
  await new Promise((resolve) => {
    const source = audioContext.createBufferSource();
    source.buffer = decoded;
    source.playbackRate.value = audioPlaybackRate;
    source.connect(audioContext.destination);
    source.addEventListener("ended", resolve, { once: true });
    source.start();
  });
}

async function playPcmWithAudioContext(bytes, sampleRate) {
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }
  const frameCount = Math.floor(bytes.byteLength / 2);
  const buffer = audioContext.createBuffer(1, frameCount, sampleRate);
  const channel = buffer.getChannelData(0);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < frameCount; index += 1) {
    channel[index] = view.getInt16(index * 2, true) / 32768;
  }
  applyShortFade(channel, sampleRate);
  await new Promise((resolve) => {
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.playbackRate.value = audioPlaybackRate;
    source.connect(audioContext.destination);
    source.addEventListener("ended", resolve, { once: true });
    source.start();
  });
}

function applyShortFade(channel, sampleRate) {
  const fadeFrames = Math.min(Math.floor(sampleRate * 0.008), Math.floor(channel.length / 2));
  if (fadeFrames <= 1) return;
  for (let index = 0; index < fadeFrames; index += 1) {
    const gain = index / fadeFrames;
    channel[index] *= gain;
    channel[channel.length - 1 - index] *= gain;
  }
}

async function playWithAudioElement(bytes, mimeType) {
  const blob = new Blob([bytes], { type: mimeType });
  const url = URL.createObjectURL(blob);
  try {
    const audio = new Audio(url);
    audio.playbackRate = audioPlaybackRate;
    await new Promise((resolve, reject) => {
      audio.addEventListener("ended", resolve, { once: true });
      audio.addEventListener("error", reject, { once: true });
      audio.play().catch(reject);
    });
  } finally {
    URL.revokeObjectURL(url);
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

function waitForPlaybackIdle() {
  if (!audioPlaying && !audioQueue.length) return Promise.resolve();
  return new Promise((resolve) => {
    audioIdleResolvers.push(resolve);
  });
}

function resolveAudioIdle() {
  const resolvers = audioIdleResolvers;
  audioIdleResolvers = [];
  for (const resolve of resolvers) {
    resolve();
  }
}

syncNameUi();
loadJournalStats();
