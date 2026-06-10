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
const journalText = document.getElementById("journalText");
const imageLoader = document.getElementById("imageLoader");
const speciesFoundCount = document.getElementById("speciesFoundCount");
const plantsRescuedCount = document.getElementById("plantsRescuedCount");

let isRunning = false;
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

let balancedTextDelayMs = 600;
let audioPrebufferChunks = 2;
let audioPrebufferMaxMs = 2200;
let audioPlaybackRate = 0.92;

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
    journalFound.textContent = species;
    speciesFoundCount.textContent = species;
    plantsRescuedCount.textContent = rescues;
  } catch {
    journalFound.textContent = "0";
    speciesFoundCount.textContent = "0";
    plantsRescuedCount.textContent = "0";
  }
}

async function runQuest(mode) {
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
  identifyBtn.disabled = value;
  healthBtn.disabled = value;
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

loadJournalStats();
