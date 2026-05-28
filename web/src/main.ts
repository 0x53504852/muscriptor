import { streamTranscribe } from "./sse";
import { AudioEngine } from "./audio";
import { PianoRoll, RollNote, instrumentColor } from "./pianoroll";

type StartEvent = {
  type: "start";
  pitch: number;
  start_time: number;
  index: number;
  instrument: string;
};
type EndEvent = {
  type: "end";
  end_time: number;
  start_event_index: number;
};
type StreamedEvent = StartEvent | EndEvent;

const dropzone = document.getElementById("dropzone")!;
const fileButton = document.getElementById("file-button")! as HTMLButtonElement;
const fileInput = document.getElementById("file-input")! as HTMLInputElement;
const status = document.getElementById("status")!;
const controls = document.getElementById("controls")!;
const playBtn = document.getElementById("play")! as HTMLButtonElement;
const stopBtn = document.getElementById("stop")! as HTMLButtonElement;
const followBtn = document.getElementById("follow")! as HTMLButtonElement;
const mixSlider = document.getElementById("mix")! as HTMLInputElement;
const clock = document.getElementById("clock")!;
const canvas = document.getElementById("piano-roll")! as HTMLCanvasElement;
const instrumentList = document.getElementById("instrument-list")!;
const condList = document.getElementById("conditioning-list")!;
const condClear = document.getElementById("cond-none") as HTMLButtonElement;

async function loadConditioningInstruments() {
  try {
    const resp = await fetch("/instruments");
    if (!resp.ok) return;
    const data = (await resp.json()) as { instruments: string[] };
    for (const name of data.instruments) {
      const label = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = name;
      cb.name = "cond-instrument";
      label.appendChild(cb);
      label.appendChild(document.createTextNode(name.replace(/_/g, " ")));
      condList.appendChild(label);
    }
  } catch {
    /* ignore — server may not expose endpoint */
  }
}
loadConditioningInstruments();

condClear.addEventListener("click", () => {
  condList
    .querySelectorAll<HTMLInputElement>("input[type=checkbox]")
    .forEach((c) => (c.checked = false));
});

function selectedConditioningInstruments(): string[] {
  return Array.from(
    condList.querySelectorAll<HTMLInputElement>("input[type=checkbox]:checked"),
  ).map((c) => c.value);
}

const roll = new PianoRoll(canvas);
const audio = new AudioEngine();
// Expose for browser-devtools debugging.
(window as unknown as { __mu: unknown }).__mu = { audio, roll };
const knownInstruments = new Set<string>();
const openNotes = new Map<number, { startNote: RollNote }>();

function addInstrument(name: string) {
  if (knownInstruments.has(name)) return;
  knownInstruments.add(name);
  const li = document.createElement("li");
  const swatch = document.createElement("span");
  swatch.className = "swatch";
  swatch.style.background = instrumentColor(name);
  li.appendChild(swatch);
  li.appendChild(document.createTextNode(name));
  instrumentList.appendChild(li);
}

function reset() {
  roll.clear();
  knownInstruments.clear();
  instrumentList.innerHTML = "";
  openNotes.clear();
  audio.reset();
  followBtn.setAttribute("hidden", "");
}

function onEvent(ev: StreamedEvent) {
  if (ev.type === "start") {
    addInstrument(ev.instrument);
    // Provisional end = start (will be patched on the matching end event).
    const note: RollNote = {
      pitch: ev.pitch,
      start: ev.start_time,
      end: ev.start_time,
      instrument: ev.instrument,
    };
    roll.addNote(note);
    openNotes.set(ev.index, { startNote: note });
  } else {
    const open = openNotes.get(ev.start_event_index);
    if (!open) return;
    open.startNote.end = ev.end_time;
    openNotes.delete(ev.start_event_index);
    audio.scheduleNote({
      instrument: open.startNote.instrument,
      pitch: open.startNote.pitch,
      start: open.startNote.start,
      end: open.startNote.end,
    });
  }
}

const AUTOPLAY_AFTER = 10;

async function transcribe(file: File) {
  reset();
  status.textContent = `Transcribing ${file.name}…`;
  controls.removeAttribute("hidden");
  // Resume the AudioContext now, while we still have the user-gesture
  // activation from the drop / file-pick. Without this, calling play()
  // later — even from a click — sometimes can't unlock the context.
  audio.unlock().catch(() => {});
  // Decode the WAV in parallel with the transcription so it's ready by the
  // time autoplay kicks in.
  audio.loadWav(file).catch(() => {});
  let eventsSeen = 0;
  let started = false;
  let maxEnd = 0;
  try {
    const cond = selectedConditioningInstruments();
    const extra = cond.length > 0 ? { instruments: cond } : undefined;
    for await (const raw of streamTranscribe("/transcribe", file, extra)) {
      const ev = raw as StreamedEvent;
      onEvent(ev);
      if (ev.type === "end" && ev.end_time > maxEnd) maxEnd = ev.end_time;
      eventsSeen++;
      if (!started && eventsSeen >= AUTOPLAY_AFTER) {
        started = true;
        audio.play();
      }
    }
    if (!started) audio.play();
    // Stop the transport ~0.3 s after the last note ends.
    if (maxEnd > 0) audio.scheduleStop(maxEnd + 0.3);
    status.textContent = `Done — ${file.name}`;
  } catch (e) {
    status.textContent = `Error: ${(e as Error).message}`;
  }
}

// Drag & drop
["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  }),
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  }),
);
// Allow dropping anywhere on the page too.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

dropzone.addEventListener("drop", (e) => {
  const dt = (e as DragEvent).dataTransfer;
  const f = dt?.files?.[0];
  if (f) transcribe(f);
});

fileButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  if (f) transcribe(f);
});

mixSlider.addEventListener("input", () => {
  audio.setMix(parseFloat(mixSlider.value));
});
audio.setMix(parseFloat(mixSlider.value));

playBtn.addEventListener("click", () => audio.play());
stopBtn.addEventListener("click", () => audio.stop());
followBtn.addEventListener("click", () => {
  roll.follow();
  followBtn.setAttribute("hidden", "");
});

// Wheel = horizontal scroll. Use deltaY when there's no deltaX (vertical wheel
// from a typical mouse) so it still works without holding shift.
canvas.addEventListener(
  "wheel",
  (e) => {
    const delta = e.deltaX !== 0 ? e.deltaX : e.deltaY;
    if (delta === 0) return;
    e.preventDefault();
    roll.scrollBy(delta / roll.pxPerSec);
    followBtn.removeAttribute("hidden");
  },
  { passive: false },
);

// Click-and-drag to scrub the timeline.
let dragStartX: number | null = null;
let dragStartOffset = 0;
canvas.addEventListener("mousedown", (e) => {
  dragStartX = e.clientX;
  dragStartOffset = roll.xToSeconds(0); // current left edge
});
window.addEventListener("mousemove", (e) => {
  if (dragStartX === null) return;
  const dx = e.clientX - dragStartX;
  const targetOffset = dragStartOffset - dx / roll.pxPerSec;
  roll.scrollBy(targetOffset - roll.xToSeconds(0));
  followBtn.removeAttribute("hidden");
});
window.addEventListener("mouseup", () => {
  dragStartX = null;
});

function frame() {
  roll.setPlayhead(audio.seconds);
  roll.render();
  clock.textContent = `${audio.seconds.toFixed(1)}s`;
  requestAnimationFrame(frame);
}
frame();
