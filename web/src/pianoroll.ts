export interface RollNote {
  pitch: number;
  start: number;
  end: number;
  instrument: string;
}

const PX_PER_SEC = 80;
const PITCH_MIN = 21; // A0
const PITCH_MAX = 108; // C8
const PITCH_RANGE = PITCH_MAX - PITCH_MIN + 1; // 88

/** Stable hue per instrument name. */
function hashHue(name: string): number {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return h % 360;
}

export function instrumentColor(name: string): string {
  return `hsl(${hashHue(name)}, 70%, 55%)`;
}

export class PianoRoll {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private notes: RollNote[] = [];
  private playhead = 0;
  /** Seconds-from-start of the left edge when the user has taken over scrolling. */
  private userOffset: number | null = null;
  /** Last offset we rendered with — handy so external code knows what's visible. */
  private lastOffset = 0;
  readonly pxPerSec = PX_PER_SEC;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  private resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  addNote(n: RollNote) {
    this.notes.push(n);
  }

  setPlayhead(seconds: number) {
    this.playhead = seconds;
  }

  clear() {
    this.notes = [];
    this.playhead = 0;
    this.userOffset = null;
  }

  /** Add `deltaSeconds` to the left-edge offset, taking the view out of follow mode. */
  scrollBy(deltaSeconds: number) {
    const base = this.userOffset ?? this.lastOffset;
    this.userOffset = Math.max(0, base + deltaSeconds);
  }

  /** Resume auto-following the playhead. */
  follow() {
    this.userOffset = null;
  }

  get isFollowing(): boolean {
    return this.userOffset === null;
  }

  /** Translate an x pixel coordinate to a transport-time in seconds. */
  xToSeconds(x: number): number {
    return Math.max(0, this.lastOffset + x / PX_PER_SEC);
  }

  /** Latest note end across all notes — drives x-axis scroll. */
  private maxEnd(): number {
    let m = 0;
    for (const n of this.notes) if (n.end > m) m = n.end;
    return m;
  }

  render() {
    const ctx = this.ctx;
    const rect = this.canvas.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;
    ctx.clearRect(0, 0, W, H);

    // Auto-scroll: keep playhead in the right third of the view, unless the
    // user has manually scrolled (userOffset !== null) in which case we honor it.
    const maxT = Math.max(this.maxEnd(), this.playhead + 1);
    const viewSec = W / PX_PER_SEC;
    let offsetSec: number;
    if (this.userOffset !== null) {
      offsetSec = this.userOffset;
    } else {
      offsetSec = this.playhead > viewSec * 0.66 ? this.playhead - viewSec * 0.66 : 0;
      offsetSec = Math.min(offsetSec, Math.max(0, maxT - viewSec));
    }
    this.lastOffset = offsetSec;

    const rowH = H / PITCH_RANGE;

    // Pitch grid: faint horizontal stripes for C notes
    ctx.fillStyle = "#16161c";
    for (let p = PITCH_MIN; p <= PITCH_MAX; p++) {
      if (p % 12 === 0) {
        const y = (PITCH_MAX - p) * rowH;
        ctx.fillRect(0, y, W, rowH);
      }
    }

    // Time grid: vertical lines every second.
    ctx.strokeStyle = "#22222a";
    ctx.lineWidth = 1;
    const startSec = Math.floor(offsetSec);
    const endSec = Math.ceil(offsetSec + viewSec);
    for (let s = startSec; s <= endSec; s++) {
      const x = (s - offsetSec) * PX_PER_SEC;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, H);
      ctx.stroke();
    }

    // Notes
    for (const n of this.notes) {
      if (n.pitch < PITCH_MIN || n.pitch > PITCH_MAX) continue;
      const x = (n.start - offsetSec) * PX_PER_SEC;
      const w = Math.max(2, (n.end - n.start) * PX_PER_SEC);
      if (x + w < 0 || x > W) continue;
      const y = (PITCH_MAX - n.pitch) * rowH;
      ctx.fillStyle = instrumentColor(n.instrument);
      ctx.fillRect(x, y, w, Math.max(2, rowH - 0.5));
    }

    // Playhead
    const px = (this.playhead - offsetSec) * PX_PER_SEC;
    ctx.strokeStyle = "#ff5577";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, H);
    ctx.stroke();
  }
}
