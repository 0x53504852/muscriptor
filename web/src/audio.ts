import * as Tone from "tone";
import { Soundfont, DrumMachine } from "smplr";

/** Map a muscriptor instrument-group name to a smplr Soundfont instrument name. */
function smplrName(name: string): string {
  const direct: Record<string, string> = {
    acoustic_piano: "acoustic_grand_piano",
    electric_piano: "electric_piano_1",
    chromatic_percussion: "glockenspiel",
    organ: "church_organ",
    acoustic_guitar: "acoustic_guitar_nylon",
    clean_electric_guitar: "electric_guitar_clean",
    distorted_electric_guitar: "distortion_guitar",
    acoustic_bass: "acoustic_bass",
    electric_bass: "electric_bass_finger",
    violin: "violin",
    viola: "viola",
    cello: "cello",
    contrabass: "contrabass",
    orchestral_harp: "orchestral_harp",
    timpani: "timpani",
    string_ensemble: "string_ensemble_1",
    synth_strings: "synth_strings_1",
    choir: "choir_aahs",
    orchestra_hit: "orchestra_hit",
    trumpet: "trumpet",
    trombone: "trombone",
    tuba: "tuba",
    horn: "french_horn",
    brass_section: "brass_section",
    soprano_alto_sax: "alto_sax",
    tenor_sax: "tenor_sax",
    baritone_sax: "baritone_sax",
    oboe: "oboe",
    english_horn: "english_horn",
    bassoon: "bassoon",
    clarinet: "clarinet",
    pipe: "flute",
    synth_lead: "lead_1_square",
    synth_pad: "pad_2_warm",
  };
  return direct[name] ?? "acoustic_grand_piano";
}

/** Map a GM drum-program pitch to a smplr DrumMachine drum name. */
function drumNameForPitch(pitch: number): string | null {
  // GM drum mapping → TR-808 voice names available in smplr's DrumMachine.
  const m: Record<number, string> = {
    35: "kick",
    36: "kick",
    38: "snare",
    40: "snare",
    42: "hihat-close",
    44: "hihat-close",
    46: "hihat-open",
    49: "crash",
    51: "ride",
    50: "tom-high",
    47: "tom-mid",
    45: "tom-low",
  };
  return m[pitch] ?? null;
}

export class AudioEngine {
  private ctx!: AudioContext;
  private samplers = new Map<string, Soundfont>();
  private drums?: DrumMachine;
  /** Notes whose loading was deferred while their sampler was still fetching. */
  private pendingByInstrument = new Map<
    string,
    Array<{ pitch: number; start: number; duration: number }>
  >();
  /** All notes ever scheduled — kept so Play-from-0 can re-schedule them. */
  private allNotes: Array<{
    instrument: string;
    pitch: number;
    start: number;
    end: number;
  }> = [];
  private autoStopAt: number | null = null;
  /** Decoded original audio + live playback node, for the WAV/MIDI mix. */
  private wavBuffer: AudioBuffer | null = null;
  private wavSource: AudioBufferSourceNode | null = null;
  private wavGain: GainNode;
  private midiGain: GainNode;
  private mix = 0.5; // 0 = full WAV, 1 = full MIDI

  constructor() {
    // Tone.getContext() lazily creates a (suspended) AudioContext — no user
    // gesture needed. Samplers can load and decode samples while suspended;
    // we only need to call Tone.start() to *resume* it on the play click.
    this.ctx = Tone.getContext().rawContext as AudioContext;
    this.midiGain = this.ctx.createGain();
    this.midiGain.connect(this.ctx.destination);
    this.wavGain = this.ctx.createGain();
    this.wavGain.connect(this.ctx.destination);
    this.drums = new DrumMachine(this.ctx, {
      instrument: "TR-808",
      destination: this.midiGain,
    });
    this.applyMix();
    Tone.getTransport().on("start", (time) => this.startWavSource(time));
    Tone.getTransport().on("stop", () => this.stopWavSource());
  }

  /** Decode `file` and remember the buffer for synced WAV playback. */
  async loadWav(file: File) {
    this.stopWavSource();
    this.wavBuffer = null;
    const bytes = await file.arrayBuffer();
    try {
      this.wavBuffer = await this.ctx.decodeAudioData(bytes);
    } catch {
      // Non-decodable input — keep MIDI-only playback.
      this.wavBuffer = null;
    }
  }

  /** Set the MIDI/WAV crossfade. `midiAmount` in [0, 1]. */
  setMix(midiAmount: number) {
    this.mix = Math.max(0, Math.min(1, midiAmount));
    this.applyMix();
  }

  private applyMix() {
    const t = this.ctx.currentTime;
    this.wavGain.gain.setTargetAtTime(1 - this.mix, t, 0.01);
    this.midiGain.gain.setTargetAtTime(this.mix, t, 0.01);
  }

  private startWavSource(at: number) {
    this.stopWavSource();
    if (!this.wavBuffer) return;
    const offset = Tone.getTransport().seconds;
    if (offset >= this.wavBuffer.duration) return;
    const src = this.ctx.createBufferSource();
    src.buffer = this.wavBuffer;
    src.connect(this.wavGain);
    src.start(at, offset);
    this.wavSource = src;
  }

  private stopWavSource() {
    if (!this.wavSource) return;
    try {
      this.wavSource.stop();
    } catch {
      // already stopped
    }
    this.wavSource.disconnect();
    this.wavSource = null;
  }

  /** Schedule a note at transport time `start` for `duration` seconds. */
  scheduleNote(opts: {
    instrument: string;
    pitch: number;
    start: number;
    end: number;
  }) {
    this.allNotes.push({ ...opts });
    this.scheduleNoteRaw(opts);
  }

  private scheduleNoteRaw(opts: {
    instrument: string;
    pitch: number;
    start: number;
    end: number;
  }) {
    const duration = Math.max(0.05, opts.end - opts.start);
    if (opts.instrument === "drums") {
      const name = drumNameForPitch(opts.pitch);
      if (!name || !this.drums) return;
      const drums = this.drums;
      Tone.getTransport().scheduleOnce((time) => {
        drums.start({ note: name, time });
      }, opts.start);
      return;
    }

    const key = opts.instrument;
    let sampler = this.samplers.get(key);
    if (!sampler) {
      sampler = new Soundfont(this.ctx, {
        instrument: smplrName(key),
        destination: this.midiGain,
      });
      this.samplers.set(key, sampler);
      this.pendingByInstrument.set(key, []);
      // Once loaded, flush any notes that were queued while waiting.
      sampler.load.then(() => {
        const queued = this.pendingByInstrument.get(key) ?? [];
        this.pendingByInstrument.delete(key);
        for (const n of queued) this.scheduleOn(sampler!, n.pitch, n.start, n.duration);
      });
      // First note for this instrument goes into the pending queue.
      this.pendingByInstrument.get(key)!.push({
        pitch: opts.pitch,
        start: opts.start,
        duration,
      });
      return;
    }
    // If still loading, queue. Otherwise schedule immediately.
    const pending = this.pendingByInstrument.get(key);
    if (pending) {
      pending.push({ pitch: opts.pitch, start: opts.start, duration });
    } else {
      this.scheduleOn(sampler, opts.pitch, opts.start, duration);
    }
  }

  private scheduleOn(
    sampler: Soundfont,
    pitch: number,
    start: number,
    duration: number,
  ) {
    Tone.getTransport().scheduleOnce((time) => {
      sampler.start({ note: pitch, time, duration });
    }, start);
  }

  /** Resume the AudioContext. Must be called from a user-gesture handler. */
  async unlock() {
    await Tone.start();
  }

  async play() {
    await Tone.start();
    const t = Tone.getTransport();
    // If we're at the very start, every previously-fired scheduleOnce is gone;
    // re-schedule every known note (and the auto-stop) from scratch.
    if (t.seconds < 0.001) {
      t.cancel();
      for (const n of this.allNotes) this.scheduleNoteRaw(n);
      if (this.autoStopAt !== null) this.scheduleStopRaw(this.autoStopAt);
    }
    t.start();
  }

  pause() {
    Tone.getTransport().pause();
  }

  stop() {
    const t = Tone.getTransport();
    t.stop();
    t.cancel();
    t.seconds = 0;
  }

  /** Wipe scheduled state for a brand-new transcription. */
  reset() {
    this.stop();
    this.allNotes = [];
    this.autoStopAt = null;
    this.stopWavSource();
    this.wavBuffer = null;
  }

  /** Remember the auto-stop time and schedule it for the current run. */
  scheduleStop(t: number) {
    this.autoStopAt = t;
    this.scheduleStopRaw(t);
  }

  private scheduleStopRaw(at: number) {
    Tone.getTransport().scheduleOnce(() => {
      Tone.getTransport().stop();
      Tone.getTransport().seconds = 0;
    }, at);
  }

  /** Current playback position in seconds. */
  get seconds(): number {
    return Tone.getTransport().seconds;
  }

  get state(): "started" | "stopped" | "paused" {
    return Tone.getTransport().state;
  }
}
