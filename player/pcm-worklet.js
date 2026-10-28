/*!
 * Toonify — pcm-worklet.js
 * Continuous playback of streamed PCM on the audio thread, modelled on miep's tts-worklet.js:
 * samples are copied out of one queue every render quantum, so chunk boundaries never click.
 * A real underrun fades out over 64 samples and fades back in when audio resumes.
 */
class ToonifyPcmProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.prebuffer = options?.processorOptions?.prebuffer ?? 0;
    this.waiting = true;   // hold output until `prebuffer` samples are queued (or the stream ended)
    this.buffers = [];
    this.offset = 0;
    this.queued = 0;
    this.fadeSamples = 64;
    this.fadeIn = this.fadeSamples;
    this.lastSample = 0;
    this.started = false;
    this.ending = false;
    this.underruns = 0;
    this.sinceReport = 0;
    this.port.onmessage = ({ data }) => {
      if (data.type === "pcm") {
        this.buffers.push(data.samples);
        this.queued += data.samples.length;
        this.ending = false;
      } else if (data.type === "end") {
        this.ending = true;
      }
    };
  }

  report(force) {
    if (force || this.sinceReport >= 512) {
      this.port.postMessage({ type: "queue", queued: this.queued, underruns: this.underruns });
      this.sinceReport = 0;
    }
  }

  fadeOut(output, start) {
    const ramp = Math.min(this.fadeSamples, output.length - start);
    for (let i = 0; i < ramp; i++) output[start + i] = this.lastSample * (1 - (i + 1) / ramp);
    output.fill(0, start + ramp);
    this.lastSample = 0;
    this.fadeIn = this.fadeSamples;
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    let written = 0;
    if (this.waiting && (this.queued >= this.prebuffer || (this.ending && this.queued > 0))) this.waiting = false;
    while (!this.waiting && written < output.length && this.buffers.length) {
      const cur = this.buffers[0];
      const n = Math.min(output.length - written, cur.length - this.offset);
      for (let i = 0; i < n; i++) {
        let s = cur[this.offset + i];
        if (this.fadeIn > 0) {
          s *= (this.fadeSamples - this.fadeIn + 1) / this.fadeSamples;
          this.fadeIn--;
        }
        output[written + i] = s;
        this.lastSample = s;
      }
      written += n;
      this.offset += n;
      this.queued -= n;
      if (this.offset === cur.length) { this.buffers.shift(); this.offset = 0; }
    }
    if (written) this.started = true;
    if (written < output.length) {
      if (this.started && !this.ending) this.underruns++;
      this.fadeOut(output, written);
      this.waiting = true;
      if (this.started) { this.started = false; this.report(true); }
    }
    this.sinceReport += output.length;
    this.report(false);
    return true;
  }
}

registerProcessor("toonify-pcm", ToonifyPcmProcessor);
