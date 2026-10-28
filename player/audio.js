/*!
 * Toonify — audio.js
 * Gapless playback of raw PCM chunks from a WebSocket (AudioWorklet queue), and RMS -> 3 mouth states.
 *
 * Protocol (see README):
 *   text   {"type":"start","sample_rate":24000,"format":"int16"|"float32","channels":1}
 *   binary raw little-endian PCM, any size (partial samples are carried over)
 *   text   {"type":"end"}   last chunk of an utterance has been sent
 */
(function(root){
'use strict';

const DEFAULTS={
  prebuffer:0.10,   // s queued before playback (re)starts; absorbs network/producer jitter
  gate:0.008,       // absolute RMS below which the mouth is always closed
  // Thresholds (with hysteresis) on the envelope relative to the recent speech peak,
  // so loud and quiet voices both use the full closed/half/open range.
  halfOn:0.30,halfOff:0.20,
  openOn:0.70,openOff:0.55,
  attack:0.012,release:0.060,   // s, envelope smoothing time constants
  peakDecay:2.0,                // s, how fast the speech peak forgets loud passages
};
const WORKLET_URL=new URL('pcm-worklet.js',document.currentScript.src).href;

// Playback runs in an AudioWorklet that copies samples out of one queue on the audio thread (the approach
// of miep's tts-worklet). Separate AudioBufferSourceNodes per chunk clicked at the chunk joins in practice.
class StreamAudio{
  constructor(opts={}){
    this.o=Object.assign({},DEFAULTS,opts);
    this.ctx=null;this.node=null;this.analyser=null;this.buf=null;this.ready=null;
    this.sampleRate=24000;this.format='int16';this.carry=null;
    this.queued=0;this.underruns=0;this.env=0;this.peak=0;this.state=0;
    this.onstatus=()=>{};
  }
  // The context runs at the stream's sample rate, so the browser resamples the output once, continuously.
  makeContext(rate){
    if(this.ctx)this.ctx.close();
    const ctx=this.ctx=new AudioContext({latencyHint:'interactive',sampleRate:rate});
    this.analyser=ctx.createAnalyser();
    this.analyser.fftSize=rate>32000?1024:512;   // ~21 ms window
    this.analyser.connect(ctx.destination);
    this.buf=new Float32Array(this.analyser.fftSize);
    this.node=null;this.queued=0;
    this.ready=ctx.audioWorklet.addModule(WORKLET_URL).then(()=>{
      if(ctx!==this.ctx)return;
      this.node=new AudioWorkletNode(ctx,'toonify-pcm',{numberOfInputs:0,outputChannelCount:[1],
        processorOptions:{prebuffer:Math.round(this.o.prebuffer*rate)}});
      this.node.port.onmessage=({data})=>{if(data.type==='queue'){this.queued=data.queued;this.underruns=data.underruns;}};
      this.node.connect(this.analyser);
    });
    if(this.unlocked)ctx.resume();
  }
  // Must be called from a user gesture (autoplay policy).
  async unlock(){
    this.unlocked=true;
    if(!this.ctx)this.makeContext(this.sampleRate);
    if(this.ctx.state!=='running')await this.ctx.resume();
    await this.ready;
  }
  get running(){return !!this.ctx&&this.ctx.state==='running'&&!!this.node;}
  connect(url){
    const open=()=>{
      const ws=new WebSocket(url);ws.binaryType='arraybuffer';
      ws.onopen=()=>this.onstatus('connected');
      ws.onclose=()=>{this.onstatus('disconnected');setTimeout(open,1000);};
      ws.onmessage=ev=>{
        if(typeof ev.data==='string'){
          let msg;try{msg=JSON.parse(ev.data);}catch(e){return;}
          if(msg.type==='start')this.start(msg);
          else if(msg.type==='end')this.end();
        }else this.chunk(ev.data);
      };
    };
    open();
  }
  start(msg){
    this.sampleRate=msg.sample_rate|0||24000;
    if(this.ctx&&this.ctx.sampleRate!==this.sampleRate)this.makeContext(this.sampleRate);
    this.format=msg.format==='float32'?'float32':'int16';
    if((msg.channels|0)>1)console.warn('toonify: only mono is supported; channels =',msg.channels);
    this.carry=null;
  }
  chunk(ab){
    if(!this.running)return;                 // audio not unlocked yet: drop instead of piling up
    let bytes=new Uint8Array(ab);
    if(this.carry){const m=new Uint8Array(this.carry.length+bytes.length);m.set(this.carry);m.set(bytes,this.carry.length);bytes=m;this.carry=null;}
    const width=this.format==='int16'?2:4,n=Math.floor(bytes.length/width);
    if(bytes.length>n*width)this.carry=bytes.slice(n*width);
    if(!n)return;
    const aligned=bytes.byteOffset===0&&bytes.buffer.byteLength>=n*width?bytes.buffer:bytes.slice().buffer;
    const f=new Float32Array(n);
    if(width===2){const s=new Int16Array(aligned,0,n);for(let i=0;i<n;i++)f[i]=s[i]/32768;}
    else f.set(new Float32Array(aligned,0,n));
    this.queued+=n;
    this.node.port.postMessage({type:'pcm',samples:f},[f.buffer]);
  }
  end(){this.carry=null;if(this.node)this.node.port.postMessage({type:'end'});}
  // Queue empty (as reported by the audio thread) = nothing left to play.
  get playing(){return this.running&&this.queued>0;}
  // Called once per animation frame. Returns 0 closed, 1 half, 2 open.
  mouth(dt){
    if(!this.playing){this.env=0;this.state=0;return 0;}
    this.analyser.getFloatTimeDomainData(this.buf);
    let s=0;for(let i=0;i<this.buf.length;i++)s+=this.buf[i]*this.buf[i];
    const rms=Math.sqrt(s/this.buf.length),o=this.o;
    const tau=rms>this.env?o.attack:o.release;
    this.env+=(rms-this.env)*(1-Math.exp(-dt/tau));
    this.peak=Math.max(this.env,this.peak*Math.exp(-dt/o.peakDecay));
    const e=this.env<o.gate?0:this.env/Math.max(this.peak,o.gate*4);let st=this.state;
    if(st===0){if(e>o.openOn)st=2;else if(e>o.halfOn)st=1;}
    else if(st===1){if(e>o.openOn)st=2;else if(e<o.halfOff)st=0;}
    else{if(e<o.halfOff)st=0;else if(e<o.openOff)st=1;}
    this.state=st;this.rms=rms;this.level=e;
    return st;
  }
}
root.StreamAudio=StreamAudio;
})(typeof self!=='undefined'?self:this);
