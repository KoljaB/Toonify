/*!
 * Toonify — player.js
 * Loads runs/<id>/avatar.psd + the three expression frames + layers.json,
 * rigs them with Anime2.5DRig's rigger and drives mouth states from StreamAudio.
 */
(function(){
'use strict';
const $=id=>document.getElementById(id);
const Q=new URLSearchParams(location.search);
const runId=Q.get('run');

function status(text){$('status').textContent=text;}

async function loadImageData(url){
  const img=new Image();img.src=url;await img.decode();
  const c=document.createElement('canvas');c.width=img.width;c.height=img.height;
  const g=c.getContext('2d');g.drawImage(img,0,0);
  return g.getImageData(0,0,img.width,img.height);
}
function leaf(name,imageData,left,top){
  return {name,imageData,left,top,right:left+imageData.width,bottom:top+imageData.height,opacity:1};
}
function lastIndex(arr,pred){for(let i=arr.length-1;i>=0;i--)if(pred(arr[i]))return i;return -1;}

// see-through names -> rigger slots, then splice the expression frames in above their layers.
async function buildPsdTree(base,info){
  const psd=agPsd.readPsd(await (await fetch(base+info.psd)).arrayBuffer(),{useImageData:true});
  const kids=psd.children;
  for(const c of kids)if(/^mouth$/i.test(c.name))c.name='mouth_close';
  const tag=n=>String(n).toLowerCase();
  for(const ex of info.expressions||[]){
    const img=await loadImageData(base+ex.file);
    const L=leaf(ex.slot,img,ex.bbox[0],ex.bbox[1]);
    let at;
    if(ex.slot==='eye_close')at=lastIndex(kids,c=>/^(eyewhite|irides|eyelash|eyes?)/.test(tag(c.name)));
    else at=lastIndex(kids,c=>/^mouth/.test(tag(c.name)));
    if(at<0)at=lastIndex(kids,c=>tag(c.name)==='face');
    kids.splice(at+1,0,L);
  }
  return psd;
}

function layout(rig){
  // Frame the visible character, not the square see-through canvas.
  let x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity;
  for(const L of rig.layers){x0=Math.min(x0,L.x);y0=Math.min(y0,L.y);x1=Math.max(x1,L.x+L.w);y1=Math.max(y1,L.y+L.h);}
  const pad=0.04*(y1-y0);x0-=pad;x1+=pad;y0-=pad;
  const stage=$('stage'),cv=$('cv');
  const fit=()=>{
    const s=Math.min(innerWidth/(x1-x0),innerHeight/(y1-y0));
    stage.style.width=(x1-x0)*s+'px';stage.style.height=Math.min(innerHeight,(y1-y0)*s)+'px';
    cv.style.width=rig.canvas.w*s+'px';cv.style.height=rig.canvas.h*s+'px';
    cv.style.left=-x0*s+'px';cv.style.top=-y0*s+'px';
  };
  addEventListener('resize',fit);fit();
}

async function main(){
  if(!runId){status('add ?run=<run id> to the URL');return;}
  const base='/runs/'+encodeURIComponent(runId)+'/';
  status('loading '+runId+' …');
  const info=await (await fetch(base+'layers.json')).json();
  const psd=await buildPsdTree(base,info);
  const rig=Rigger.buildRig(psd);
  if(rig.warnings.length)console.info('rig warnings:',rig.warnings);
  const cv=$('cv');cv.width=rig.canvas.w;cv.height=rig.canvas.h;
  const gl=cv.getContext('webgl',{alpha:true,antialias:true,premultipliedAlpha:true});
  if(!gl){status('WebGL is not available');return;}
  const puppet=ToonRig.create(gl,rig);
  layout(rig);

  const opts={};
  for(const k of ['gate','halfOn','halfOff','openOn','openOff','attack','release','peakDecay','prebuffer'])if(Q.has(k))opts[k]=parseFloat(Q.get(k));
  const audio=new StreamAudio(opts);
  window.toonAudio=audio;   // for debugging in the console
  const wsUrl=(location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws/play';
  let link='connecting';
  audio.onstatus=s=>{link=s;};
  audio.connect(wsUrl);
  if(!window.isSecureContext){
    $('unlock').textContent='Audio needs https or localhost (AudioWorklet); open this page via localhost or https';
    $('unlock').disabled=true;
  }
  const unlock=async()=>{await audio.unlock();$('unlock').hidden=true;};
  $('unlock').addEventListener('click',unlock);

  let last=performance.now(),fpsN=0,fpsT=last,fps=0;
  const names=['closed','half','open'];
  function tick(now){
    requestAnimationFrame(tick);
    const dt=Math.max(0,Math.min(0.05,(now-last)/1000));last=now;
    const m=audio.mouth(dt);
    puppet.frame(dt,m);
    fpsN++;if(now-fpsT>500){fps=Math.round(fpsN*1000/(now-fpsT));fpsN=0;fpsT=now;}
    status(`${runId} · ${fps} fps · ws ${link} · audio ${audio.running?'on':'locked'} · mouth ${names[m]}`);
  }
  requestAnimationFrame(tick);
}
main().catch(err=>{console.error(err);status('error: '+err.message);});
})();
