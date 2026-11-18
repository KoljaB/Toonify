/*!
 * Toonify — rig.js
 * Idle-animated layered puppet. The mesh deformation, hair spring physics, blink
 * and breathing below are taken from Anime2.5DRig's app.js (MIT, see
 * vendor/LICENSE-Anime2.5DRig) with the editor, OBS, camera, microphone and
 * export code stripped. Toonify change: three discrete mouth states
 * (closed / half / open) driven from outside instead of a continuous mouth slider, and emotion frame
 * sets that are crossfaded as whole groups, together with a matching posture.
 */
(function(root){
'use strict';
const clamp=(v,a,b)=>v<a?a:v>b?b:v;
function smooth(t){t=clamp(t,0,1);return t*t*(3-2*t);}
// From Anime2.5DRig runtime.js
function meshSize(w,h,cell){
  let nx=Math.max(2,Math.round(w/cell)),ny=Math.max(2,Math.round(h/cell));
  while((nx+1)*(ny+1)>65000){if(nx>=ny)nx=Math.max(2,Math.floor(nx*0.9));else ny=Math.max(2,Math.floor(ny*0.9));}
  return {nx,ny};
}
function spring(s,target,k,damping,dt){
  const count=Math.max(1,Math.ceil(dt/(1/120))),h=dt/count;
  for(let i=0;i<count;i++){s.v+=(-k*(s.x-target)-damping*s.v)*h;s.x+=s.v*h;}
}
// Default parameters of Anime2.5DRig (only the ones the idle loop uses).
const P={angleX:0,angleY:0,angleZ:0,eyeOpenL:1,eyeOpenR:1,eyeX:0,eyeY:0,brow:0,body:0,physAmp:3.4,soft:2.6,
  bust:0,bustY:1,irisScale:1,eyeEase:0.3,fhAmp:2.4,fhSoft:0.5,accAmp:1.4};
// Toonify: hair amplitudes raised over Anime2.5DRig's defaults (2 / 2 / 2 / 0.4); hair now mostly swings from
// the head's inertia, with only a light breeze on top.
const WIND=0.55;
const JAW=13;   // px (at face scale 1) the chin drops with a fully open mouth
const RANGES={angleX:[-1,1],angleY:[-1,1],angleZ:[-1,1],eyeOpenL:[0,1],eyeOpenR:[0,1],eyeX:[-1,1],eyeY:[-1,1],body:[-1,1]};
const FRONTISH={'front hair':1,'side hair':1,'ahoge':1};
// Posture and motion per emotion (scaled by the message's intensity): head pitch / tilt offsets, and multipliers
// for body sway, idle motion speed, speech nods and the gap between blinks.
const EMOTIONS={
  neutral:{},
  happy:{pitch:0.08,tilt:0.08,sway:1.3,speed:1.15,nod:1.3,blink:1},
  sad:{pitch:-0.30,tilt:-0.06,sway:0.55,speed:0.7,nod:0.6,blink:1.3},
  angry:{pitch:-0.14,tilt:0,sway:0.8,speed:1.2,nod:1.5,blink:1.4},
  surprised:{pitch:0.15,tilt:0.04,sway:0.9,speed:1.1,nod:1,blink:2.0},
};
const ALIASES={joy:'happy',joyful:'happy',glad:'happy',cheerful:'happy',excited:'happy',amused:'happy',
  unhappy:'sad',sorrow:'sad',sorrowful:'sad',upset:'sad',disappointed:'sad',mad:'angry',annoyed:'angry',
  furious:'angry',irritated:'angry',shocked:'surprised',astonished:'surprised',amazed:'surprised',
  calm:'neutral',none:'neutral',default:'neutral'};
const EMO_DEFAULTS={fade:0.3,maxWait:0.25};   // s crossfade; s an emotion change waits for a closed mouth at most

function create(gl,rig,opts={}){
  const EO=Object.assign({},EMO_DEFAULTS,opts);
  const renderer=root.RigRenderer.create(gl);
  const W=rig.canvas.w,H=rig.canvas.h,A=rig.anchors;
  const FS=A.faceScale,NP=A.neckPivot,BP=A.bodyPivot,FC={x:A.face.cx,y:A.face.cy};
  const fh=A.face.y1-A.face.y0;
  const CHEST={cx:NP.cx,cy:A.neckBottom+fh*0.60,rx:Math.max(1,(A.face.x1-A.face.x0)*0.60),ry:Math.max(1,fh*0.45)};
  const bounce={x:0,v:0,dy:0};
  const layers=[],sets={};
  let emoAt=-1;   // draw position of the emotion groups (all emotion layers sit together above the face parts)
  for(const Lr of rig.layers){
    const L=Object.assign({},Lr);delete L.img;
    const cell=(L.phys?30:42)*Math.max(0.6,W/768);
    const {nx,ny}=meshSize(L.w,L.h,cell),nv=(nx+1)*(ny+1);
    const base=new Float32Array(nv*2),uv=new Float32Array(nv*2);
    let k=0;
    for(let j=0;j<=ny;j++)for(let i=0;i<=nx;i++){base[k]=L.x+L.w*i/nx;base[k+1]=L.y+L.h*j/ny;uv[k]=i/nx;uv[k+1]=j/ny;k+=2;}
    const idx=new Uint16Array(nx*ny*6);let q=0;
    for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){const a=j*(nx+1)+i,b=a+1,c=a+nx+1,d=c+1;idx[q++]=a;idx[q++]=b;idx[q++]=c;idx[q++]=b;idx[q++]=d;idx[q++]=c;}
    L.base=base;L.cur=new Float32Array(base);
    L.bn=root.Rigger.baseName(L.name.replace(/_(l|r)$/,''));
    if(L.strands&&L.strands.length){
      const S=L.strands,nS=S.length;
      let spacing=120;
      if(nS>1){const ds=[];for(let s=1;s<nS;s++)ds.push(S[s].x-S[s-1].x);ds.sort((a,b)=>a-b);spacing=ds[ds.length>>1];}
      const sig=Math.max(1,spacing*0.6);
      L.sw=new Float32Array(nv*nS);L.su=new Float32Array(nv);
      L.spr=S.map((s,i)=>({stiff:{x:0,v:0,dx:0},soft:{x:0,v:0,dx:0},phase:i*1.37+L.z}));
      for(let v=0;v<nv;v++){
        const x=base[v*2],y=base[v*2+1];let tot=0;
        for(let s=0;s<nS;s++){const w=Math.exp(-Math.pow((x-S[s].x)/sig,2));L.sw[v*nS+s]=w;tot+=w;}
        let rY=0,tY=0;
        if(tot>1e-6){for(let s=0;s<nS;s++){L.sw[v*nS+s]/=tot;rY+=L.sw[v*nS+s]*S[s].rootY;tY+=L.sw[v*nS+s]*S[s].tipY;}}
        else{L.sw[v*nS]=1;rY=S[0].rootY;tY=S[0].tipY;}
        L.su[v]=clamp((y-rY)/Math.max(1,tY-rY),0,1);
      }
      // Toonify: short hair (strands shorter than ~half a face) barely sways; long hair gets full motion.
      const lens=S.map(s=>s.tipY-s.rootY).sort((a,b)=>a-b),faceH=Math.max(1,A.face.y1-A.face.y0);
      L.lenK=L.phys==='sway'?1:clamp((lens[lens.length>>1]/faceH-0.45)/0.9,0.08,1);
    }
    renderer.upload(L,{positions:L.cur,uvs:uv,indices:idx,image:new ImageData(Lr.img.data,Lr.img.width,Lr.img.height)});
    if(L.emo){(sets[L.emo]||(sets[L.emo]=[])).push(L);if(emoAt<0)emoAt=layers.length;}
    layers.push(L);
  }
  console.info('hair motion scale:',layers.filter(L=>L.lenK!=null).map(L=>L.name+'='+L.lenK.toFixed(2)).join(', '));

  // ---------- animation state ----------
  const cur=Object.assign({},P);
  // Toonify: idle head motion runs through critically damped springs (ease in / ease out, no snapping).
  // A glance target is held for a few seconds; the eyes reach it fast, the head follows slowly, the body last.
  const look={ax:0,ay:0,az:0,ex:0,ey:0},next={t:2500};
  const mk=()=>({x:0,v:0});
  const sp={ax:mk(),ay:mk(),az:mk(),body:mk(),ex:mk(),ey:mk()};
  function damped(s,target,omega,dt){   // critically damped spring with natural frequency omega (rad/s)
    const n=Math.max(1,Math.ceil(dt*240)),h=dt/n;
    for(let i=0;i<n;i++){s.v+=(omega*omega*(target-s.x)-2*omega*s.v)*h;s.x+=s.v*h;}
    return s.x;
  }
  let blinkT=-1,nextBlink=1800,irisBounceT=-1,blinkBounceStarted=false;
  const mouthW={close:1,half:0,open:0};
  const mouthO={x:0,v:0};   // continuous openness 0 (closed) .. 0.5 (half) .. 1 (open)

  function fadeAlpha(L,e){
    if(!L.fade)return 1;
    if(L.fade==='eyeOpen'){const v=L.side==='L'?e.eyeOpenL:e.eyeOpenR;return smooth((v-(0.10+e.eyeEase*0.45))/0.15);}
    if(L.fade==='eyeClose'){const v=L.side==='L'?e.eyeOpenL:e.eyeOpenR;return 1-smooth((v-(0.10+e.eyeEase*0.45))/0.15);}
    if(L.fade==='eyeClose2')return 0;
    if(L.fade==='mouthClose')return mouthW.close;
    if(L.fade==='mouthHalf')return mouthW.half;
    if(L.fade==='mouthOpen')return mouthW.open;
    return 1;
  }
  function deform(L,e){
    const b=L.base,o=L.cur,n=b.length;
    const isHead=L.group==='head';
    const az=e.angleZ*0.07,cz=Math.cos(az),sz=Math.sin(az);
    const ab=e.body*0.036,cb=Math.cos(ab),sb=Math.sin(ab);   // Toonify: a bit more lean than Anime2.5DRig (0.028)
    const bn=L.bn,eyeSide=L.side,EA=eyeSide==='L'?A.eyeL:(eyeSide==='R'?A.eyeR:null);
    const vOpen=eyeSide==='L'?e.eyeOpenL:e.eyeOpenR;
    const nS=L.strands?L.strands.length:0;
    const kind=L.phys==='sway'?'acc':FRONTISH[bn]?'fh':'bh';
    const physOn=nS&&L.su;
    const ampK=(kind==='acc'?e.accAmp*1.4:kind==='fh'?e.fhAmp:e.physAmp)*(L.lenK||1);
    const softK=kind==='acc'?0.8:kind==='fh'?e.fhSoft:e.soft;
    const powA=kind==='acc'?1.2:kind==='fh'?1.8:2.1;
    for(let k=0;k<n;k+=2){
      let x=b[k],y=b[k+1];
      const vi=k>>1;
      // --- eyes: lid compression, iris motion ---
      if(L.fade==='eyeOpen'&&EA){
        if(bn==='irides'){
          const isc=e.irisScale,ibx=e.irisBounceX,iby=e.irisBounceY;
          x=EA.icx+(x-EA.icx)*isc*ibx;y=EA.icy+(y-EA.icy)*isc*iby;
          x+=e.eyeX*11*FS;y+=e.eyeY*6*FS;
          const tl=smooth((0.32-vOpen)/0.32);
          y=EA.closeY+(y-EA.closeY)*(1-0.80*tl);
        }else{
          y=EA.closeY+(y-EA.closeY)*(1-0.85*(1-vOpen));
        }
      }
      if(L.fade==='eyeClose'&&EA)y-=vOpen*3;
      if(bn==='eyebrow')y+=(-e.brow*9+(1-vOpen)*3.5)*FS;
      // jaw drop (from Anime2.5DRig): the face below the mouth moves down with the mouth openness, most at the chin
      if((bn==='face'||L.part==='face')&&y>A.mouth.cy)y+=mouthO.x*JAW*FS*smooth((y-A.mouth.cy)/Math.max(1,A.face.y1-A.mouth.cy));
      // --- head transform ---
      let hw=isHead?1:(L.group==='body'?0.16:0);
      if(bn==='neck')hw=0.55*smooth((A.neckBottom-y)/Math.max(1,A.neckBottom-A.neckTop));
      if(hw>0){
        const rx=x-NP.cx,ry=y-NP.cy,rx2=rx*cz-ry*sz,ry2=rx*sz+ry*cz;
        x+=(rx2-rx)*hw;y+=(ry2-ry)*hw;
        const dd=L.depth;
        // Toonify: half of Anime2.5DRig's depth parallax, so back and front hair do not slide apart on turns
        x+=hw*FS*(e.angleX*(14+20*(dd-1))+e.angleX*(NP.cy-y)*0.028);
        y+=hw*FS*(-e.angleY*(9+15*(dd-1))-e.angleY*(dd-1)*(y-FC.y)*0.03);
      }
      // --- breathing ---
      y-=(L.group==='body'?e.breath*2.0:e.breathHead*1.6)*FS;
      if(bn==='topwear'&&y<CHEST.cy)y-=e.breath*2.2*FS*smooth((CHEST.cy-y)/(CHEST.ry*2));
      if(bn==='topwear')x=NP.cx+(x-NP.cx)*(1+e.breath*0.003);
      // --- strand physics ---
      if(physOn){
        const u=kind==='fh'?Math.min(1,L.su[vi]*1.6):L.su[vi];
        const amp=Math.pow(u,powA)*ampK,softMix=Math.min(1,Math.pow(u,1.2)*softK);
        let dx=0;
        for(let s=0;s<nS;s++){const w=L.sw[vi*nS+s];if(w<0.001)continue;const sp=L.spr[s];dx+=w*(sp.stiff.dx*(1-softMix)+sp.soft.dx*softMix);}
        x+=dx*amp;y+=Math.abs(dx)*amp*0.12;
      }
      o[k]=x;o[k+1]=y;
    }
    if(Math.abs(ab)>1e-4){
      for(let k=0;k<n;k+=2){const rx=o[k]-BP.cx,ry=o[k+1]-BP.cy;o[k]=BP.cx+rx*cb-ry*sb;o[k+1]=BP.cy+rx*sb+ry*cb;}
    }
  }
  function updateSprings(e,t,dt){
    const headDX=(e.angleX*14+e.angleZ*0.07*(NP.cy-FC.y))*FS;
    const bodyDX=e.body*0.028*(BP.cy-NP.cy)*0.35+e.angleX*14*0.16*FS;
    for(const L of layers){
      if(!L.spr)continue;
      const base=L.group==='head'?headDX:bodyDX,windScale=L.phys==='sway'&&L.group!=='head'?2.5:1;
      for(const sp of L.spr){
        const w=(1.8*Math.sin(t*0.6+sp.phase)+0.9*Math.sin(t*1.3+sp.phase*2.3))*windScale*WIND;
        const txv=base+w*FS;
        spring(sp.stiff,txv,70,9,dt);sp.stiff.dx=-(sp.stiff.x-txv)*2.2;
        spring(sp.soft,txv,16,1.3,dt);sp.soft.dx=-(sp.soft.x-txv)*3.0;
      }
    }
  }
  // Idle: glances with eased head follow-through, blinking, breathing (blink/breath from Anime2.5DRig).
  // Speech energy (0..1, from the audio analyser) drives extra motion while talking: a slow envelope sways the
  // body, and emphasis (fast minus slow envelope) gives small nods.
  const talk={fast:{x:0,v:0},slow:{x:0,v:0}};
  // Emotions. The face frames are a stack of groups ({name, a}), drawn bottom to top over the neutral face. A new
  // emotion is pushed on top with a = 0 and fades in; once it covers everything, the groups below are dropped.
  // Going to neutral (or to an emotion without frames) fades all groups out. Each group has its own closed-eye
  // and mouth frames, so blinking and lip-sync go on during a crossfade. A change waits up to maxWait for a
  // closed mouth, because two closed faces blend more cleanly than two different open mouths.
  const emo={stack:[],pending:null,wait:0,current:'neutral',intensity:1};
  const posture={pitch:mk(),tilt:mk(),sway:mk(),speed:mk(),nod:mk(),blink:mk()};
  for(const k of ['sway','speed','nod','blink'])posture[k].x=1;
  let idleT=0;
  function resolveEmotion(name){
    let n=String(name||'').trim().toLowerCase();
    n=ALIASES[n]||n;
    if(!EMOTIONS[n]&&!sets[n]){console.warn('toonify: unknown emotion',name);return null;}
    return n;
  }
  function setEmotion(name,intensity=1){
    const n=resolveEmotion(name);
    if(!n)return;
    emo.pending={name:n,intensity:clamp(+intensity||0,0,1)};emo.wait=0;
  }
  function updateEmotion(dt){
    if(emo.pending){
      emo.wait+=dt;
      if((mouthO.x<0.08&&blinkT<0)||emo.wait>=EO.maxWait){
        const {name,intensity}=emo.pending;emo.pending=null;
        emo.current=name;emo.intensity=intensity;
        const top=emo.stack[emo.stack.length-1];
        if(sets[name]&&!(top&&top.name===name))emo.stack.push({name,a:0});
      }
    }
    const step=dt/Math.max(0.01,EO.fade),top=emo.stack[emo.stack.length-1];
    if(top&&top.name===emo.current){
      top.a=Math.min(1,top.a+step);
      if(top.a>=1)emo.stack=[top];
    }else{
      for(const g of emo.stack)g.a-=step;
      emo.stack=emo.stack.filter(g=>g.a>0);
    }
    const P0=EMOTIONS[emo.current]||{},i=emo.intensity;
    const off=k=>(P0[k]||0)*i,mul=k=>1+((P0[k]==null?1:P0[k])-1)*i;
    damped(posture.pitch,off('pitch'),3,dt);damped(posture.tilt,off('tilt'),3,dt);
    for(const k of ['sway','speed','nod','blink'])damped(posture[k],mul(k),3,dt);
  }
  function animate(nowMs,dt,level){
    const tgt=Object.assign({},P);
    idleT+=dt*Math.max(0.2,posture.speed.x);
    const t=idleT;
    const tf=damped(talk.fast,level,9,dt),ts=damped(talk.slow,level,1.6,dt),stress=Math.max(0,tf-ts)*posture.nod.x;
    if(nowMs>next.t){   // new glance: mostly small, sometimes back to centre
      next.t=nowMs+2500+Math.random()*4000;
      const home=Math.random()<0.35,r=()=>Math.random()*2-1;
      look.ax=home?0:r()*0.45;look.ay=home?0:r()*0.25;look.az=home?0:r()*0.2;
      look.ex=home?0:clamp(look.ax*0.9+r()*0.25,-1,1);look.ey=home?0:clamp(look.ay*0.8+r()*0.2,-1,1);
    }
    // slow breathing-like drift on top of the held glance
    const ax=look.ax+0.10*Math.sin(t*0.35)+0.04*Math.sin(t*0.83+1.0);
    const ay=look.ay+0.06*Math.sin(t*0.27+1.7)-0.35*stress+posture.pitch.x;
    const az=look.az+0.06*Math.sin(t*0.21+0.5)+posture.tilt.x;
    tgt.eyeX=damped(sp.ex,look.ex,14,dt);tgt.eyeY=damped(sp.ey,look.ey,14,dt);
    tgt.angleX=damped(sp.ax,ax,2.6,dt);tgt.angleY=damped(sp.ay,ay,2.4,dt);tgt.angleZ=damped(sp.az,az,2.0,dt);
    const sw=posture.sway.x;
    tgt.body=damped(sp.body,0.6*sp.ax.x+sw*(0.38*Math.sin(t*0.19+2.1)+0.14*Math.sin(t*0.47+0.4)   // slow weight shift
      +0.35*ts*Math.sin(t*1.1+0.3)),1.6,dt);                                                     // livelier while talking
    // No new blink while an emotion change waits for a calm moment.
    if(blinkT<0&&nowMs>nextBlink&&!emo.pending){
      blinkT=0;blinkBounceStarted=false;nextBlink=nowMs+(1600+Math.random()*3800)*posture.blink.x;
      if(Math.random()<0.18)nextBlink=nowMs+280;
    }
    if(blinkT>=0){blinkT+=dt;
      const d=blinkT,hold=0.34;let v;
      if(d<0.08)v=1-d/0.08;else if(d<0.08+hold)v=0;else if(d<0.24+hold){
        v=(d-0.08-hold)/0.16;
        if(!blinkBounceStarted&&v>0.12){blinkBounceStarted=true;irisBounceT=0;}
      }else{v=1;blinkT=-1;}
      tgt.eyeOpenL=Math.min(tgt.eyeOpenL,v);tgt.eyeOpenR=Math.min(tgt.eyeOpenR,v);
    }
    if(irisBounceT>=0){irisBounceT+=dt;if(irisBounceT>0.52)irisBounceT=-1;}
    // Motion is already smoothed by the springs; blinks follow their own curve.
    for(const k in cur){if(RANGES[k])tgt[k]=clamp(tgt[k],...RANGES[k]);cur[k]=tgt[k];}
    const e=Object.assign({},cur);
    e.irisBounceX=1;e.irisBounceY=1;
    if(irisBounceT>=0){
      const p=irisBounceT/0.52,damp=Math.exp(-2.2*p);
      const scale=1+0.06*Math.sin(p*Math.PI*4.0)*damp,squash=0.03*Math.sin(p*Math.PI*4.0+Math.PI/2)*damp;
      e.irisBounceX=scale*(1+squash);e.irisBounceY=scale*(1-squash);
    }
    e.breath=0.5+0.5*Math.sin(t*2*Math.PI/3.4);
    e.breathHead=0.5+0.5*Math.sin(t*2*Math.PI/3.4-0.6);
    updateSprings(e,t,dt);
    return e;
  }
  // mouthState: 0 closed, 1 half, 2 open (from the audio thresholds). The picture follows a spring-smoothed
  // openness and blends only neighbouring frames (closed<->half<->open), so the mouth glides instead of popping.
  function setMouth(state,dt){
    const o=damped(mouthO,state*0.5,38,dt);
    const c=clamp(o,0,1);
    mouthW.close=clamp(1-2*c,0,1);
    mouthW.half=clamp(1-Math.abs(2*c-1),0,1);
    mouthW.open=clamp(2*c-1,0,1);
  }
  function render(e){
    const items=[],masks=[];
    for(let i=0;i<layers.length;i++){
      const L=layers[i];
      if(i===emoAt){
        for(const g of emo.stack){
          const group=[];
          for(const S of sets[g.name]){
            const alpha=fadeAlpha(S,e);
            if(alpha<0.004)continue;
            deform(S,e);renderer.positions(S,S.cur);group.push({L:S,alpha});
          }
          items.push({group,alpha:smooth(g.a)});
        }
      }
      if(L.emo)continue;
      const isMask=L.bn==='eyewhite'&&!!L.side;
      const alpha=fadeAlpha(L,e)*(L.opacity==null?1:L.opacity);
      if(alpha<0.004&&!isMask)continue;
      deform(L,e);renderer.positions(L,L.cur);
      if(isMask)masks.push({L,side:L.side});
      if(alpha>=0.004)items.push({L,alpha,clip:L.bn==='irides'&&L.side?L.side:null});
    }
    renderer.draw({width:W,height:H,background:[0,0,0,0],masks,items});
  }
  let simMs=0;
  return {
    width:W,height:H,
    emotions:Object.keys(sets),
    setEmotion,
    get emotion(){return emo.pending?emo.pending.name+'…':emo.current;},
    frame(dt,mouthState,level=0){
      simMs+=dt*1000;updateEmotion(dt);const e=animate(simMs,dt,level);setMouth(mouthState,dt);render(e);
    },
  };
}
root.ToonRig={create};
})(typeof self!=='undefined'?self:this);
