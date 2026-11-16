/* Darkroom · Instant Frame — polaroid template tool. Everything runs in the browser. */
(function(){
"use strict";
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const clone = o => JSON.parse(JSON.stringify(o));
const pad2 = n => String(n).padStart(2,"0");
const isoOf = d => `${d.getFullYear()}-${pad2(d.getMonth()+1)}-${pad2(d.getDate())}`;
const todayISO = isoOf(new Date());
const r4 = v => Math.round(v*1e4)/1e4;

/* ================= design tokens ================= */
const PAPERS = {
  ivory:  {name:"象牙", bg:"#FAF9F5", ink:"#141413", meta:"#87867F", accent:"#D97757", edge:"rgba(20,20,19,.10)", well:"rgba(20,20,19,.14)"},
  oat:    {name:"燕麦", bg:"#F0EEE6", ink:"#141413", meta:"#83827B", accent:"#D97757", edge:"rgba(20,20,19,.10)", well:"rgba(20,20,19,.14)"},
  manilla:{name:"麻纸", bg:"#EBDBBC", ink:"#141413", meta:"#7D6B52", accent:"#C15F3C", edge:"rgba(20,20,19,.10)", well:"rgba(20,20,19,.16)"},
  clay:   {name:"陶土", bg:"#D97757", ink:"#FAF9F5", meta:"rgba(250,249,245,.8)", accent:"#141413", edge:"rgba(20,20,19,.12)", well:"rgba(20,20,19,.20)"},
  slate:  {name:"石墨", bg:"#1F1E1D", ink:"#FAF9F5", meta:"#A8A69D", accent:"#D97757", edge:"rgba(255,255,255,.06)", well:"rgba(0,0,0,.45)"}
};
const BGS = {
  oat:{name:"燕麦", c:"#F0EEE6"}, cloud:{name:"雾灰", c:"#E3DACC"}, clay:{name:"陶土", c:"#D97757"},
  kraft:{name:"牛皮", c:"#D4A27F"}, olive:{name:"橄榄", c:"#788C5D"}, sky:{name:"雾蓝", c:"#A9C3DC"}, slate:{name:"石墨", c:"#1F1E1D"}
};
const ILLO_COLORS = {C:"#D97757", K:"#D4A27F", O:"#788C5D", S:"#6A9BCC", M:"#EBDBBC", W:"#FAF9F5", H:"#CBCADB", I:"#141413"};
const THEMES = {
  orig: {name:"原色"},
  clay: {name:"陶土", map:{C:"#D97757",K:"#E9A88D",O:"#B8553A",S:"#F2C9B8",M:"#F7E2D7",H:"#ECC0AE",W:"#FDF6F1"}},
  kraft:{name:"牛皮", map:{C:"#B07A52",K:"#D4A27F",O:"#8E6142",S:"#E4C6AA",M:"#EFDFC6",H:"#E2CFB8",W:"#FAF3E8"}},
  olive:{name:"橄榄", map:{C:"#788C5D",K:"#A3B285",O:"#55663F",S:"#C6D0B0",M:"#E4E9D8",H:"#D2DABF",W:"#F6F8F0"}},
  sky:  {name:"雾蓝", map:{C:"#4F7FB0",K:"#8DB2D8",O:"#335A82",S:"#B9D0E8",M:"#E0EAF4",H:"#C9D9EC",W:"#F3F7FB"}},
  mono: {name:"灰阶", map:{C:"#5E5D59",K:"#9A988F",O:"#3D3D3A",S:"#C2C0B6",M:"#E8E6DC",H:"#D6D4CA",W:"#FAF9F5"}},
  line: {name:"线稿", line:true}
};
const DECO_COLORS = {clay:{name:"陶土",c:"#D97757"}, kraft:{name:"牛皮",c:"#D4A27F"}, olive:{name:"橄榄",c:"#788C5D"}, sky:{name:"雾蓝",c:"#6A9BCC"}, heather:{name:"薰衣草",c:"#CBCADB"}, masking:{name:"美纹纸",c:"#EDE6D6"}};
const TEXT_COLORS = {auto:{name:"自动"}, clay:{name:"陶土",c:"#C6613F"}, olive:{name:"橄榄",c:"#5E7048"}, sky:{name:"雾蓝",c:"#4F7FB0"}, kraft:{name:"牛皮",c:"#9C6B47"}};
const BRUSH_COLORS = {ink:{name:"墨黑",c:"#141413"}, clay:{name:"陶土",c:"#D97757"}, olive:{name:"橄榄",c:"#788C5D"}, sky:{name:"雾蓝",c:"#6A9BCC"}, kraft:{name:"牛皮",c:"#D4A27F"}, ivory:{name:"象牙白",c:"#FAF9F5"}};
const STYLES = {
  classic:{side:.058, top:.062, bottom:.27, caption:true},
  wide:   {side:.058, top:.062, bottom:.40, caption:true},
  even:   {side:.075, top:.075, bottom:.075, caption:false}
};
const RATIOS = [["orig","原图"],["1:1"],["4:5"],["3:4"],["2:3"],["9:16"],["5:4"],["4:3"],["3:2"],["16:9"]];
const FONTS = {
  italic:{k:1.08, css:s=>`italic 400 ${s}px "Newsreader","Noto Serif SC",Georgia,serif`},
  serif: {k:1.0,  css:s=>`400 ${s}px "Newsreader","Noto Serif SC",Georgia,serif`},
  sans:  {k:.92,  css:s=>`500 ${s}px "DM Sans","Noto Sans SC",system-ui,sans-serif`},
  mono:  {k:.78,  css:s=>`400 ${s}px "JetBrains Mono","Noto Sans SC",ui-monospace,monospace`},
  hand:  {k:1.32, css:s=>`500 ${s}px "Caveat","Ma Shan Zheng",cursive`},
  casual:{k:1.05, css:s=>`400 ${s}px "Kalam","Zhi Mang Xing",cursive`},
  brush: {k:.78,  css:s=>`400 ${s}px "Homemade Apple","Liu Jian Mao Cao",cursive`},
  cute:  {k:1.12, css:s=>`400 ${s}px "Patrick Hand","ZCOOL KuaiLe",cursive`}
};
const META_FONT = s => `500 ${s}px "JetBrains Mono","Noto Sans SC",ui-monospace,monospace`;
const STAMP_FONT = s => `${s}px "Share Tech Mono","JetBrains Mono",ui-monospace,monospace`;
const EXPORT_K = {std:1500, hd:2400, max:3200};
const COLLAGE_LONG = {std:2000, hd:3000, max:4000};
const MAX_PIXELS = 15e6;
const MONTHS = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"];
const MAX_ILLOS = 8, MAX_PRESET = 4, PAD = .1;
const PRINT_SIZES = {mini:{n:"Mini", w:54, hint:"3:4"}, square:{n:"Square", w:72, hint:"1:1"}, classic:{n:"经典 600", w:88, hint:"1:1"}, wide:{n:"Wide", w:108, hint:"3:2"}};
const ILLO_BY_ID = Object.fromEntries(ILLOS.map(i => [i.id, i]));
const customs = {};   // uploaded stickers: id -> {id,n,img,w,h,url,cat:"mine"}
const getIllo = id => ILLO_BY_ID[id] || customs[id];

const DEFAULTS = {
  ratio:"1:1", style:"classic", paper:"ivory", border:1, grain:true,
  illos:["sprig","sun"], illoMode:"right", illoSize:1, illoPer:false, illoSeed:7, lastPreset:"right", illoTheme:"orig",
  free:{inited:false, illos:[], cap:null, meta:null, sticker:false},
  deco:"none", decoColor:"clay", decoSeed:3,
  caption:"午后的光，刚刚好", perCaption:false, font:"italic", align:"left", textSize:1, textColor:"auto",
  dateSource:"exif", date:todayISO, dateFmt:"dot", note:"No. 001", accent:true, autoNum:true,
  filter:"none", stamp:false, leak:false, vignette:false, fadeEdge:false,
  brush:{color:"ink", width:1},
  quality:"hd", format:"png", withBg:false, bg:"clay", tilt:0,
  collage:{layout:"scatter", canvas:"3:4", bg:"oat", seed:1},
  print:{size:"square", fill:false, marks:true}
};
const BUILTIN_TPLS = [
  {name:"晨光", c:"#FAF9F5", data:{}},
  {name:"复古傻瓜机", c:"#F0EEE6", data:{paper:"oat", illos:[], font:"mono", filter:"warm", stamp:true, leak:true, deco:"tape", decoColor:"masking"}},
  {name:"陶土贴纸", c:"#D97757", data:{paper:"clay", illos:["heart","sparkle","flame"], illoMode:"corner", font:"hand", align:"center", deco:"pin", decoColor:"olive"}},
  {name:"夜读", c:"#1F1E1D", data:{paper:"slate", style:"wide", illos:["moon","book","stars"], font:"serif", filter:"fade", vignette:true}},
  {name:"手账", c:"#EBDBBC", data:{paper:"manilla", illos:["plant","cup"], illoMode:"sides", font:"casual", deco:"clip", illoTheme:"kraft"}},
  {name:"晾衣绳合集", c:"#A9C3DC", data:{paper:"ivory", illos:["sun"], illoTheme:"sky", font:"cute", collage:{layout:"line", canvas:"3:4", bg:"sky", seed:2}}}
];

const state = clone(DEFAULTS);
const photos = [];          // {id,url,name,w,h,thumb,zoom,px,py,caption,date,seed,free,doodles}
const scratch = {doodles:[]};
let cur = -1, sel = null, tool = "move", illoCat = "all";
const curPhoto = () => photos[cur] || null;
const docOf = ph => ph || scratch;
const isFree = () => state.illoMode === "free";
const freeOf = ph => (ph && ph.free) || state.free;
const getPath = p => p.split(".").reduce((o,k) => o == null ? o : o[k], state);
function setPath(p, v){ const ks = p.split("."); let o = state; for(let i=0;i<ks.length-1;i++) o = o[ks[i]]; o[ks[ks.length-1]] = v; }

/* ================= helpers ================= */
function rng(seed){ let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function rr(ctx,x,y,w,h,r){
  ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.lineTo(x+w-r,y); ctx.quadraticCurveTo(x+w,y,x+w,y+r);
  ctx.lineTo(x+w,y+h-r); ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);
  ctx.lineTo(x+r,y+h); ctx.quadraticCurveTo(x,y+h,x,y+h-r);
  ctx.lineTo(x,y+r); ctx.quadraticCurveTo(x,y,x+r,y); ctx.closePath();
}
function hexA(hex, a){ const n = parseInt(hex.slice(1),16); return `rgba(${n>>16&255},${n>>8&255},${n&255},${a})`; }
const noiseTile = (() => {
  const n = 180, c = document.createElement("canvas"); c.width = c.height = n;
  const x = c.getContext("2d"), d = x.createImageData(n,n);
  for(let i=0;i<d.data.length;i+=4){ const v = Math.random() < .5 ? 0 : 255; d.data[i]=d.data[i+1]=d.data[i+2]=v; d.data[i+3]=Math.random()*26; }
  x.putImageData(d,0,0); return c;
})();
const HAS_FILTER = "filter" in CanvasRenderingContext2D.prototype;
const HAS_LS = "letterSpacing" in CanvasRenderingContext2D.prototype;
const setLS = (ctx,v) => { if(HAS_LS) ctx.letterSpacing = v; };
const measureCtx = document.createElement("canvas").getContext("2d");
const easeIO = t => t < .5 ? 2*t*t : 1 - Math.pow(-2*t+2, 2)/2;

/* ================= per-photo text ================= */
function dateFor(ph){ return (state.dateSource === "exif" && ph && ph.date) ? ph.date : state.date; }
function dateText(ph){
  const iso = dateFor(ph);
  if(state.dateFmt === "off" || !iso) return "";
  const [y,m,d] = iso.split("-").map(Number);
  if(!y) return "";
  if(state.dateFmt === "en") return `${MONTHS[m-1]} ${d} ${y}`;
  if(state.dateFmt === "cn") return `${y}年${m}月${d}日`;
  return `${y}.${pad2(m)}.${pad2(d)}`;
}
function noteText(idx){
  const s = state.note.trim();
  if(!state.autoNum || idx <= 0) return s;
  const m = s.match(/(\d+)(?!.*\d)/);
  if(!m) return s;
  return s.slice(0,m.index) + String(parseInt(m[1],10) + idx).padStart(m[1].length,"0") + s.slice(m.index + m[1].length);
}
function captionFor(ph){ return (state.perCaption && ph && ph.caption != null) ? ph.caption : state.caption; }
const capLines = ph => captionFor(ph).split(/\n/).map(s => s.trim()).filter(Boolean).slice(0,3);
function textInk(){ const t = state.textColor; if(t === "auto") return PAPERS[state.paper].ink; return TEXT_COLORS[t] ? TEXT_COLORS[t].c : t; }

/* ================= geometry ================= */
function ratioValue(ph){
  if(state.ratio === "orig") return ph ? Math.min(2.5, Math.max(.4, ph.w/ph.h)) : 1;
  const [a,b] = state.ratio.split(":").map(Number); return a/b;
}
function geom(K, ph){
  const r = ratioValue(ph), st = STYLES[state.style], b = state.border;
  const iw = K*Math.sqrt(r), ih = K/Math.sqrt(r);
  const side = st.side*K*b, top = st.top*K*b;
  const bottom = st.caption ? st.bottom*K*(.55+.45*b) : st.bottom*K*b;
  return {K, st, iw, ih, side, top, bottom, x0:side, y0:top, W:iw+2*side, H:top+ih+bottom};
}
function bgBox(g){
  const a = state.tilt*Math.PI/180, c = Math.abs(Math.cos(a)), s = Math.abs(Math.sin(a)), pad = .17*g.K;
  return {pad, a, w: g.W*c + g.H*s + 2*pad, h: g.W*s + g.H*c + 2*pad};
}
function photoRect(g, ph){
  const s = Math.max(g.iw/ph.w, g.ih/ph.h) * ph.zoom;
  const dw = ph.w*s, dh = ph.h*s, ox = (dw-g.iw)/2, oy = (dh-g.ih)/2;
  return {x: g.x0 - ox + ph.px*ox, y: g.y0 - oy + ph.py*oy, w: dw, h: dh, ox, oy};
}

/* ================= illustrations ================= */
const pathCache = new Map();
function partsOf(id){
  if(pathCache.has(id)) return pathCache.get(id);
  const def = ILLO_BY_ID[id];
  const parts = def.p.map(p => ({path:new Path2D(p.d), f:p.f, s:p.s !== false, w:p.w || 1}));
  const all = new Path2D(); parts.forEach(p => all.addPath(p.path));
  const v = {parts, all}; pathCache.set(id, v); return v;
}
function illoPalette(paper, theme){
  const th = THEMES[theme] || THEMES.orig;
  const c = Object.assign({}, ILLO_COLORS, th.map || {});
  let ink = "#141413";
  if(paper === "clay" && (!th.map || theme === "clay")) c.C = "#F0EEE6";
  if(paper === "slate"){ ink = "#F0EEE6"; c.I = "#F0EEE6"; }
  return {c, ink, line: !!th.line};
}
function drawIllo(ctx, id, cx, cy, size, rot, paper, sticker, theme){
  const cu = customs[id];
  if(cu){
    const k = size/Math.max(cu.w, cu.h), dw = cu.w*k, dh = cu.h*k;
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(rot);
    if(sticker){ ctx.shadowColor = "rgba(20,20,19,.3)"; ctx.shadowBlur = size*.05; ctx.shadowOffsetY = size*.02; }
    ctx.drawImage(cu.img, -dw/2, -dh/2, dw, dh);
    ctx.restore(); return;
  }
  if(!ILLO_BY_ID[id]) return;
  const {parts, all} = partsOf(id), pal = illoPalette(sticker ? "ivory" : paper, theme), s = size/100;
  ctx.save();
  ctx.translate(cx, cy); ctx.rotate(rot); ctx.scale(s, s); ctx.translate(-50, -50);
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  if(sticker){
    ctx.save();
    ctx.shadowColor = "rgba(20,20,19,.28)"; ctx.shadowBlur = size*.06; ctx.shadowOffsetY = size*.025;
    ctx.strokeStyle = "#FFFFFF"; ctx.lineWidth = 17; ctx.stroke(all);
    ctx.restore();
    ctx.fillStyle = "#FFFFFF"; ctx.fill(all); ctx.strokeStyle = "#FFFFFF"; ctx.lineWidth = 17; ctx.stroke(all);
  }
  for(const p of parts){
    if(p.f && !(pal.line && p.f !== "I")){
      ctx.fillStyle = pal.c[p.f];
      if(p.s){ ctx.save(); ctx.translate(2.4, 2); ctx.fill(p.path); ctx.restore(); }
      else ctx.fill(p.path);
    }
    if(p.s){ ctx.lineWidth = 3.2*p.w; ctx.strokeStyle = pal.ink; ctx.stroke(p.path); }
  }
  ctx.restore();
}
function illoSVG(id){
  const def = ILLO_BY_ID[id];
  let s = `<svg viewBox="-4 -4 108 108" xmlns="http://www.w3.org/2000/svg" stroke-linecap="round" stroke-linejoin="round">`;
  for(const p of def.p){
    const stroke = p.s !== false;
    if(p.f) s += `<path d="${p.d}" fill="${ILLO_COLORS[p.f]}" ${stroke ? 'transform="translate(2.4 2)"' : ""}/>`;
    if(stroke) s += `<path d="${p.d}" fill="none" stroke="#141413" stroke-width="${3.2*(p.w||1)}"/>`;
  }
  return s + "</svg>";
}
function illosFor(ph, idx){
  const lim = isFree() ? MAX_ILLOS : MAX_PRESET;
  if(state.illoPer && photos.length){
    const n = Math.min(state.illos.length || 2, lim);
    const r = rng(((ph ? ph.seed : 1) * 2654435761) ^ state.illoSeed);
    const pool = ILLOS.map(i => i.id), out = [];
    while(out.length < n){ out.push(pool.splice(Math.floor(r()*pool.length),1)[0]); }
    return out;
  }
  return state.illos.filter(getIllo).slice(0, lim);
}
function presetMode(g, mode){ return mode === "free" ? "free" : (g.st.caption ? mode : "corner"); }
function illoLayout(g, ph, idx, forceMode){
  let ids = illosFor(ph, idx); if(forceMode) ids = ids.slice(0, MAX_PRESET);
  const mode = presetMode(g, forceMode || state.illoMode);
  const out = {items:[], reserveL:0, reserveR:0, sticker:false, mode};
  if(!ids.length || mode === "free") return out;
  const r = rng((state.illoSeed * 7919) ^ ((ph ? ph.seed : 3) * 31));
  const bandTop = g.y0 + g.ih, band = g.H - bandTop, left = g.side, right = g.W - g.side;
  const jit = () => (r()*2-1) * .2;
  if(mode === "corner"){
    out.sticker = true;
    const size = Math.min((g.st.caption ? band*.6 : g.K*.18) * state.illoSize, Math.min(g.iw,g.ih)*.5);
    const a = [[g.x0+g.iw - size*.08, g.y0+g.ih - size*.02],[g.x0 + size*.08, g.y0 + size*.04],[g.x0+g.iw - size*.06, g.y0 + size*.06],[g.x0 + size*.1, g.y0+g.ih - size*.04]];
    ids.forEach((id,i) => out.items.push({id, slot:i, cx:a[i][0], cy:a[i][1], size: i ? size*.82 : size, rot: jit()*1.4}));
    if(g.st.caption) out.reserveR = size*.55;
    return out;
  }
  const size = Math.min(band*.58*state.illoSize, band*.92), step = size*.8, gap = g.K*.025, cy0 = bandTop + band*.5;
  const place = (list, fromRight) => {
    list.forEach(([id,slot], i) => {
      const off = size/2 + i*step;
      out.items.push({id, slot, cx: fromRight ? right - off : left + off, cy: cy0 + (r()-.5)*band*.08, size, rot: jit()});
    });
    return list.length ? size + (list.length-1)*step + gap : 0;
  };
  const ent = ids.map((id,i) => [id,i]);
  if(mode === "right") out.reserveR = place(ent.slice().reverse(), true);
  else {
    out.reserveL = place(ent.filter((_,i) => i%2 === 0).reverse(), false);
    out.reserveR = place(ent.filter((_,i) => i%2 === 1).reverse(), true);
  }
  return out;
}

/* ================= text layout ================= */
function metaInfo(ph, idx, K, scale){
  const ms = .021*K*(scale || 1), dt = dateText(ph), note = noteText(idx);
  const str = [dt, note].filter(Boolean).join("  ·  ");
  measureCtx.font = META_FONT(ms); setLS(measureCtx, (ms*.08)+"px");
  const tw = str ? measureCtx.measureText(str).width : 0; setLS(measureCtx, "0px");
  const dotR = ms*.3, dotW = state.accent ? dotR*2 + (str ? ms*.75 : 0) : 0;
  return {ms, dt, note, str, tw, dotR, dotW, w: dotW + tw, show: !!(str || state.accent)};
}
function capInfo(ph, K, maxW, availH, scale){
  const lines = capLines(ph); if(!lines.length) return null;
  const f = FONTS[state.font]; let size = .056*K*state.textSize*f.k*(scale || 1);
  measureCtx.font = f.css(size);
  let mw = Math.max(...lines.map(l => measureCtx.measureText(l).width));
  if(maxW && mw > maxW){ size *= maxW/mw; mw = maxW; }
  if(availH && lines.length*size*1.22 > availH){ const k = availH/(lines.length*size*1.22); size *= k; mw *= k; }
  return {lines, size, w: mw, lh: size*1.22, h: lines.length*size*1.22, f};
}
function presetText(g, ph, idx, lay){
  const K = g.K, bandTop = g.y0 + g.ih, band = g.H - bandTop;
  const left = g.side + lay.reserveL, right = g.W - g.side - lay.reserveR;
  const inBand = lay.items.length > 0 && (lay.mode === "right" || lay.mode === "sides");
  const center = state.align === "center" || (lay.items.length > 0 && lay.mode === "sides");
  const cxm = (left + right)/2, maxW = Math.max(10, right - left);
  const m = metaInfo(ph, idx, K);
  const c = capInfo(ph, K, maxW, band*(m.show ? .58 : .8));
  let cap = null, meta = null;
  if(c){
    const cy = m.show ? bandTop + band*.42 : bandTop + band*.5;
    cap = Object.assign(c, {cy, ax: center ? cxm : left, align: center ? "center" : "left", cx: center ? cxm : left + c.w/2});
  }
  if(m.show){
    const y = c ? g.H - band*.2 : bandTop + band*.5 + m.ms*.36;
    const combined = center || inBand;
    const x0 = combined ? (center ? cxm - m.w/2 : left) : g.side;
    meta = Object.assign(m, {y, x0, split: combined ? null : g.W - g.side, cx: x0 + m.w/2, cy: y - m.ms*.36});
  }
  return {cap, meta};
}
function drawMetaLine(ctx, m, x0, y, splitRight){
  const pal = PAPERS[state.paper];
  ctx.font = META_FONT(m.ms); setLS(ctx, (m.ms*.08)+"px"); ctx.textBaseline = "alphabetic";
  let x = x0;
  if(state.accent){ ctx.fillStyle = pal.accent; ctx.beginPath(); ctx.arc(x + m.dotR, y - m.ms*.34, m.dotR, 0, Math.PI*2); ctx.fill(); x += m.dotW; }
  ctx.fillStyle = pal.meta; ctx.textAlign = "left";
  if(splitRight != null){
    if(m.dt) ctx.fillText(m.dt, x, y);
    if(m.note){ ctx.textAlign = "right"; ctx.fillText(m.note, splitRight, y); }
  } else if(m.str) ctx.fillText(m.str, x, y);
  setLS(ctx, "0px");
}
function drawCapLines(ctx, c, x, cy, align){
  ctx.font = c.f.css(c.size); ctx.fillStyle = textInk(); ctx.textAlign = align; ctx.textBaseline = "middle";
  c.lines.forEach((l,i) => ctx.fillText(l, x, cy + (i - (c.lines.length-1)/2)*c.lh));
}
function drawText(ctx, g, ph, idx, lay){
  if(!g.st.caption || lay.mode === "free") return;
  const T = presetText(g, ph, idx, lay);
  if(T.cap) drawCapLines(ctx, T.cap, T.cap.ax, T.cap.cy, T.cap.align);
  if(T.meta) drawMetaLine(ctx, T.meta, T.meta.x0, T.meta.y, T.meta.split);
}

/* ================= free layout ================= */
function defaultFreeSlot(i, g){
  const band = g.st.caption ? (g.y0 + g.ih + (g.H - g.y0 - g.ih)*.5)/g.H : (g.y0 + g.ih*.82)/g.H;
  const row = Math.floor(i/4);
  return {cx: Math.min(.92, .6 + (i%4)*.1), cy: Math.max(.08, band - row*.12), scale: .9, rot: (((i*37)%11)-5)*.035};
}
function freeElements(g, ph, idx){
  const F = freeOf(ph), els = [];
  if(!F.cap) F.cap = {cx:.32, cy:.9, scale:1, rot:0};
  if(!F.meta) F.meta = {cx:.25, cy:.96, scale:1, rot:0};
  const c = capInfo(ph, g.K, g.W*1.3, null, F.cap.scale);
  if(c) els.push({type:"cap", cx:F.cap.cx*g.W, cy:F.cap.cy*g.H, w:c.w, h:c.h, rot:F.cap.rot, c});
  const m = metaInfo(ph, idx, g.K, F.meta.scale);
  if(m.show) els.push({type:"meta", cx:F.meta.cx*g.W, cy:F.meta.cy*g.H, w:m.w, h:m.ms*1.5, rot:F.meta.rot, m});
  illosFor(ph, idx).forEach((id, i) => {
    if(!F.illos[i]) F.illos[i] = defaultFreeSlot(i, g);
    const sl = F.illos[i], size = g.K*.16*sl.scale;
    els.push({type:"illo", slot:i, id, cx:sl.cx*g.W, cy:sl.cy*g.H, w:size*.9, h:size*.9, rot:sl.rot, size, theme: sl.theme || state.illoTheme});
  });
  return els;
}
function drawFreeText(ctx, els){
  for(const e of els){
    if(e.type === "illo") continue;
    ctx.save(); ctx.translate(e.cx, e.cy); ctx.rotate(e.rot);
    if(e.type === "cap") drawCapLines(ctx, e.c, 0, 0, "center");
    else drawMetaLine(ctx, e.m, -e.m.w/2, e.m.ms*.36, null);
    ctx.restore();
  }
}
function initFree(fromMode, F){
  F = F || freeOf(curPhoto());
  const ph = curPhoto(), idx = Math.max(0, cur), K = 1000, g = geom(K, ph);
  const mode = (!fromMode || fromMode === "free") ? "right" : fromMode;
  const lay = illoLayout(g, ph, idx, mode), ids = illosFor(ph, idx);
  const oldThemes = (F.illos || []).map(s => s && s.theme);
  F.illos = [];
  lay.items.forEach(it => { F.illos[it.slot] = {cx: r4(it.cx/g.W), cy: r4(it.cy/g.H), scale: r4(it.size/(K*.16)), rot: r4(it.rot)}; });
  ids.forEach((_, i) => { if(!F.illos[i]) F.illos[i] = defaultFreeSlot(i, g); if(oldThemes[i]) F.illos[i].theme = oldThemes[i]; });
  if(g.st.caption){
    const T = presetText(g, ph, idx, lay);
    F.cap = T.cap ? {cx: r4(T.cap.cx/g.W), cy: r4(T.cap.cy/g.H), scale: r4(T.cap.size/(.056*K*state.textSize*T.cap.f.k)), rot:0} : {cx:.32, cy:(g.y0+g.ih+(g.H-g.y0-g.ih)*.42)/g.H, scale:1, rot:0};
    F.meta = T.meta ? {cx: r4(T.meta.cx/g.W), cy: r4(T.meta.cy/g.H), scale:1, rot:0} : {cx:.25, cy:.95, scale:1, rot:0};
  } else {
    F.cap = {cx:.5, cy: r4((g.H - g.bottom/2)/g.H), scale:.55, rot:0};
    F.meta = {cx:.5, cy: r4((g.top/2)/g.H), scale:.85, rot:0};
  }
  F.sticker = presetMode(g, mode) === "corner";
  F.inited = true;
}

/* ================= doodles ================= */
function drawDoodles(ctx, g, doc){
  if(!doc.doodles.length) return;
  ctx.save(); ctx.lineCap = "round"; ctx.lineJoin = "round";
  for(const st of doc.doodles){
    const p = st.pts, n = p.length/2; if(!n) continue;
    ctx.strokeStyle = ctx.fillStyle = (BRUSH_COLORS[st.c] || BRUSH_COLORS.ink).c;
    ctx.lineWidth = Math.max(.6, st.w*g.K*.008);
    const X = i => p[i*2]*g.W, Y = i => p[i*2+1]*g.H;
    if(n === 1){ ctx.beginPath(); ctx.arc(X(0), Y(0), ctx.lineWidth/2, 0, Math.PI*2); ctx.fill(); continue; }
    ctx.beginPath(); ctx.moveTo(X(0), Y(0));
    for(let i=1;i<n-1;i++) ctx.quadraticCurveTo(X(i), Y(i), (X(i)+X(i+1))/2, (Y(i)+Y(i+1))/2);
    ctx.lineTo(X(n-1), Y(n-1)); ctx.stroke();
  }
  ctx.restore();
}

/* ================= decorations ================= */
function drawTape(ctx, cx, cy, w, h, rot, col, r){
  ctx.save(); ctx.translate(cx, cy); ctx.rotate(rot);
  const teeth = 7, j = h*.07;
  ctx.beginPath(); ctx.moveTo(-w/2, -h/2); ctx.lineTo(w/2, -h/2);
  for(let i=1;i<=teeth;i++) ctx.lineTo(w/2 - (i%2 ? j : 0) - r()*j*.5, -h/2 + h*i/teeth);
  ctx.lineTo(-w/2, h/2);
  for(let i=1;i<=teeth;i++) ctx.lineTo(-w/2 + (i%2 ? j : 0) + r()*j*.5, h/2 - h*i/teeth);
  ctx.closePath();
  ctx.save(); ctx.shadowColor = "rgba(20,20,19,.16)"; ctx.shadowBlur = h*.18; ctx.shadowOffsetY = h*.06;
  ctx.fillStyle = hexA(col, .86); ctx.fill(); ctx.restore();
  ctx.save(); ctx.clip();
  ctx.strokeStyle = "rgba(255,255,255,.28)"; ctx.lineWidth = h*.11;
  for(let x = -w/2 - h; x < w/2 + h; x += h*.42){ ctx.beginPath(); ctx.moveTo(x, -h/2 - 2); ctx.lineTo(x + h*.7, h/2 + 2); ctx.stroke(); }
  ctx.fillStyle = "rgba(255,255,255,.12)"; ctx.fillRect(-w/2, -h/2, w, h*.3);
  ctx.restore(); ctx.restore();
}
function drawPaperclip(ctx, x, y, len, rot){
  const s = len/100;
  ctx.save(); ctx.translate(x, y); ctx.rotate(rot); ctx.scale(s, s); ctx.translate(-50, -34);
  const p = new Path2D("M40 6L40 74C40 88 62 88 62 74L62 20C62 10 46 10 46 20L46 64");
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.save(); ctx.shadowColor = "rgba(20,20,19,.28)"; ctx.shadowBlur = 4; ctx.shadowOffsetX = 2; ctx.shadowOffsetY = 3;
  ctx.strokeStyle = "#7B7A74"; ctx.lineWidth = 6.2; ctx.stroke(p); ctx.restore();
  ctx.strokeStyle = "#B9B7AE"; ctx.lineWidth = 3.4; ctx.stroke(p);
  ctx.strokeStyle = "rgba(255,255,255,.75)"; ctx.lineWidth = 1.2; ctx.translate(-.8, -.6); ctx.stroke(p);
  ctx.restore();
}
function drawPin(ctx, x, y, r, col){
  ctx.save();
  ctx.fillStyle = "rgba(20,20,19,.22)"; ctx.beginPath(); ctx.ellipse(x + r*.55, y + r*.75, r*.95, r*.6, .3, 0, Math.PI*2);
  ctx.filter !== undefined && HAS_FILTER ? (ctx.filter = `blur(${r*.2}px)`) : 0; ctx.fill(); if(HAS_FILTER) ctx.filter = "none";
  ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x + r*.08, y + r*.1, r, 0, Math.PI*2); ctx.fill();
  ctx.lineWidth = r*.16; ctx.strokeStyle = "#141413"; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI*2); ctx.stroke();
  ctx.lineWidth = r*.1; ctx.beginPath(); ctx.arc(x, y, r*.52, 0, Math.PI*2); ctx.stroke();
  ctx.fillStyle = "rgba(255,255,255,.75)"; ctx.beginPath(); ctx.ellipse(x - r*.38, y - r*.42, r*.22, r*.13, -.7, 0, Math.PI*2); ctx.fill();
  ctx.restore();
}
function drawPeg(ctx, x, y, len, rot){
  const s = len/100;
  ctx.save(); ctx.translate(x, y); ctx.rotate(rot); ctx.scale(s, s); ctx.translate(-50, -46);
  ctx.lineJoin = "round"; ctx.lineCap = "round";
  const L = new Path2D(), R = new Path2D();
  L.moveTo(37,4); L.lineTo(48,4); L.lineTo(48,96); L.quadraticCurveTo(48,100,44,100); L.lineTo(41,100); L.quadraticCurveTo(37,100,37,96); L.closePath();
  R.moveTo(52,4); R.lineTo(63,4); R.lineTo(63,96); R.quadraticCurveTo(63,100,59,100); R.lineTo(56,100); R.quadraticCurveTo(52,100,52,96); R.closePath();
  ctx.save(); ctx.shadowColor = "rgba(20,20,19,.3)"; ctx.shadowBlur = 5; ctx.shadowOffsetX = 2; ctx.shadowOffsetY = 4;
  ctx.fillStyle = "#D4A27F"; ctx.fill(L); ctx.fill(R); ctx.restore();
  ctx.fillStyle = "#DDB08F"; ctx.save(); ctx.translate(1.6, 1.2); ctx.fill(L); ctx.fill(R); ctx.restore();
  ctx.strokeStyle = "#141413"; ctx.lineWidth = 2.4; ctx.stroke(L); ctx.stroke(R);
  ctx.lineWidth = 1; ctx.strokeStyle = "rgba(20,20,19,.35)";
  ctx.beginPath(); ctx.moveTo(41,14); ctx.lineTo(41,30); ctx.moveTo(44,60); ctx.lineTo(44,84); ctx.moveTo(57,20); ctx.lineTo(57,40); ctx.moveTo(59,66); ctx.lineTo(59,90); ctx.stroke();
  ctx.lineWidth = 3.2; ctx.strokeStyle = "#87867F"; ctx.beginPath(); ctx.arc(50, 38, 6, 0, Math.PI*2); ctx.stroke();
  ctx.lineWidth = 1.4; ctx.strokeStyle = "#141413"; ctx.beginPath(); ctx.arc(50, 38, 7.8, 0, Math.PI*2); ctx.stroke();
  ctx.restore();
}
function drawDeco(ctx, g, ph, type){
  if(!type || type === "none") return;
  const r = rng(((ph ? ph.seed : 9) * 131) ^ (state.decoSeed * 7)), K = g.K, col = DECO_COLORS[state.decoColor].c;
  if(type === "tape") drawTape(ctx, g.W/2 + (r()-.5)*g.W*.16, K*.004, K*.36, K*.085, (r()-.5)*.16, col, r);
  if(type === "tape2"){
    drawTape(ctx, K*.035, K*.035, K*.3, K*.075, -Math.PI/4 + (r()-.5)*.12, col, r);
    drawTape(ctx, g.W - K*.035, g.H - K*.035, K*.3, K*.075, -Math.PI/4 + (r()-.5)*.12, col, r);
  }
  if(type === "clip") drawPaperclip(ctx, g.W*(.14 + r()*.1), 0, K*.26, (r()-.5)*.12);
  if(type === "pin") drawPin(ctx, g.W/2 + (r()-.5)*g.W*.1, g.top*.55, K*.032, col);
  if(type === "peg") drawPeg(ctx, g.W/2 + (r()-.5)*g.W*.12, 0, K*.24, (r()-.5)*.1);
}

/* ================= card drawing ================= */
function drawPlaceholder(ctx,x,y,w,h){
  const sky = ctx.createLinearGradient(0,y,0,y+h);
  sky.addColorStop(0,"#E6DCCB"); sky.addColorStop(1,"#F2E8D8");
  ctx.fillStyle = sky; ctx.fillRect(x,y,w,h);
  const m = Math.min(w,h);
  ctx.fillStyle = "#D97757"; ctx.beginPath(); ctx.arc(x+w*.64, y+h*.47, m*.15, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = "#D4A27F"; ctx.beginPath(); ctx.moveTo(x, y+h*.70);
  ctx.quadraticCurveTo(x+w*.28, y+h*.50, x+w*.60, y+h*.68); ctx.quadraticCurveTo(x+w*.84, y+h*.80, x+w, y+h*.62);
  ctx.lineTo(x+w, y+h); ctx.lineTo(x, y+h); ctx.closePath(); ctx.fill();
  ctx.fillStyle = "#2E2D2A"; ctx.beginPath(); ctx.moveTo(x, y+h*.86);
  ctx.quadraticCurveTo(x+w*.48, y+h*.70, x+w, y+h*.87); ctx.lineTo(x+w, y+h); ctx.lineTo(x, y+h); ctx.closePath(); ctx.fill();
}
function drawPhoto(ctx, g, ph, img, opt){
  const K = g.K, {x0, y0, iw, ih} = g;
  ctx.save();
  ctx.beginPath(); ctx.rect(x0,y0,iw,ih); ctx.clip();
  ctx.fillStyle = "#1C1B19"; ctx.fillRect(x0,y0,iw,ih);
  const f = state.filter, dev = opt.develop;
  if(HAS_FILTER){
    let fs = f==="warm" ? "sepia(.16) saturate(1.12) contrast(1.04) brightness(1.03)"
           : f==="fade" ? "contrast(.86) saturate(.82) brightness(1.06)"
           : f==="mono" ? "grayscale(1) contrast(1.1) brightness(1.02)" : "";
    if(dev != null) fs += ` saturate(${(.15 + .85*dev).toFixed(3)}) contrast(${(.7 + .3*dev).toFixed(3)})`;
    ctx.filter = fs.trim() || "none";
  }
  ctx.imageSmoothingQuality = "high";
  if(ph && img){ const p = photoRect(g, ph); ctx.drawImage(img, p.x, p.y, p.w, p.h); }
  else if(!ph) drawPlaceholder(ctx,x0,y0,iw,ih);
  if(HAS_FILTER) ctx.filter = "none";
  if(f==="warm"){ ctx.fillStyle="rgba(217,119,87,.07)"; ctx.fillRect(x0,y0,iw,ih); }
  if(f==="fade"){ ctx.fillStyle="rgba(240,232,215,.12)"; ctx.fillRect(x0,y0,iw,ih); }
  const rnd = rng(((ph ? ph.seed : 5) * 97) ^ 0x9e37);
  if(state.leak){
    const sideI = Math.floor(rnd()*4), t = .15 + rnd()*.7;
    const pt = [[x0 + iw*t, y0],[x0 + iw, y0 + ih*t],[x0 + iw*t, y0 + ih],[x0, y0 + ih*t]][sideI];
    const R = Math.max(iw, ih)*(.55 + rnd()*.25);
    ctx.globalCompositeOperation = "screen";
    let gr = ctx.createRadialGradient(pt[0], pt[1], 0, pt[0], pt[1], R);
    gr.addColorStop(0, "rgba(255,150,70,.85)"); gr.addColorStop(.3, "rgba(240,90,50,.45)"); gr.addColorStop(1, "rgba(240,90,50,0)");
    ctx.fillStyle = gr; ctx.fillRect(x0,y0,iw,ih);
    const p2 = [x0 + iw*rnd(), y0 + ih*rnd()];
    gr = ctx.createRadialGradient(p2[0], p2[1], 0, p2[0], p2[1], R*.45);
    gr.addColorStop(0, "rgba(255,210,130,.35)"); gr.addColorStop(1, "rgba(255,210,130,0)");
    ctx.fillStyle = gr; ctx.fillRect(x0,y0,iw,ih);
    ctx.globalCompositeOperation = "source-over";
  }
  if(state.fadeEdge){
    for(const [cx,cy] of [[x0,y0],[x0+iw,y0],[x0,y0+ih],[x0+iw,y0+ih]]){
      const gr = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.min(iw,ih)*.42);
      gr.addColorStop(0, "rgba(246,238,222,.55)"); gr.addColorStop(1, "rgba(246,238,222,0)");
      ctx.fillStyle = gr; ctx.fillRect(x0,y0,iw,ih);
    }
    ctx.fillStyle = "rgba(240,228,205,.07)"; ctx.fillRect(x0,y0,iw,ih);
  }
  if(state.vignette){
    const cx = x0+iw/2, cy = y0+ih/2, gr = ctx.createRadialGradient(cx, cy, Math.min(iw,ih)*.32, cx, cy, Math.hypot(iw,ih)*.56);
    gr.addColorStop(0, "rgba(20,16,12,0)"); gr.addColorStop(1, "rgba(20,16,12,.55)");
    ctx.fillStyle = gr; ctx.fillRect(x0,y0,iw,ih);
  }
  if(state.stamp){
    const iso = dateFor(ph);
    if(iso){
      const [y,m,d] = iso.split("-").map(Number), s = .046*K;
      ctx.font = STAMP_FONT(s); ctx.textAlign = "right"; ctx.textBaseline = "alphabetic"; setLS(ctx, (s*.12)+"px");
      const txt = `'${String(y).slice(2)} ${m} ${d}`, tx = x0 + iw - K*.045, ty = y0 + ih - K*.04;
      ctx.shadowColor = "rgba(255,90,20,.9)"; ctx.shadowBlur = s*.35; ctx.fillStyle = "rgba(255,150,60,.92)"; ctx.fillText(txt, tx, ty);
      ctx.shadowBlur = 0; ctx.fillStyle = "rgba(255,196,120,.85)"; ctx.fillText(txt, tx, ty);
      setLS(ctx, "0px");
    }
  }
  if(dev != null){
    const a = 1 - easeIO(Math.min(1, Math.max(0, dev)));
    if(a > 0){ ctx.fillStyle = `rgba(48,52,54,${a.toFixed(3)})`; ctx.fillRect(x0,y0,iw,ih); }
  }
  let t = ctx.createLinearGradient(0,y0,0,y0+K*.022);
  t.addColorStop(0,"rgba(0,0,0,.18)"); t.addColorStop(1,"rgba(0,0,0,0)");
  ctx.fillStyle = t; ctx.fillRect(x0,y0,iw,K*.022);
  t = ctx.createLinearGradient(x0,0,x0+K*.014,0);
  t.addColorStop(0,"rgba(0,0,0,.09)"); t.addColorStop(1,"rgba(0,0,0,0)");
  ctx.fillStyle = t; ctx.fillRect(x0,y0,K*.014,ih);
  ctx.restore();
  ctx.strokeStyle = PAPERS[state.paper].well; ctx.lineWidth = Math.max(1, K*.0013); ctx.strokeRect(x0,y0,iw,ih);
}
function drawCard(ctx, g, ph, img, idx, opt){
  opt = opt || {};
  const pal = PAPERS[state.paper], K = g.K, r = .013*K;
  const lay = illoLayout(g, ph, idx);
  const freeEls = lay.mode === "free" ? freeElements(g, ph, idx) : null;
  ctx.save();
  rr(ctx,0,0,g.W,g.H,r); ctx.fillStyle = pal.bg; ctx.fill();
  ctx.save(); ctx.clip();
  const sh = ctx.createLinearGradient(0,0,g.W,g.H);
  sh.addColorStop(0,"rgba(255,255,255,.08)"); sh.addColorStop(1,"rgba(0,0,0,.035)");
  ctx.fillStyle = sh; ctx.fillRect(0,0,g.W,g.H);
  if(state.grain){
    const pat = ctx.createPattern(noiseTile,"repeat");
    try{ pat.setTransform(new DOMMatrix().scale(Math.max(.5, K/1000))); }catch(e){}
    ctx.globalAlpha = .9; ctx.fillStyle = pat; ctx.fillRect(0,0,g.W,g.H); ctx.globalAlpha = 1;
  }
  drawPhoto(ctx, g, ph, img, opt);
  drawText(ctx, g, ph, idx, lay);
  if(freeEls) drawFreeText(ctx, freeEls);
  if(!lay.sticker) lay.items.forEach(it => drawIllo(ctx, it.id, it.cx, it.cy, it.size, it.rot, state.paper, false, state.illoTheme));
  drawDoodles(ctx, g, docOf(ph));
  ctx.restore();
  rr(ctx,0,0,g.W,g.H,r); ctx.strokeStyle = pal.edge; ctx.lineWidth = Math.max(1,K*.0016); ctx.stroke();
  if(lay.sticker) lay.items.forEach(it => drawIllo(ctx, it.id, it.cx, it.cy, it.size, it.rot, "ivory", true, state.illoTheme));
  if(freeEls){
    const F = freeOf(ph);
    freeEls.forEach(e => { if(e.type === "illo") drawIllo(ctx, e.id, e.cx, e.cy, e.size, e.rot, state.paper, F.sticker, e.theme); });
  }
  drawDeco(ctx, g, ph, opt.deco !== undefined ? opt.deco : state.deco);
  ctx.restore();
  return freeEls;
}
function drawCardShadowed(ctx, g, ph, img, idx, opt){ // ctx at card centre
  const K = g.K, r = .013*K, pal = PAPERS[state.paper];
  ctx.save();
  ctx.shadowColor = "rgba(20,20,19,.22)"; ctx.shadowBlur = .07*K; ctx.shadowOffsetY = .026*K;
  rr(ctx,-g.W/2,-g.H/2,g.W,g.H,r); ctx.fillStyle = pal.bg; ctx.fill();
  ctx.shadowColor = "rgba(20,20,19,.14)"; ctx.shadowBlur = .008*K; ctx.shadowOffsetY = .004*K; ctx.fill();
  ctx.restore();
  ctx.save(); ctx.translate(-g.W/2,-g.H/2); drawCard(ctx, g, ph, img, idx, opt); ctx.restore();
}
function renderCard(ph, img, idx, K, opt){
  opt = opt || {};
  const g = geom(K, ph), P = opt.pad ? PAD*K : 0;
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(g.W + 2*P)); c.height = Math.max(1, Math.round(g.H + 2*P));
  const x = c.getContext("2d"); x.translate(P, P);
  drawCard(x, g, ph, img, idx, opt);
  return c;
}

/* ================= image cache ================= */
const imgCache = new Map();
async function getImg(ph){
  if(imgCache.has(ph.id)){ const im = imgCache.get(ph.id); imgCache.delete(ph.id); imgCache.set(ph.id, im); return im; }
  const im = new Image(); im.src = ph.url; await im.decode();
  imgCache.set(ph.id, im);
  while(imgCache.size > 3){
    const k = imgCache.keys().next().value;
    if(curPhoto() && k === curPhoto().id){ const v = imgCache.get(k); imgCache.delete(k); imgCache.set(k, v); if(imgCache.size <= 3) break; continue; }
    imgCache.delete(k);
  }
  return im;
}

/* ================= preview ================= */
const stage = $("#stage"), canvas = $("#card"), ctx = canvas.getContext("2d");
let view = {s:1, pad:0};
function draw(){
  const rect = stage.getBoundingClientRect();
  if(rect.width < 10 || rect.height < 10) return;
  const ph = curPhoto(), narrow = rect.width < 600, hasStrip = photos.length > 0;
  const padX = Math.max(narrow ? 10 : 30, rect.width*.05);
  const padTop = narrow ? 46 : Math.max(56, rect.height*.07);
  const trayEl = $("#tray"), trayOpen = trayEl && !trayEl.hidden;
  const padBot = trayOpen ? trayEl.offsetHeight + 22 : hasStrip ? (narrow ? 66 : 90) : (narrow ? 20 : Math.max(40, rect.height*.05));
  const g1 = geom(1, ph), TW = g1.W + 2*PAD, TH = g1.H + 2*PAD;
  let bw = TW, bh = TH;
  if(state.withBg && state.tilt){
    const a = state.tilt*Math.PI/180, c = Math.abs(Math.cos(a)), s = Math.abs(Math.sin(a));
    bw = TW*c + TH*s; bh = TW*s + TH*c;
  }
  const s = Math.max(.01, Math.min((rect.width-2*padX)/bw, (rect.height-padTop-padBot)/bh));
  const dpr = Math.min(3, window.devicePixelRatio || 1);
  const cssW = TW*s, cssH = TH*s;
  canvas.style.width = cssW+"px"; canvas.style.height = cssH+"px";
  canvas.style.marginTop = (padTop-padBot)+"px";
  const pw = Math.round(cssW*dpr), phh = Math.round(cssH*dpr);
  if(canvas.width !== pw) canvas.width = pw;
  if(canvas.height !== phh) canvas.height = phh;
  ctx.setTransform(1,0,0,1,0,0); ctx.clearRect(0,0,pw,phh);
  ctx.setTransform(pw/(cssW*dpr), 0, 0, phh/(cssH*dpr), 0, 0);
  ctx.translate(PAD*s*dpr, PAD*s*dpr);
  let img = null;
  if(ph){ img = imgCache.get(ph.id) || null; if(!img) getImg(ph).then(schedule).catch(() => {}); }
  const els = drawCard(ctx, geom(s*dpr, ph), ph, img, Math.max(0,cur));
  if(els && sel && tool === "move"){ const e = els.find(sameSel); if(e) drawSelection(ctx, e, dpr); else { sel = null; syncSel(); } }
  canvas.style.transform = state.withBg ? `rotate(${state.tilt}deg)` : "none";
  stage.style.background = state.withBg ? BGS[state.bg].c : "";
  view = {s, pad: PAD*s};
  updateDims();
}
function drawSelection(ctx, e, dpr){
  const pad = 6*dpr, w = e.w/2 + pad, h = e.h/2 + pad;
  ctx.save(); ctx.translate(e.cx, e.cy); ctx.rotate(e.rot);
  ctx.setLineDash([5*dpr, 4*dpr]); ctx.lineWidth = 1.5*dpr; ctx.strokeStyle = "#C6613F"; ctx.strokeRect(-w, -h, 2*w, 2*h);
  ctx.setLineDash([]);
  ctx.beginPath(); ctx.arc(w, h, 7*dpr, 0, Math.PI*2); ctx.fillStyle = "#FFFFFF"; ctx.fill(); ctx.lineWidth = 2*dpr; ctx.stroke();
  ctx.beginPath(); ctx.arc(w, h, 2.5*dpr, 0, Math.PI*2); ctx.fillStyle = "#C6613F"; ctx.fill();
  if(e.type === "illo"){
    const r = 8*dpr;
    ctx.beginPath(); ctx.arc(-w, -h, r, 0, Math.PI*2); ctx.fillStyle = "#C6613F"; ctx.fill();
    ctx.strokeStyle = "#FFFFFF"; ctx.lineWidth = 1.8*dpr; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(-w - r*.38, -h - r*.38); ctx.lineTo(-w + r*.38, -h + r*.38); ctx.moveTo(-w + r*.38, -h - r*.38); ctx.lineTo(-w - r*.38, -h + r*.38); ctx.stroke();
  }
  ctx.restore();
}
let raf = 0;
function schedule(){ queueCommit(); if(!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); }); }
new ResizeObserver(() => { if(!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); }); }).observe(stage);

/* ================= history (undo / redo) ================= */
const hist = []; let hIdx = -1, commitT = 0;
function snap(){ return JSON.stringify({s: state, p: photos.map(p => [p.id, p.zoom, p.px, p.py, p.caption, p.free, p.doodles]), d: scratch.doodles}); }
function commit(){
  const now = snap();
  if(hist[hIdx] === now) return;
  hist.splice(hIdx + 1); hist.push(now);
  if(hist.length > 120) hist.shift();
  hIdx = hist.length - 1; syncHistBtns();
}
function queueCommit(){ clearTimeout(commitT); commitT = setTimeout(commit, 320); }
function replaceState(s){
  const base = clone(DEFAULTS);
  Object.keys(state).forEach(k => delete state[k]);
  Object.assign(state, base, s);
  for(const k of ["free","brush","collage","print"]) state[k] = Object.assign(clone(DEFAULTS[k]), s && s[k] || {});
}
function restore(str){
  const o = JSON.parse(str);
  replaceState(o.s);
  o.p.forEach(([id,zoom,px,py,caption,free,doodles]) => { const p = photos.find(x => x.id === id); if(p) Object.assign(p, {zoom,px,py,caption,free,doodles}); });
  scratch.doodles = o.d || [];
  sel = null; syncControls(); syncUI(); draw();
}
function undo(){ clearTimeout(commitT); commit(); if(hIdx > 0){ hIdx--; restore(hist[hIdx]); syncHistBtns(); } }
function redo(){ clearTimeout(commitT); if(hIdx < hist.length - 1){ hIdx++; restore(hist[hIdx]); syncHistBtns(); } }
function syncHistBtns(){ $("#undoBtn").disabled = hIdx <= 0; $("#redoBtn").disabled = hIdx >= hist.length - 1; }
$("#undoBtn").onclick = undo; $("#redoBtn").onclick = redo;

/* ================= export ================= */
function decoPadOn(){ return state.deco !== "none"; }
function exportPlan(ph){
  let K = EXPORT_K[state.quality];
  const measure = k => { const g = geom(k, ph); if(state.withBg){ const b = bgBox(g); return [b.w,b.h]; } const P = decoPadOn() ? 2*PAD*k : 0; return [g.W + P, g.H + P]; };
  let [w,h] = measure(K);
  if(w*h > MAX_PIXELS){ K *= Math.sqrt(MAX_PIXELS/(w*h)); [w,h] = measure(K); }
  return {K, w:Math.round(w), h:Math.round(h)};
}
function updateDims(){ const p = exportPlan(curPhoto()); $("#dims").textContent = `${p.w} × ${p.h}`; }
const toBlob = (c, mime, q) => new Promise((res,rej) => c.toBlob(b => b ? res(b) : rej(new Error("toBlob")), mime, q));
async function renderExport(ph, idx, format){
  const img = ph ? await getImg(ph) : null;
  const {K,w,h} = exportPlan(ph);
  let c;
  if(state.withBg){
    const g = geom(K, ph), box = bgBox(g);
    c = document.createElement("canvas"); c.width = w; c.height = h;
    const x = c.getContext("2d");
    x.fillStyle = BGS[state.bg].c; x.fillRect(0,0,w,h);
    x.translate(w/2, h/2); x.rotate(box.a);
    drawCardShadowed(x, g, ph, img, idx);
  } else {
    c = renderCard(ph, img, idx, K, {pad: decoPadOn()});
    if(format === "jpg"){ const j = document.createElement("canvas"); j.width = c.width; j.height = c.height; const x = j.getContext("2d"); x.fillStyle = "#FFFFFF"; x.fillRect(0,0,j.width,j.height); x.drawImage(c,0,0); c = j; }
  }
  const blob = await toBlob(c, format === "jpg" ? "image/jpeg" : "image/png", .93);
  c.width = c.height = 1;
  return blob;
}
function baseName(ph){ return ph ? ph.name.replace(/\.[^.]+$/,"").replace(/[\\/:*?"<>|]+/g,"_").slice(0,60) : "instant-frame"; }
function fileName(ph, idx, ext){ return `${String(idx+1).padStart(3,"0")}-${baseName(ph)}.${ext}`; }

/* ZIP (store) */
const CRC_T = (() => { const t = new Uint32Array(256); for(let n=0;n<256;n++){ let c=n; for(let k=0;k<8;k++) c = c&1 ? 0xEDB88320^(c>>>1) : c>>>1; t[n]=c>>>0; } return t; })();
function crc32(u8){ let c = 0xFFFFFFFF; for(let i=0;i<u8.length;i++) c = CRC_T[(c^u8[i])&0xFF]^(c>>>8); return (c^0xFFFFFFFF)>>>0; }
async function makeZip(files){
  const enc = new TextEncoder(), parts = [], central = [];
  const d = new Date(), dosTime = (d.getHours()<<11)|(d.getMinutes()<<5)|(d.getSeconds()>>1), dosDate = ((d.getFullYear()-1980)<<9)|((d.getMonth()+1)<<5)|d.getDate();
  let offset = 0;
  for(const f of files){
    const data = new Uint8Array(await f.blob.arrayBuffer()), crc = crc32(data), name = enc.encode(f.name);
    const lh = new DataView(new ArrayBuffer(30));
    lh.setUint32(0,0x04034b50,true); lh.setUint16(4,20,true); lh.setUint16(6,0x0800,true); lh.setUint16(8,0,true);
    lh.setUint16(10,dosTime,true); lh.setUint16(12,dosDate,true); lh.setUint32(14,crc,true);
    lh.setUint32(18,data.length,true); lh.setUint32(22,data.length,true); lh.setUint16(26,name.length,true); lh.setUint16(28,0,true);
    parts.push(lh, name, f.blob);
    const ch = new DataView(new ArrayBuffer(46));
    ch.setUint32(0,0x02014b50,true); ch.setUint16(4,20,true); ch.setUint16(6,20,true); ch.setUint16(8,0x0800,true); ch.setUint16(10,0,true);
    ch.setUint16(12,dosTime,true); ch.setUint16(14,dosDate,true); ch.setUint32(16,crc,true); ch.setUint32(20,data.length,true); ch.setUint32(24,data.length,true);
    ch.setUint16(28,name.length,true); ch.setUint16(30,0,true); ch.setUint16(32,0,true); ch.setUint16(34,0,true); ch.setUint16(36,0,true);
    ch.setUint32(38,0,true); ch.setUint32(42,offset,true);
    central.push(ch, name);
    offset += 30 + name.length + data.length;
  }
  const cdSize = central.reduce((s,p) => s + p.byteLength, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0,0x06054b50,true); end.setUint16(8,files.length,true); end.setUint16(10,files.length,true);
  end.setUint32(12,cdSize,true); end.setUint32(16,offset,true);
  return new Blob([...parts, ...central, end], {type:"application/zip"});
}
/* PDF with one full-page JPEG per page */
function makePdf(pages, wmm, hmm){
  const enc = new TextEncoder(), parts = [], offs = []; let len = 0;
  const push = x => { const b = typeof x === "string" ? enc.encode(x) : x; parts.push(b); len += b.length; };
  const W = (wmm/25.4*72).toFixed(2), H = (hmm/25.4*72).toFixed(2);
  const obj = (num, body) => { offs[num] = len; push(`${num} 0 obj\n`); body(); push("\nendobj\n"); };
  push("%PDF-1.4\n"); push(new Uint8Array([37,226,227,207,211,10]));
  const n = pages.length, kids = pages.map((_,i) => `${3+i*3} 0 R`).join(" ");
  obj(1, () => push("<< /Type /Catalog /Pages 2 0 R >>"));
  obj(2, () => push(`<< /Type /Pages /Kids [${kids}] /Count ${n} >>`));
  pages.forEach((p,i) => {
    const po = 3+i*3, io = po+1, co = po+2, cs = `q ${W} 0 0 ${H} 0 0 cm /Im${i} Do Q`;
    obj(po, () => push(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${W} ${H}] /Resources << /XObject << /Im${i} ${io} 0 R >> >> /Contents ${co} 0 R >>`));
    obj(io, () => { push(`<< /Type /XObject /Subtype /Image /Width ${p.w} /Height ${p.h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${p.jpeg.length} >>\nstream\n`); push(p.jpeg); push("\nendstream"); });
    obj(co, () => push(`<< /Length ${cs.length} >>\nstream\n${cs}\nendstream`));
  });
  const xref = len, total = 3 + n*3;
  push(`xref\n0 ${total}\n0000000000 65535 f \n`);
  for(let k=1;k<total;k++) push(String(offs[k]).padStart(10,"0") + " 00000 n \n");
  push(`trailer\n<< /Size ${total} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`);
  return new Blob(parts, {type:"application/pdf"});
}

let downloads = null;
function initDownloads(){
  try{ if(window.claude && typeof window.claude.use === "function"){ window.claude.use("downloads").then(d => { downloads = d; }).catch(() => {}); return true; } }catch(e){}
  return false;
}
if(!initDownloads()) window.addEventListener("load", initDownloads);

/* modal / busy */
let modalURLs = [];
function closeModal(){ $("#modal").hidden = true; modalURLs.forEach(u => URL.revokeObjectURL(u)); modalURLs = []; $("#modalBody").innerHTML = ""; }
function openModal(title, body, actions){
  closeModal();
  $("#modalTitle").textContent = title || ""; $("#modalTitle").hidden = !title;
  const b = $("#modalBody"); if(typeof body === "string") b.innerHTML = body; else { b.innerHTML = ""; b.appendChild(body); }
  const f = $("#modalFoot"); f.innerHTML = "";
  (actions || [{label:"关闭"}]).forEach(a => {
    const btn = document.createElement("button"); btn.className = "btn " + (a.cls || ""); btn.textContent = a.label;
    btn.onclick = async () => { if(a.onClick){ const keep = await a.onClick(btn); if(keep) return; } closeModal(); };
    f.appendChild(btn);
  });
  $("#modal").hidden = false;
}
$("#modal").addEventListener("click", e => { if(e.target.id === "modal") closeModal(); });
function busy(msg){ $("#busy").hidden = !msg; if(msg) $("#busyText").textContent = msg; }
function showGallery(blobs, note){
  const gal = document.createElement("div"); gal.className = "gallery" + (blobs.length > 1 ? " many" : "");
  blobs.forEach(b => { const u = URL.createObjectURL(b); const im = document.createElement("img"); im.src = u; gal.appendChild(im); setTimeout(() => modalURLs.push(u)); });
  const wrap = document.createElement("div"); wrap.style.cssText = "display:flex;flex-direction:column;gap:10px;align-items:center";
  wrap.appendChild(gal); const p = document.createElement("p"); p.textContent = note || "右键（手机上长按）图片，即可另存为。"; wrap.appendChild(p);
  openModal("", wrap);
}
const EMBEDDED = (() => { try{ return window.self !== window.top; }catch(e){ return true; } })();
function directDownload(name, blob){
  try{
    const u = URL.createObjectURL(blob), a = document.createElement("a");
    a.href = u; a.download = name; a.rel = "noopener"; a.style.display = "none";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(u), 60000);
    return true;
  }catch(e){ return false; }
}
async function offerSave(name, blob, fallback){
  if(!downloads && !EMBEDDED){
    const b = typeof blob === "string" ? new Blob([blob], {type: name.endsWith(".json") ? "application/json" : "text/plain"}) : blob;
    if(directDownload(name, b)){ toast("已开始下载"); return true; }
  }
  if(downloads){
    try{ await downloads.save({filename:name, data:blob}); toast("已保存"); return true; }
    catch(err){
      const code = err && err.code;
      if(code === "declined"){ toast("已取消保存"); return false; }
      if(code === "rate_limited"){ toast("上一个保存还没处理完，稍后再试"); return false; }
    }
  }
  if(fallback) fallback(); else toast("当前环境不支持直接下载");
  return false;
}

$("#export").addEventListener("click", async () => {
  busy("正在生成图片…");
  try{
    await fontsReady();
    const ph = curPhoto(), idx = Math.max(0,cur), ext = state.format;
    const blob = await renderExport(ph, idx, ext);
    busy(null);
    await offerSave(fileName(ph, idx, ext), blob, () => showGallery([blob]));
  }catch(e){ toast("导出失败：" + (e.message || e)); }
  finally{ busy(null); schedule(); }
});
$("#exportAll").addEventListener("click", async () => {
  if(!photos.length) return;
  const ext = state.format, files = [];
  try{
    await fontsReady();
    for(let i=0;i<photos.length;i++){
      busy(`正在生成 ${i+1} / ${photos.length}…`);
      files.push({name:fileName(photos[i], i, ext), blob: await renderExport(photos[i], i, ext)});
      await new Promise(r => setTimeout(r, 0));
    }
    if(downloads){
      busy("正在打包 ZIP…");
      const zip = await makeZip(files); busy(null);
      await offerSave(`instant-frame-${todayISO.replace(/-/g,"")}-${files.length}.zip`, zip, () => showGallery(files.map(f => f.blob)));
    } else { busy(null); showGallery(files.map(f => f.blob)); }
  }catch(e){ toast("批量导出失败：" + (e.message || e)); }
  finally{ busy(null); if(curPhoto()) getImg(curPhoto()).then(() => draw()); }
});
$("#copy").addEventListener("click", async () => {
  const ph = curPhoto(), idx = Math.max(0,cur);
  try{
    if(!navigator.clipboard || !window.ClipboardItem) throw 0;
    const p = fontsReady().then(() => renderExport(ph, idx, "png"));
    await navigator.clipboard.write([new ClipboardItem({"image/png": p})]);
    toast("已复制到剪贴板");
  }catch(e){
    try{ showGallery([await renderExport(ph, idx, "png")]); }catch(_){ toast("复制失败，请用「导出」"); }
  }
});

/* develop video */
$("#exportVideo").addEventListener("click", async () => {
  const types = ["video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
  if(!window.MediaRecorder || !HTMLCanvasElement.prototype.captureStream){ toast("这个浏览器不支持录制视频，试试 Chrome 或 Safari"); return; }
  const mime = types.find(t => { try{ return MediaRecorder.isTypeSupported(t); }catch(e){ return false; } });
  await fontsReady();
  const ph = curPhoto(), idx = Math.max(0,cur), img = ph ? await getImg(ph) : null;
  const K = 720, g = geom(K, ph), bgc = state.withBg ? BGS[state.bg].c : "#F0EEE6";
  const padv = .24*K;
  let CW = Math.round(g.W + 2*padv), CH = Math.round(g.H + 2*padv); CW += CW % 2; CH += CH % 2;
  const c = document.createElement("canvas"); c.width = CW; c.height = CH;
  c.style.cssText = "position:fixed;left:-99999px;top:0"; document.body.appendChild(c);
  const x = c.getContext("2d"), tilt = state.withBg ? state.tilt : -2.5;
  const frame = t => {
    const slide = 1 - Math.pow(1 - Math.min(1, t/800), 3), dev = Math.min(1, Math.max(0, (t - 600)/3000));
    x.setTransform(1,0,0,1,0,0); x.fillStyle = bgc; x.fillRect(0,0,CW,CH);
    x.save(); x.translate(CW/2, CH/2 + (1 - slide)*CH*.7); x.rotate((tilt*slide - 8*(1 - slide))*Math.PI/180);
    drawCardShadowed(x, g, ph, img, idx, {develop: dev});
    x.restore();
  };
  let rec;
  try{
    frame(0);
    const stream = c.captureStream(30);
    rec = new MediaRecorder(stream, mime ? {mimeType: mime, videoBitsPerSecond: 8e6} : undefined);
  }catch(e){ c.remove(); toast("无法开始录制：" + (e.message || e)); return; }
  const chunks = []; rec.ondataavailable = e => { if(e.data && e.data.size) chunks.push(e.data); };
  const stopped = new Promise(res => rec.onstop = res);
  busy("正在录制显影动画…（约 5 秒）");
  rec.start(200);
  const D = 4800, t0 = performance.now();
  await new Promise(res => { const loop = () => { const t = performance.now() - t0; frame(t); if(t < D) requestAnimationFrame(loop); else res(); }; requestAnimationFrame(loop); });
  rec.stop(); await stopped; c.remove(); busy(null);
  const type = rec.mimeType || mime || "video/webm", ext = type.includes("mp4") ? "mp4" : "webm";
  const blob = new Blob(chunks, {type: type.split(";")[0]});
  if(!blob.size){ toast("录制失败，请重试"); return; }
  await offerSave(`${baseName(ph)}-develop.${ext}`, blob, () => {
    const u = URL.createObjectURL(blob); const v = document.createElement("video");
    v.src = u; v.controls = true; v.autoplay = true; v.loop = true; v.muted = true; v.playsInline = true; v.style.cssText = "max-width:100%;max-height:62vh;border-radius:8px";
    const w = document.createElement("div"); w.className = "gallery"; w.appendChild(v);
    const p = document.createElement("p"); p.textContent = "右键视频（或点菜单）即可另存为。";
    const box = document.createElement("div"); box.style.cssText = "display:flex;flex-direction:column;gap:10px;align-items:center"; box.append(w, p);
    openModal("显影动画", box); modalURLs.push(u);
  });
});

/* collage */
const COLLAGE_RATIOS = {"3:4":[3,4], "4:5":[4,5], "1:1":[1,1], "9:16":[9,16]};
async function renderCollage(longSide){
  const [a,b] = COLLAGE_RATIOS[state.collage.canvas];
  let CW, CH; if(a >= b){ CW = longSide; CH = Math.round(longSide*b/a); } else { CH = longSide; CW = Math.round(longSide*a/b); }
  const c = document.createElement("canvas"); c.width = CW; c.height = CH;
  const x = c.getContext("2d"), bgc = BGS[state.collage.bg].c;
  x.fillStyle = bgc; x.fillRect(0,0,CW,CH);
  const pat = x.createPattern(noiseTile, "repeat"); try{ pat.setTransform(new DOMMatrix().scale(Math.max(1, longSide/1200))); }catch(e){}
  x.globalAlpha = .7; x.fillStyle = pat; x.fillRect(0,0,CW,CH); x.globalAlpha = 1;
  const list = photos.length ? photos.slice(0,12).map((p,i) => ({ph:p, idx:i})) : [0,1,2].map(i => ({ph:null, idx:i}));
  const n = list.length, r = rng(state.collage.seed*977 + n*13), layout = state.collage.layout;
  const m = Math.min(CW,CH)*.06, avgR = list.reduce((s,it) => { const g = geom(1,it.ph); return s + g.W/g.H; }, 0)/n;
  const slots = [], strings = [];
  if(layout === "line"){
    const maxPer = n <= 2 ? n : (CW/CH < .7 ? 2 : 3), rows = Math.ceil(n/maxPer), perRow = Math.ceil(n/rows), rowH = (CH - 2*m)/rows, cellW = (CW - 2*m)/perRow;
    for(let row=0; row<rows; row++){
      const items = list.slice(row*perRow, row*perRow + perRow), k = items.length;
      const sy = m + row*rowH + rowH*.08, sag = rowH*.07;
      strings.push({sy, sag});
      items.forEach((it,j) => {
        const xc = CW/2 + (j - (k-1)/2)*cellW, t = xc/CW, ys = sy + 4*sag*t*(1-t);
        slots.push({it, cx:xc, cy:null, top:ys, maxW:cellW*.84, maxH:rowH*.84, rot:(r()-.5)*.14, peg:true});
      });
    }
  } else {
    let best = null;
    const gap = layout === "grid" ? m*.55 : 0;
    for(let cols=1; cols<=n; cols++){
      const rows = Math.ceil(n/cols), cw = (CW - 2*m - (cols-1)*gap)/cols, ch = (CH - 2*m - (rows-1)*gap)/rows;
      const s = Math.min(cw/avgR, ch);
      if(!best || s > best.s) best = {cols, rows, cw, ch, s};
    }
    const {cols, cw, ch} = best;
    list.forEach((it,i) => {
      const row = Math.floor(i/cols), col = i%cols, inRow = Math.min(cols, n - row*cols), off = (cols - inRow)*(cw + gap)/2;
      let cx = m + off + col*(cw + gap) + cw/2, cy = m + row*(ch + gap) + ch/2, maxW = cw*.9, maxH = ch*.9, rot = (r()-.5)*.05;
      if(layout === "scatter"){ cx += (r()-.5)*cw*.18; cy += (r()-.5)*ch*.16; maxW = cw*(n > 1 ? 1.04 : .86); maxH = ch*(n > 1 ? 1.04 : .86); rot = (r()-.5)*.38; }
      slots.push({it, cx, cy, maxW, maxH, rot});
    });
    if(layout === "scatter") slots.sort(() => r() - .5);
  }
  const ink = state.collage.bg === "slate" ? "#E8E6DC" : "#3D3D3A";
  strings.forEach(s => { x.strokeStyle = ink; x.lineWidth = Math.max(1.5, Math.min(CW,CH)*.0035); x.beginPath(); x.moveTo(-10, s.sy); x.quadraticCurveTo(CW/2, s.sy + 2*s.sag, CW + 10, s.sy); x.stroke(); });
  for(const sl of slots){
    const g1 = geom(1, sl.it.ph), K = Math.min(sl.maxW/g1.W, sl.maxH/g1.H);
    const img = sl.it.ph ? await getImg(sl.it.ph) : null;
    const card = renderCard(sl.it.ph, img, sl.it.idx, K, {pad:true, deco: sl.peg ? "none" : undefined});
    const g = geom(K, sl.it.ph), P = PAD*K;
    x.save();
    if(sl.peg){ x.translate(sl.cx, sl.top + K*.02); x.rotate(sl.rot); x.translate(0, g.H/2 - K*.02); }
    else {
      const c = Math.abs(Math.cos(sl.rot)), sn = Math.abs(Math.sin(sl.rot)), hw = (g.W*c + g.H*sn)/2, hh = (g.W*sn + g.H*c)/2, e = m*.35;
      const cx = Math.min(CW - e - hw, Math.max(e + hw, sl.cx)), cy = Math.min(CH - e - hh, Math.max(e + hh, sl.cy));
      x.translate(hw*2 < CW - 2*e ? cx : CW/2, hh*2 < CH - 2*e ? cy : CH/2); x.rotate(sl.rot);
    }
    x.shadowColor = "rgba(20,20,19,.24)"; x.shadowBlur = K*.06; x.shadowOffsetY = K*.024;
    x.drawImage(card, -g.W/2 - P, -g.H/2 - P);
    x.restore();
    if(sl.peg){ x.save(); x.translate(sl.cx, sl.top); x.rotate(sl.rot); drawPeg(x, 0, K*.04, K*.2, 0); x.restore(); }
  }
  return c;
}
$("#collageOpen").addEventListener("click", async () => {
  await fontsReady();
  const img = document.createElement("img"); img.style.cssText = "max-width:100%;max-height:64vh;border-radius:8px;box-shadow:0 8px 30px rgba(20,20,19,.12)";
  const refresh = async () => { busy("正在拼图…"); const c = await renderCollage(1100); const b = await toBlob(c, "image/jpeg", .9); busy(null); const u = URL.createObjectURL(b); modalURLs.push(u); img.src = u; };
  const wrap = document.createElement("div"); wrap.className = "gallery"; wrap.appendChild(img);
  openModal("拼图预览", wrap, [
    {label:"换个排列", onClick: async () => { state.collage.seed = (state.collage.seed + 1) % 100000; await refresh(); return true; }},
    {label:"导出拼图", cls:"primary", onClick: async () => {
      busy("正在生成高清拼图…");
      try{
        const ext = state.format, long = COLLAGE_LONG[state.quality];
        const [a,b] = COLLAGE_RATIOS[state.collage.canvas]; const area = long*long*Math.min(a,b)/Math.max(a,b);
        const c = await renderCollage(area > MAX_PIXELS ? Math.floor(long*Math.sqrt(MAX_PIXELS/area)) : long);
        const blob = await toBlob(c, ext === "jpg" ? "image/jpeg" : "image/png", .92); busy(null);
        await offerSave(`instant-frame-collage-${state.collage.layout}.${ext}`, blob, () => showGallery([blob]));
      }catch(e){ busy(null); toast("拼图导出失败：" + (e.message || e)); }
      return true;
    }},
    {label:"关闭"}
  ]);
  await refresh();
});

/* print */
async function renderPrintPdf(){
  const pxmm = 300/25.4, PW = Math.round(210*pxmm), PH = Math.round(297*pxmm), margin = 10*pxmm, gap = 6*pxmm;
  const list = photos.length ? photos.map((p,i) => ({ph:p, idx:i})) : [{ph:null, idx:0}];
  const cw = PRINT_SIZES[state.print.size].w*pxmm;
  const ch = Math.max(...list.map(it => { const g = geom(1, it.ph); return cw*g.H/g.W; }));
  const cols = Math.max(1, Math.floor((PW - 2*margin + gap)/(cw + gap))), rows = Math.max(1, Math.floor((PH - 2*margin + gap)/(ch + gap))), per = cols*rows;
  let items = list;
  if(state.print.fill){ const tot = Math.ceil(list.length/per)*per; items = Array.from({length: tot}, (_,i) => list[i % list.length]); }
  const offX = (PW - (cols*cw + (cols-1)*gap))/2, offY = (PH - (rows*ch + (rows-1)*gap))/2;
  const pages = [];
  for(let p0 = 0; p0 < items.length; p0 += per){
    busy(`正在排版第 ${pages.length + 1} 页…`);
    const c = document.createElement("canvas"); c.width = PW; c.height = PH;
    const x = c.getContext("2d"); x.fillStyle = "#FFFFFF"; x.fillRect(0,0,PW,PH);
    const pageItems = items.slice(p0, p0 + per);
    for(let i=0;i<pageItems.length;i++){
      const it = pageItems[i], row = Math.floor(i/cols), col = i%cols;
      const g1 = geom(1, it.ph), K = cw/g1.W, hh = g1.H*K;
      const X = offX + col*(cw + gap), Y = offY + row*(ch + gap) + (ch - hh)/2;
      const img = it.ph ? await getImg(it.ph) : null;
      x.drawImage(renderCard(it.ph, img, it.idx, K, {deco:"none"}), X, Y);
      if(state.print.marks){
        x.strokeStyle = "#8A8984"; x.lineWidth = .25*pxmm; const o = 1*pxmm, l = 3.5*pxmm;
        for(const [cx,cy,sx,sy] of [[X,Y,-1,-1],[X+cw,Y,1,-1],[X,Y+hh,-1,1],[X+cw,Y+hh,1,1]]){
          x.beginPath(); x.moveTo(cx + sx*o, cy); x.lineTo(cx + sx*(o+l), cy); x.moveTo(cx, cy + sy*o); x.lineTo(cx, cy + sy*(o+l)); x.stroke();
        }
      }
    }
    x.fillStyle = "#B0AEA5"; x.font = `${2.4*pxmm}px "JetBrains Mono", monospace`; x.textAlign = "right";
    x.fillText(`Instant Frame · ${PRINT_SIZES[state.print.size].n} ${PRINT_SIZES[state.print.size].w}mm · ${pages.length + 1}`, PW - margin, PH - 4*pxmm);
    const jpeg = new Uint8Array(await (await toBlob(c, "image/jpeg", .92)).arrayBuffer());
    pages.push({jpeg, w:PW, h:PH}); c.width = c.height = 1;
  }
  busy(null);
  return makePdf(pages, 210, 297);
}
$("#printPdf").addEventListener("click", async () => {
  try{
    await fontsReady();
    const pdf = await renderPrintPdf();
    await offerSave(`instant-frame-print-${state.print.size}.pdf`, pdf, () => {
      const u = URL.createObjectURL(pdf);
      openModal("打印 PDF", `<p>当前环境不能直接下载 PDF。<br><a href="${u}" target="_blank" rel="noopener">在新窗口打开 PDF</a>，再用浏览器的打印或保存。</p>`);
      modalURLs.push(u);
    });
  }catch(e){ busy(null); toast("PDF 生成失败：" + (e.message || e)); }
});

/* ================= fonts ================= */
function fontsReady(){
  if(!document.fonts) return Promise.resolve();
  const f = FONTS[state.font];
  const caps = [state.caption, ...photos.map(p => p.caption || "")].join("").replace(/\s+/g,"") || "Aa";
  const meta = [dateText(curPhoto()), state.note, "年月日0123456789·"].join(" ");
  return Promise.all([document.fonts.load(f.css(40), caps), document.fonts.load(META_FONT(20), meta), document.fonts.load(STAMP_FONT(20), "'0123456789")]).catch(() => {});
}
function refreshFonts(){ fontsReady().then(() => draw()); }
if(document.fonts){
  document.fonts.ready.then(() => draw());
  document.fonts.addEventListener && document.fonts.addEventListener("loadingdone", () => draw());
}

/* ================= EXIF ================= */
async function readExifDate(file){
  try{
    const buf = await file.slice(0, 256*1024).arrayBuffer(), v = new DataView(buf);
    if(v.byteLength < 4 || v.getUint16(0) !== 0xFFD8) return null;
    let o = 2;
    while(o + 4 < v.byteLength){
      const mk = v.getUint16(o); if((mk & 0xFF00) !== 0xFF00) return null;
      const len = v.getUint16(o+2);
      if(mk === 0xFFE1 && o + 10 < v.byteLength && v.getUint32(o+4) === 0x45786966) return parseTiff(v, o+10);
      if(mk === 0xFFDA) return null;
      o += 2 + len;
    }
  }catch(e){}
  return null;
}
function parseTiff(v, t){
  const le = v.getUint16(t) === 0x4949;
  const u16 = p => v.getUint16(p, le), u32 = p => v.getUint32(p, le);
  const readIFD = start => { const n = u16(start), tags = {}; for(let i=0;i<n;i++){ const e = start+2+i*12; tags[u16(e)] = {count:u32(e+4), off:e+8}; } return tags; };
  const str = tag => { if(!tag) return null; const p = tag.count > 4 ? t + u32(tag.off) : tag.off; let s = ""; for(let i=0;i<tag.count-1 && p+i<v.byteLength;i++) s += String.fromCharCode(v.getUint8(p+i)); return s; };
  const t0 = readIFD(t + u32(t+4)); let d = null;
  if(t0[0x8769]){ const ex = readIFD(t + u32(t0[0x8769].off)); d = str(ex[0x9003]) || str(ex[0x9004]); }
  if(!d) d = str(t0[0x0132]);
  const m = d && d.match(/^(\d{4}):(\d{2}):(\d{2})/);
  return m && m[1] !== "0000" ? `${m[1]}-${m[2]}-${m[3]}` : null;
}

/* ================= photos ================= */
let uid = 0;
function makeThumb(img){
  const n = 112, c = document.createElement("canvas"); c.width = c.height = n;
  const x = c.getContext("2d"), s = Math.max(n/img.naturalWidth, n/img.naturalHeight);
  x.drawImage(img, (n - img.naturalWidth*s)/2, (n - img.naturalHeight*s)/2, img.naturalWidth*s, img.naturalHeight*s);
  return c.toDataURL("image/jpeg", .8);
}
async function addFiles(list){
  const files = [...list].filter(f => f && f.type.startsWith("image/"));
  if(!files.length){ toast("请选择图片文件"); return; }
  let added = 0, failed = 0, first = -1;
  for(const file of files){
    const url = URL.createObjectURL(file), img = new Image(); img.src = url;
    try{ await img.decode(); }catch(e){ URL.revokeObjectURL(url); failed++; continue; }
    const ph = {
      id: ++uid, url, name: file.name || `photo-${uid}.jpg`, w: img.naturalWidth, h: img.naturalHeight,
      thumb: makeThumb(img), zoom:1, px:0, py:0, caption: state.perCaption ? state.caption : null,
      date: (await readExifDate(file)) || (file.lastModified ? isoOf(new Date(file.lastModified)) : null),
      seed: Math.floor(Math.random()*1e9), free:null, doodles:[]
    };
    photos.push(ph); imgCache.set(ph.id, img);
    while(imgCache.size > 3) imgCache.delete(imgCache.keys().next().value);
    if(first < 0) first = photos.length - 1;
    added++;
    if(added === 1){ cur = first; renderStrip(); syncUI(); schedule(); }
  }
  if(first >= 0) await select(first);
  renderStrip(); syncUI(); refreshFonts();
  if(files.length > 1 || failed) toast(`已添加 ${added} 张` + (failed ? `，${failed} 张无法读取（HEIC 请先转 JPG）` : ""));
}
async function select(i){
  if(i < 0 || i >= photos.length) return;
  cur = i; sel = null;
  try{ await getImg(photos[i]); }catch(e){}
  renderStrip(); syncUI(); draw();
  const el = $("#strip").children[i]; if(el && el.scrollIntoView) el.scrollIntoView({block:"nearest", inline:"nearest"});
}
function removePhoto(i){
  const ph = photos[i]; if(!ph) return;
  URL.revokeObjectURL(ph.url); imgCache.delete(ph.id); photos.splice(i,1);
  if(cur >= photos.length) cur = photos.length - 1;
  if(cur >= 0) select(cur); else { renderStrip(); syncUI(); schedule(); }
}
function renderStrip(){
  const s = $("#strip"); s.hidden = !photos.length;
  s.innerHTML = photos.map((p,i) =>
    `<button class="th${i===cur?" on":""}" data-i="${i}" title="${p.name.replace(/"/g,"&quot;")}"><img src="${p.thumb}" alt=""><span class="num">${i+1}</span><span class="x" data-rm="${i}">×</span></button>`
  ).join("") + `<button class="th add" id="stripAdd" title="添加照片">＋</button>`;
}
$("#strip").addEventListener("click", e => {
  const rm = e.target.closest("[data-rm]");
  if(rm){ e.stopPropagation(); removePhoto(+rm.dataset.rm); return; }
  if(e.target.closest("#stripAdd")){ $("#file").click(); return; }
  const th = e.target.closest("[data-i]"); if(th) select(+th.dataset.i);
});
$("#pick").onclick = () => $("#file").click();
$("#clearAll").onclick = () => { photos.forEach(p => URL.revokeObjectURL(p.url)); photos.length = 0; imgCache.clear(); cur = -1; sel = null; renderStrip(); syncUI(); schedule(); };
$("#file").addEventListener("change", e => { addFiles(e.target.files); e.target.value = ""; });
["dragenter","dragover"].forEach(t => stage.addEventListener(t, e => { e.preventDefault(); stage.classList.add("dragover"); }));
["dragleave","drop"].forEach(t => stage.addEventListener(t, e => { e.preventDefault(); stage.classList.remove("dragover"); }));
stage.addEventListener("drop", e => { if(e.dataTransfer.files && e.dataTransfer.files.length) addFiles(e.dataTransfer.files); });
window.addEventListener("paste", e => {
  if(/INPUT|TEXTAREA/.test(document.activeElement.tagName)) return;
  const items = (e.clipboardData && e.clipboardData.items) || [];
  const fs = []; for(const it of items) if(it.type.startsWith("image/")) fs.push(it.getAsFile());
  if(fs.length){ e.preventDefault(); addFiles(fs); }
});
/* batch tools */
$("#sortDate").onclick = () => {
  const keep = curPhoto();
  photos.sort((a,b) => (a.date || "9999").localeCompare(b.date || "9999") || a.name.localeCompare(b.name));
  cur = photos.indexOf(keep); renderStrip(); syncUI(); schedule(); toast("已按拍摄日期排序");
};
$("#nameCaption").onclick = () => {
  photos.forEach(p => { p.caption = p.name.replace(/\.[^.]+$/,"").replace(/[_]+/g," "); });
  state.perCaption = true; syncControls(); syncUI(); refreshFonts(); schedule(); toast("已用文件名作为每张的说明");
};
$("#pasteCaptions").onclick = () => {
  const ta = document.createElement("textarea"); ta.rows = 8; ta.style.width = "100%";
  ta.placeholder = "每行一句，按顺序对应每张照片\n第一张的说明\n第二张的说明\n…";
  ta.value = photos.map(p => captionFor(p).replace(/\n/g," / ")).join("\n");
  const wrap = document.createElement("div"); wrap.style.width = "100%"; wrap.appendChild(ta);
  const tip = document.createElement("p"); tip.textContent = `共 ${photos.length} 张照片。一行对应一张；想在一张里换行，用 “ / ” 分隔。`; wrap.appendChild(tip);
  openModal("批量粘贴说明", wrap, [{label:"取消"}, {label:"应用", cls:"primary", onClick: () => {
    const lines = ta.value.split(/\r?\n/);
    photos.forEach((p,i) => { if(lines[i] != null) p.caption = lines[i].split(/\s*\/\s*/).join("\n"); });
    state.perCaption = true; syncControls(); syncUI(); refreshFonts(); schedule(); toast("已更新每张的说明");
  }}]);
  setTimeout(() => ta.focus(), 50);
};

/* ================= pointer: pan / free drag / brush ================= */
const pointers = new Map();
let pinch = null, drag = null, stroke = null, suppressClick = false;
function setZoom(z){
  const ph = curPhoto(); if(!ph) return;
  ph.zoom = Math.min(4, Math.max(1, z));
  $("#zoom").value = ph.zoom; $("#zoomOut").textContent = ph.zoom.toFixed(2)+"×";
  schedule();
}
function toCard(e){
  const r = canvas.getBoundingClientRect();
  const W = parseFloat(canvas.style.width) || r.width, H = parseFloat(canvas.style.height) || r.height;
  let dx = e.clientX - (r.left + r.width/2), dy = e.clientY - (r.top + r.height/2);
  const a = state.withBg ? -state.tilt*Math.PI/180 : 0;
  if(a){ const c = Math.cos(a), s = Math.sin(a); [dx, dy] = [dx*c - dy*s, dx*s + dy*c]; }
  return {x: dx + W/2 - view.pad, y: dy + H/2 - view.pad};
}
const sameSel = x => !!sel && x.type === sel.type && (x.type !== "illo" || x.slot === sel.slot);
function freeRef(t){ const F = freeOf(curPhoto()); return t.type === "illo" ? F.illos[t.slot] : F[t.type]; }
function toLocal(e, p){ const dx = p.x - e.cx, dy = p.y - e.cy, c = Math.cos(-e.rot), s = Math.sin(-e.rot); return {x: dx*c - dy*s, y: dx*s + dy*c}; }
function hitFree(p, touch){
  if(!isFree() || tool !== "move") return null;
  const ph = curPhoto(), els = freeElements(geom(view.s, ph), ph, Math.max(0, cur));
  const R = touch ? 24 : 14, padHit = touch ? 12 : 5;
  if(sel){
    const e = els.find(sameSel);
    if(e){
      const l = toLocal(e, p), core = Math.abs(l.x) < e.w*.3 && Math.abs(l.y) < e.h*.3;
      if(!core && Math.hypot(l.x - (e.w/2 + 6), l.y - (e.h/2 + 6)) < R) return {e, handle:true};
      if(!core && e.type === "illo" && Math.hypot(l.x + (e.w/2 + 6), l.y + (e.h/2 + 6)) < R) return {e, del:true};
    }
  }
  for(let i = els.length - 1; i >= 0; i--){ const e = els[i], l = toLocal(e, p); if(Math.abs(l.x) <= e.w/2 + padHit && Math.abs(l.y) <= e.h/2 + padHit) return {e, handle:false}; }
  return null;
}
/* in a preset layout, grabbing a sticker switches to free layout without moving anything */
function hitPresetSticker(p, touch){
  if(isFree() || tool !== "move") return null;
  const ph = curPhoto(), g = geom(view.s, ph), lay = illoLayout(g, ph, Math.max(0, cur));
  for(let i = lay.items.length - 1; i >= 0; i--){
    const it = lay.items[i], l = toLocal({cx: it.cx, cy: it.cy, rot: it.rot}, p), half = it.size*.45 + (touch ? 10 : 2);
    if(Math.abs(l.x) <= half && Math.abs(l.y) <= half) return it;
  }
  return null;
}
let freeHinted = false;
function enterFreeFromPreset(){
  const prev = state.illoMode;
  initFree(prev, freeOf(curPhoto()));
  state.lastPreset = prev; state.illoMode = "free";
  syncControls(); syncUI();
  if(!freeHinted){ freeHinted = true; toast("已切到自由摆放：拖动移动，双指缩放旋转"); }
}
let elPinch = null;
const touchPts = new Map();
canvas.addEventListener("pointerdown", e => {
  suppressClick = false;
  const p = toCard(e), touch = e.pointerType !== "mouse";
  touchPts.set(e.pointerId, p);
  if(tool === "brush"){
    if(stroke) return;
    const g = geom(view.s, curPhoto());
    stroke = {id:e.pointerId, s:{c: state.brush.color, w: state.brush.width, pts:[r4(p.x/g.W), r4(p.y/g.H)]}, last:p, g};
    docOf(curPhoto()).doodles.push(stroke.s);
    canvas.setPointerCapture(e.pointerId); suppressClick = true; schedule(); return;
  }
  // second finger while dragging an element → pinch-scale / rotate that element
  if(drag && e.pointerId !== drag.id && touchPts.has(drag.id)){
    const a = touchPts.get(drag.id), r = freeRef(sel);
    if(r){
      elPinch = {ids:[drag.id, e.pointerId], d0: Math.max(1, Math.hypot(p.x - a.x, p.y - a.y)), a0: Math.atan2(p.y - a.y, p.x - a.x),
        m0: {x:(a.x + p.x)/2, y:(a.y + p.y)/2}, ref: Object.assign({}, r), g: drag.g};
      drag = null; canvas.setPointerCapture(e.pointerId); suppressClick = true; return;
    }
  }
  if(!isFree() && hitPresetSticker(p, touch)) enterFreeFromPreset();
  const hit = hitFree(p, touch);
  if(hit && hit.del){ removeSelIllo(); suppressClick = true; return; }
  if(hit){
    sel = {type: hit.e.type, slot: hit.e.slot};
    drag = {id: e.pointerId, handle: hit.handle, start: p, ref: Object.assign({}, freeRef(sel)), c: {x: hit.e.cx, y: hit.e.cy}, g: geom(view.s, curPhoto())};
    canvas.setPointerCapture(e.pointerId); canvas.classList.add("dragging");
    suppressClick = true; syncSel(); syncThemes(); schedule(); return;
  }
  if(isFree() && sel){ sel = null; syncSel(); syncThemes(); schedule(); }
  if(!curPhoto()) return;
  canvas.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, p);
  canvas.classList.add("dragging");
  if(pointers.size === 2){ const [a,b] = [...pointers.values()]; pinch = {d: Math.hypot(a.x-b.x, a.y-b.y), z: curPhoto().zoom}; }
});
canvas.addEventListener("pointermove", e => {
  if(touchPts.has(e.pointerId)) touchPts.set(e.pointerId, toCard(e));
  if(elPinch && elPinch.ids.includes(e.pointerId)){
    const [a, b] = elPinch.ids.map(id => touchPts.get(id)), r = freeRef(sel);
    if(!a || !b || !r) return;
    const d = Math.hypot(b.x - a.x, b.y - a.y), ang = Math.atan2(b.y - a.y, b.x - a.x), m = {x:(a.x + b.x)/2, y:(a.y + b.y)/2};
    r.scale = r4(Math.min(8, Math.max(.15, elPinch.ref.scale * d / elPinch.d0)));
    let rot = elPinch.ref.rot + ang - elPinch.a0;
    const deg = rot*180/Math.PI, near = Math.round(deg/90)*90;
    if(Math.abs(deg - near) < 3) rot = near*Math.PI/180;
    r.rot = r4(rot);
    r.cx = r4(Math.min(1, Math.max(0, elPinch.ref.cx + (m.x - elPinch.m0.x)/elPinch.g.W)));
    r.cy = r4(Math.min(1, Math.max(0, elPinch.ref.cy + (m.y - elPinch.m0.y)/elPinch.g.H)));
    schedule(); return;
  }
  if(stroke && e.pointerId === stroke.id){
    const p = toCard(e);
    if(Math.hypot(p.x - stroke.last.x, p.y - stroke.last.y) < 1.5) return;
    stroke.last = p; stroke.s.pts.push(r4(p.x/stroke.g.W), r4(p.y/stroke.g.H)); schedule(); return;
  }
  if(drag && e.pointerId === drag.id){
    const p = toCard(e), r = freeRef(sel); if(!r) return;
    if(drag.handle){
      const d0 = Math.hypot(drag.start.x - drag.c.x, drag.start.y - drag.c.y), d1 = Math.hypot(p.x - drag.c.x, p.y - drag.c.y);
      r.scale = r4(Math.min(8, Math.max(.15, drag.ref.scale * d1 / Math.max(1, d0))));
      let rot = drag.ref.rot + Math.atan2(p.y - drag.c.y, p.x - drag.c.x) - Math.atan2(drag.start.y - drag.c.y, drag.start.x - drag.c.x);
      const deg = rot*180/Math.PI, near = Math.round(deg/90)*90;
      if(Math.abs(deg - near) < 3) rot = near*Math.PI/180;
      r.rot = r4(rot);
    } else {
      r.cx = r4(Math.min(1, Math.max(0, drag.ref.cx + (p.x - drag.start.x)/drag.g.W)));
      r.cy = r4(Math.min(1, Math.max(0, drag.ref.cy + (p.y - drag.start.y)/drag.g.H)));
    }
    schedule(); return;
  }
  if(!pointers.has(e.pointerId)){
    if(tool === "move" && e.pointerType === "mouse"){
      const q = toCard(e), h = isFree() ? hitFree(q) : (hitPresetSticker(q) ? {} : null);
      canvas.style.cursor = h ? (h.handle ? "nwse-resize" : h.del ? "pointer" : "move") : "";
    }
    return;
  }
  const ph = curPhoto(); if(!ph) return;
  const prev = pointers.get(e.pointerId), now = toCard(e);
  pointers.set(e.pointerId, now);
  if(pointers.size >= 2 && pinch){ const [a,b] = [...pointers.values()]; setZoom(pinch.z * Math.hypot(a.x-b.x, a.y-b.y) / pinch.d); return; }
  const pr = photoRect(geom(view.s, ph), ph);
  if(pr.ox > .5) ph.px = r4(Math.max(-1, Math.min(1, ph.px + (now.x-prev.x)/pr.ox)));
  if(pr.oy > .5) ph.py = r4(Math.max(-1, Math.min(1, ph.py + (now.y-prev.y)/pr.oy)));
  schedule();
});
function endPointer(e){
  touchPts.delete(e.pointerId);
  if(elPinch && elPinch.ids.includes(e.pointerId)) elPinch = null;
  if(stroke && e.pointerId === stroke.id){ stroke = null; schedule(); syncUI(); }
  if(drag && e.pointerId === drag.id) drag = null;
  pointers.delete(e.pointerId); if(pointers.size < 2) pinch = null;
  if(!pointers.size && !drag) canvas.classList.remove("dragging");
}
canvas.addEventListener("pointerup", endPointer);
canvas.addEventListener("pointercancel", endPointer);
canvas.addEventListener("wheel", e => {
  const h = hitFree(toCard(e));
  if(h){ e.preventDefault(); const r = freeRef(h.e); r.scale = r4(Math.min(8, Math.max(.15, r.scale*Math.exp(-e.deltaY*.0015)))); sel = {type:h.e.type, slot:h.e.slot}; syncSel(); schedule(); return; }
  if(!curPhoto() || tool !== "move") return; e.preventDefault(); setZoom(curPhoto().zoom * Math.exp(-e.deltaY*.0015));
}, {passive:false});
canvas.addEventListener("dblclick", e => {
  if(tool !== "move") return;
  const h = hitFree(toCard(e)); if(h){ freeRef(h.e).rot = 0; schedule(); return; }
  const ph = curPhoto(); if(!ph) return; ph.px = ph.py = 0; setZoom(1);
});
canvas.addEventListener("click", () => { if(suppressClick){ suppressClick = false; return; } if(!photos.length && tool === "move" && !(isFree() && sel)) $("#file").click(); });
$("#zoom").addEventListener("input", e => setZoom(parseFloat(e.target.value)));

function setTool(t){ tool = t; sel = null; canvas.style.cursor = ""; syncUI(); draw(); }
$$("[data-tool]").forEach(b => b.onclick = () => setTool(b.dataset.tool));
$("#brushToggle").onclick = () => setTool(tool === "brush" ? "move" : "brush");
$("#doodleClear").onclick = () => { const d = docOf(curPhoto()); if(!d.doodles.length) return; d.doodles = []; syncUI(); schedule(); toast("已清除（可用 ⌘Z 撤销）"); };

window.addEventListener("keydown", e => {
  const inField = /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName);
  const mod = e.metaKey || e.ctrlKey;
  if(mod && !inField && (e.key === "z" || e.key === "Z")){ e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
  if(mod && !inField && (e.key === "y" || e.key === "Y")){ e.preventDefault(); redo(); return; }
  if(inField || mod) return;
  if(e.key === "Escape" && !$("#modal").hidden){ closeModal(); return; }
  if(e.key === "b" || e.key === "B"){ setTool("brush"); return; }
  if(e.key === "s" || e.key === "S"){ setTray(tray.hidden); return; }
  if(e.key === "v" || e.key === "V"){ setTool("move"); return; }
  if(isFree() && sel){
    const r = freeRef(sel), st = e.shiftKey ? .02 : .004;
    if(e.key === "Delete" || e.key === "Backspace"){ e.preventDefault(); removeSelIllo(); return; }
    if(e.key === "Escape"){ sel = null; syncSel(); syncThemes(); schedule(); return; }
    const mv = {ArrowLeft:[-st,0], ArrowRight:[st,0], ArrowUp:[0,-st], ArrowDown:[0,st]}[e.key];
    if(mv && r){ e.preventDefault(); r.cx = r4(Math.min(1, Math.max(0, r.cx + mv[0]))); r.cy = r4(Math.min(1, Math.max(0, r.cy + mv[1]))); schedule(); return; }
  }
  if(e.key === "ArrowRight" && cur < photos.length-1) select(cur+1);
  if(e.key === "ArrowLeft" && cur > 0) select(cur-1);
});
function removeSelIllo(){
  if(!sel || sel.type !== "illo") return;
  if(state.illoPer){ toast("开启「每张照片随机搭配」时不能单独移除"); return; }
  state.illos.splice(sel.slot, 1);
  [state.free, ...photos.map(p => p.free).filter(Boolean)].forEach(F => F.illos.splice(sel.slot, 1));
  sel = null; syncUI(); schedule();
}
function syncSel(){
  let t = "点击相纸上的插画或文字来选中它。";
  if(sel){
    if(sel.type === "illo"){ const id = illosFor(curPhoto(), Math.max(0,cur))[sel.slot]; const il = id && getIllo(id); t = `已选中：插画 ${sel.slot+1} · ${il ? il.n : ""}`; }
    else t = sel.type === "cap" ? "已选中：说明文字（拖圆点可缩放、旋转）" : "已选中：日期 / 编号";
  }
  $("#selLine").textContent = t;
  $("#freeDelete").hidden = !(sel && sel.type === "illo");
}
$("#freeDelete").onclick = removeSelIllo;
$("#freeReset").onclick = () => { sel = null; initFree(state.lastPreset, freeOf(curPhoto())); syncControls(); syncUI(); schedule(); };
$("#freeSticker").addEventListener("change", e => { freeOf(curPhoto()).sticker = e.target.checked; schedule(); });
$("#freeOwn").addEventListener("change", e => {
  const ph = curPhoto(); if(!ph){ e.target.checked = false; return; }
  ph.free = e.target.checked ? clone(state.free) : null; sel = null; syncControls(); syncUI(); schedule();
  toast(e.target.checked ? `第 ${cur+1} 张现在有自己的排版` : "这一张恢复使用共用排版");
});

/* ================= controls ================= */
RATIOS.forEach(([v,label]) => {
  const b = document.createElement("button");
  b.className = "dr-chip"; b.dataset.group = "ratio"; b.dataset.value = v;
  if(v === "orig") b.textContent = label;
  else { const [a,c] = v.split(":").map(Number), m = 13/Math.max(a,c);
    b.innerHTML = `<span class="ri" style="width:${Math.round(a*m)}px;height:${Math.round(c*m)}px"></span>${v}`; }
  $("#ratios").appendChild(b);
});
function swatches(box, map, group, key, small){
  Object.entries(map).forEach(([k,p]) => {
    const b = document.createElement("button");
    b.className = "sw" + (small ? " sm" : ""); b.dataset.group = group; b.dataset.value = k; b.title = p.name;
    b.style.setProperty("--c", p[key]); box.appendChild(b);
  });
  const l = document.createElement("span"); l.className = "sw-label"; box.appendChild(l); return l;
}
const paperLabel = swatches($("#papers"), PAPERS, "paper", "bg");
const bgLabel = swatches($("#bgs"), BGS, "bg", "c");
const decoLabel = swatches($("#decoColors"), DECO_COLORS, "decoColor", "c", true);
const brushLabel = swatches($("#brushColors"), BRUSH_COLORS, "brush.color", "c", true);
const collageBgLabel = swatches($("#collageBgs"), BGS, "collage.bg", "c", true);
Object.entries(TEXT_COLORS).forEach(([k,v]) => {
  const b = document.createElement("button"); b.className = "dr-chip"; b.dataset.group = "textColor"; b.dataset.value = k;
  b.innerHTML = k === "auto" ? "自动" : `<span class="dot" style="--c:${v.c}"></span>${v.name}`;
  $("#textColors").appendChild(b);
});
const colorPick = document.createElement("input"); colorPick.type = "color"; colorPick.title = "自定义颜色"; colorPick.value = "#C6613F";
colorPick.addEventListener("input", () => { state.textColor = colorPick.value; syncUI(); schedule(); });
$("#textColors").appendChild(colorPick);
Object.entries(THEMES).forEach(([k,v]) => {
  const b = document.createElement("button"); b.className = "dr-chip"; b.dataset.theme = k; b.textContent = v.name; $("#themeRow").appendChild(b);
});
const inheritBtn = document.createElement("button"); inheritBtn.className = "dr-chip"; inheritBtn.dataset.theme = "inherit"; inheritBtn.textContent = "跟随全部"; $("#themeRow").appendChild(inheritBtn);
$("#themeRow").addEventListener("click", e => {
  const b = e.target.closest("[data-theme]"); if(!b) return;
  const v = b.dataset.theme;
  if(isFree() && sel && sel.type === "illo"){ const r = freeRef(sel); if(r){ if(v === "inherit") delete r.theme; else r.theme = v; } }
  else if(v !== "inherit") state.illoTheme = v;
  syncThemes(); schedule();
});
function syncThemes(){
  const target = isFree() && sel && sel.type === "illo" ? freeRef(sel) : null;
  const curT = target ? (target.theme || "inherit") : state.illoTheme;
  $$("#themeRow [data-theme]").forEach(b => b.classList.toggle("on", b.dataset.theme === curT));
  inheritBtn.hidden = !target;
  $("#themeLabel").textContent = target ? `插画 ${sel.slot+1} 配色` : "配色";
}

/* illustration grid */
const grid = $("#illoGrid");
function addGridItem(id){
  const it = getIllo(id), b = document.createElement("button");
  b.className = "il"; b.dataset.illo = id; b.title = it.n;
  if(customs[id]) b.innerHTML = `<img src="${customs[id].url}" alt="">`; else b.innerHTML = illoSVG(id);
  grid.insertBefore(b, $("#uploadTile"));
}
const upTile = document.createElement("button"); upTile.className = "il upload"; upTile.id = "uploadTile"; upTile.title = "上传 PNG / SVG 贴纸"; upTile.textContent = "＋";
grid.appendChild(upTile);
ILLOS.forEach(il => addGridItem(il.id));
upTile.onclick = () => $("#stickerFile").click();
$("#stickerUpload").onclick = () => $("#stickerFile").click();
let cuid = 0;
$("#stickerFile").addEventListener("change", async e => {
  const files = [...e.target.files]; e.target.value = "";
  for(const f of files){
    const url = URL.createObjectURL(f), img = new Image(); img.src = url;
    try{ await img.decode(); }catch(err){ URL.revokeObjectURL(url); toast(`${f.name} 无法读取`); continue; }
    const id = `my${++cuid}`;
    customs[id] = {id, n: f.name.replace(/\.[^.]+$/,"").slice(0,16) || "我的贴纸", img, url, w: img.naturalWidth || 300, h: img.naturalHeight || 300, cat:"mine"};
    addGridItem(id); addTrayItem(id);
    if(!state.illos.includes(id)){ state.illos.push(id); if(state.illos.length > MAX_ILLOS){ state.illos.shift(); state.free.illos.shift(); } }
  }
  illoCat = "mine"; syncUI(); schedule();
  if(files.length) toast("已加入「我的」，并放到相纸上");
});
$("#illoTabs").addEventListener("click", e => { const t = e.target.closest("[data-cat]"); if(t){ illoCat = t.dataset.cat; syncUI(); } });
const allFrees = () => [state.free, ...photos.map(p => p.free).filter(Boolean)];
function toggleIllo(id, fromTray){
  const i = state.illos.indexOf(id);
  if(i >= 0){ state.illos.splice(i,1); allFrees().forEach(F => F.illos.splice(i,1)); sel = null; syncUI(); schedule(); return; }
  if(fromTray && !isFree()) enterFreeFromPreset();
  state.illos.push(id);
  if(state.illos.length > MAX_ILLOS){ state.illos.shift(); allFrees().forEach(F => F.illos.shift()); }
  sel = null;
  if(fromTray && isFree() && !state.illoPer){
    const slot = state.illos.length - 1, F = freeOf(curPhoto()), g = geom(1, curPhoto());
    const n = slot % 5;
    F.illos[slot] = {cx: r4(.5 + (n - 2)*.06), cy: r4((g.y0 + g.ih*(.42 + (n % 2)*.08))/g.H), scale: 1.1, rot: r4(((slot*37)%11 - 5)*.035)};
    sel = {type:"illo", slot};
  }
  syncUI(); schedule();
}
grid.addEventListener("click", e => { const b = e.target.closest("[data-illo]"); if(b) toggleIllo(b.dataset.illo, false); });

/* sticker tray on the stage (thumb-friendly on phones) */
const tray = $("#tray"), trayGrid = $("#trayGrid");
function addTrayItem(id){
  const it = getIllo(id), b = document.createElement("button");
  b.className = "il"; b.dataset.tray = id; b.title = it.n;
  b.innerHTML = customs[id] ? `<img src="${customs[id].url}" alt="">` : illoSVG(id);
  trayGrid.appendChild(b);
}
ILLOS.forEach(il => addTrayItem(il.id));
function setTray(open){
  tray.hidden = !open; stage.classList.toggle("tray-open", open);
  if(open && tool !== "move") setTool("move");
  syncUI(); draw();
}
$("#trayBtn").onclick = () => setTray(tray.hidden);
$("#trayClose").onclick = () => setTray(false);
trayGrid.addEventListener("click", e => { const b = e.target.closest("[data-tray]"); if(b) toggleIllo(b.dataset.tray, true); });
$("#trayTabs").addEventListener("click", e => { const t = e.target.closest("[data-cat]"); if(t){ illoCat = t.dataset.cat; syncUI(); trayGrid.scrollLeft = 0; } });
$("#illoRandom").onclick = () => {
  const pool = ILLOS.filter(i => !["all","picked","mine"].includes(illoCat) ? i.cat === illoCat : true).map(i => i.id);
  const n = Math.min(pool.length, isFree() ? 3 + Math.floor(Math.random()*3) : 2 + Math.floor(Math.random()*2)), out = [];
  while(out.length < n){ out.push(pool.splice(Math.floor(Math.random()*pool.length),1)[0]); }
  state.illos = out; state.illoSeed = Math.floor(Math.random()*1e6); sel = null;
  syncUI(); schedule();
};
$("#illoShuffle").onclick = () => { state.illoSeed = Math.floor(Math.random()*1e6); schedule(); };
$("#illoClear").onclick = () => { state.illos = []; [state.free, ...photos.map(p => p.free).filter(Boolean)].forEach(F => F.illos = []); sel = null; syncUI(); schedule(); };
$("#decoShuffle").onclick = () => { state.decoSeed = Math.floor(Math.random()*1e6); schedule(); };

document.addEventListener("click", e => {
  const el = e.target.closest("[data-group]");
  if(!el) return;
  const g = el.dataset.group, prev = getPath(g);
  setPath(g, el.dataset.value);
  if(g === "illoMode"){
    if(state.illoMode === "free"){ if(!state.free.inited) initFree(prev, state.free); }
    else { state.lastPreset = state.illoMode; sel = null; }
  }
  if(g === "font" || g === "dateFmt") refreshFonts();
  syncControls(); syncUI(); schedule();
});

const syncers = [];
function bindRange(id, path, fmt){
  const el = $("#"+id), out = $("#"+id+"Out");
  el.addEventListener("input", () => { setPath(path, parseFloat(el.value)); out.textContent = fmt(getPath(path)); schedule(); });
  syncers.push(() => { el.value = getPath(path); out.textContent = fmt(getPath(path)); });
}
bindRange("border","border", v => Math.round(v*100)+"%");
bindRange("textSize","textSize", v => Math.round(v*100)+"%");
bindRange("illoSize","illoSize", v => Math.round(v*100)+"%");
bindRange("tilt","tilt", v => (v>0?"+":"")+v.toFixed(1)+"°");
bindRange("brushWidth","brush.width", v => v.toFixed(1)+"×");
function bindText(id, path){
  const el = $("#"+id);
  el.addEventListener("input", () => { setPath(path, el.value); refreshFonts(); schedule(); });
  syncers.push(() => { if(document.activeElement !== el) el.value = getPath(path); });
}
bindText("note","note"); bindText("date","date");
function bindCheck(id, path, after){
  const el = $("#"+id);
  el.addEventListener("change", () => { setPath(path, el.checked); if(after) after(); syncUI(); schedule(); });
  syncers.push(() => { el.checked = !!getPath(path); });
}
["grain","accent","withBg","autoNum","illoPer","stamp","leak","vignette","fadeEdge"].forEach(k => bindCheck(k, k));
bindCheck("printMarks", "print.marks"); bindCheck("printFill", "print.fill");
bindCheck("perCaption", "perCaption", () => { if(state.perCaption) photos.forEach(p => { if(p.caption == null) p.caption = state.caption; }); });
syncers.push(() => { const ph = curPhoto(); $("#freeSticker").checked = !!freeOf(ph).sticker; $("#freeOwn").checked = !!(ph && ph.free); $("#freeOwn").disabled = !ph; });
const capEl = $("#caption");
capEl.addEventListener("input", () => {
  const ph = curPhoto();
  if(state.perCaption && ph) ph.caption = capEl.value; else state.caption = capEl.value;
  refreshFonts(); schedule();
});
function syncControls(){ syncers.forEach(f => f()); if(document.activeElement !== capEl) capEl.value = captionFor(curPhoto()); }

/* ================= templates ================= */
const TPL_KEY = "instant-frame-templates-v1";
let userTpls = [];
try{ userTpls = JSON.parse(localStorage.getItem(TPL_KEY) || "[]"); if(!Array.isArray(userTpls)) userTpls = []; }catch(e){ userTpls = []; }
function saveTpls(){ try{ localStorage.setItem(TPL_KEY, JSON.stringify(userTpls)); return true; }catch(e){ return false; } }
function tplData(){ const d = clone(state); delete d.date; d.illos = d.illos.filter(id => ILLO_BY_ID[id]); return d; }
function applyTpl(data){
  const keep = {date: state.date, caption: state.caption, note: state.note, perCaption: state.perCaption, dateSource: state.dateSource};
  replaceState(Object.assign(clone(DEFAULTS), clone(data), keep));
  state.illos = state.illos.filter(getIllo);
  photos.forEach(p => { p.free = null; });
  if(state.illoMode === "free" && !state.free.inited) initFree(state.lastPreset, state.free);
  sel = null; syncControls(); syncUI(); refreshFonts(); schedule();
}
function renderTpls(){
  const box = $("#tpls"); box.innerHTML = "";
  BUILTIN_TPLS.forEach((t,i) => { const b = document.createElement("button"); b.className = "tpl"; b.dataset.b = i; b.innerHTML = `<i style="--c:${t.c}"></i>${t.name}`; box.appendChild(b); });
  userTpls.forEach((t,i) => {
    const b = document.createElement("button"); b.className = "tpl mine"; b.dataset.u = i;
    b.innerHTML = `<i style="--c:${(PAPERS[t.data.paper] || PAPERS.ivory).bg}"></i>${t.name.replace(/</g,"&lt;")}<span class="del" data-del="${i}" title="删除">×</span>`;
    box.appendChild(b);
  });
  $("#tplNote").textContent = userTpls.length ? `我的 ${userTpls.length} 个` : "";
}
$("#tpls").addEventListener("click", e => {
  const del = e.target.closest("[data-del]");
  if(del){ e.stopPropagation(); const i = +del.dataset.del; const n = userTpls[i].name; userTpls.splice(i,1); saveTpls(); renderTpls(); toast(`已删除「${n}」`); return; }
  const b = e.target.closest(".tpl"); if(!b) return;
  if(b.dataset.b != null){ const t = BUILTIN_TPLS[+b.dataset.b]; applyTpl(t.data); toast(`已套用「${t.name}」（⌘Z 可撤销）`); }
  else { const t = userTpls[+b.dataset.u]; applyTpl(t.data); toast(`已套用「${t.name}」（⌘Z 可撤销）`); }
});
$("#tplSave").onclick = () => { $("#tplSaveRow").hidden = false; $("#tplName").value = `我的模板 ${userTpls.length + 1}`; $("#tplName").focus(); $("#tplName").select(); };
function doSaveTpl(){
  const name = $("#tplName").value.trim() || `我的模板 ${userTpls.length + 1}`;
  userTpls.push({name, data: tplData()});
  const ok = saveTpls(); renderTpls(); $("#tplSaveRow").hidden = true;
  toast(ok ? `已保存「${name}」` : `已保存「${name}」（浏览器不允许长期保存，建议再导出 .json 备份）`);
}
$("#tplSaveOk").onclick = doSaveTpl;
$("#tplName").addEventListener("keydown", e => { if(e.key === "Enter") doSaveTpl(); if(e.key === "Escape") $("#tplSaveRow").hidden = true; });
$("#tplExport").onclick = async () => {
  const payload = JSON.stringify({app:"instant-frame", version:1, templates:[{name:"当前设置", data: tplData()}, ...userTpls]}, null, 2);
  await offerSave("instant-frame-templates.json", payload, () => {
    const ta = document.createElement("textarea"); ta.value = payload; ta.rows = 12; ta.style.width = "100%"; ta.readOnly = true;
    const w = document.createElement("div"); w.style.width = "100%"; w.appendChild(ta);
    const p = document.createElement("p"); p.textContent = "当前环境不能直接下载，复制上面的内容保存为 .json 文件即可。"; w.appendChild(p);
    openModal("导出模板", w, [{label:"复制", cls:"primary", onClick: async () => { try{ await navigator.clipboard.writeText(payload); toast("已复制"); }catch(e){ ta.select(); } return true; }}, {label:"关闭"}]);
  });
};
$("#tplImport").onclick = () => $("#tplFile").click();
$("#tplFile").addEventListener("change", async e => {
  const f = e.target.files[0]; e.target.value = ""; if(!f) return;
  try{
    const o = JSON.parse(await f.text());
    const list = Array.isArray(o) ? o : Array.isArray(o.templates) ? o.templates : (o.data ? [o] : []);
    const good = list.filter(t => t && typeof t === "object" && t.data && typeof t.data === "object").map(t => ({name: String(t.name || "导入的模板").slice(0,20), data: t.data}));
    if(!good.length) throw new Error("文件里没有找到模板");
    userTpls.push(...good); saveTpls(); renderTpls(); toast(`已导入 ${good.length} 个模板`);
  }catch(err){ toast("导入失败：" + (err.message || err)); }
});

/* ================= sync UI ================= */
function syncUI(){
  const ph = curPhoto(), free = isFree(), even = !STYLES[state.style].caption;
  $$("[data-group]").forEach(el => el.classList.toggle("on", String(getPath(el.dataset.group)) === el.dataset.value));
  $$("[data-tool]").forEach(b => b.classList.toggle("on", b.dataset.tool === tool));
  canvas.classList.toggle("brush", tool === "brush");
  $("#brushToggle").textContent = tool === "brush" ? "退出画笔" : "进入画笔";
  $("#ratioOut").textContent = state.ratio === "orig" ? (ph ? `原图 ${ratioValue(ph).toFixed(2)}:1` : "原图") : state.ratio;
  paperLabel.textContent = PAPERS[state.paper].name; bgLabel.textContent = BGS[state.bg].name;
  decoLabel.textContent = DECO_COLORS[state.decoColor].name; brushLabel.textContent = BRUSH_COLORS[state.brush.color].name;
  collageBgLabel.textContent = BGS[state.collage.bg].name;
  $("#decoNote").textContent = state.deco === "none" ? "" : $(`[data-group="deco"][data-value="${state.deco}"]`).textContent;
  // stage hint
  $("#hint").innerHTML = tool === "brush" ? "<b>✎</b> 画笔模式：直接在相纸上画 · 按 V 回到移动"
    : free ? "拖动插画 / 文字 · 拖圆点缩放旋转 · 空白处拖动照片"
    : photos.length ? "拖动调整构图 · 滚轮 / 双指缩放 · 双击复位" : "<b>＋</b> 点击相纸、拖入或粘贴照片（可多选）";
  canvas.classList.toggle("empty", !photos.length && tool === "move");
  // photos
  $("#imgInfo").textContent = ph ? `${cur+1} / ${photos.length} · ${ph.w}×${ph.h}` : "";
  $("#clearAll").hidden = !photos.length;
  $("#batchTools").hidden = photos.length < 2;
  $("#zoom").value = ph ? ph.zoom : 1; $("#zoomOut").textContent = (ph ? ph.zoom : 1).toFixed(2)+"×";
  $("#zoomRow").classList.toggle("dim-off", !ph);
  // illos
  $$("#illoTabs [data-cat]").forEach(t => t.classList.toggle("on", t.dataset.cat === illoCat));
  $$("#illoGrid [data-illo]").forEach(b => {
    const id = b.dataset.illo, i = state.illos.indexOf(id), cat = getIllo(id).cat;
    b.hidden = !(illoCat === "all" || (illoCat === "picked" ? i >= 0 : cat === illoCat));
    b.classList.toggle("on", i >= 0);
    let badge = b.querySelector(".badge");
    if(i >= 0){ if(!badge){ badge = document.createElement("span"); badge.className = "badge"; b.appendChild(badge); } badge.textContent = i+1; }
    else if(badge) badge.remove();
  });
  upTile.hidden = !(illoCat === "mine" || illoCat === "all");
  $$("#trayTabs [data-cat]").forEach(t => t.classList.toggle("on", t.dataset.cat === illoCat));
  $$("#trayGrid [data-tray]").forEach(b => {
    const id = b.dataset.tray, i = state.illos.indexOf(id), cat = getIllo(id).cat;
    b.hidden = !(illoCat === "all" || (illoCat === "picked" ? i >= 0 : cat === illoCat));
    b.classList.toggle("on", i >= 0);
  });
  $("#trayBtn").classList.toggle("on", !tray.hidden);
  $("#illoCount").textContent = state.illoPer ? "每张随机" : `${state.illos.length} / ${MAX_ILLOS}`;
  $$('#illoModes [data-value="right"],#illoModes [data-value="sides"]').forEach(c => c.classList.toggle("dim-off", even));
  $("#freeBox").hidden = !free;
  $("#illoSizeRow").classList.toggle("dim-off", free);
  $("#illoShuffle").classList.toggle("dim-off", free && !state.illoPer);
  const n = free ? MAX_ILLOS : MAX_PRESET;
  $("#illoNote").textContent = state.illoPer ? `每张照片随机挑 ${Math.min(state.illos.length || 2, n)} 个插画（上面选几个就挑几个）。`
    : free ? "自由摆放最多用 8 个插画；新加的插画会先出现在底边右侧。"
    : even && state.illoMode !== "corner" ? "等边样式没有底边留白，插画会以贴纸形式压在照片角上。"
    : state.illos.length > MAX_PRESET ? `预设排版只用前 4 个，其余 ${state.illos.length - MAX_PRESET} 个在「自由摆放」里才会出现。`
    : "最多选 8 个；预设排版用前 4 个，点选的顺序就是排列顺序。";
  syncSel(); syncThemes();
  // text
  $("#textBody").classList.toggle("dim-off", even && !free);
  $("#alignRow").classList.toggle("dim-off", free);
  $("#textNote").textContent = even && !free ? "等边样式不显示文字" : (state.perCaption && ph ? `第 ${cur+1} 张` : "");
  if(document.activeElement !== capEl) capEl.value = captionFor(ph);
  capEl.placeholder = state.perCaption ? "为这一张写一句话…（回车换行）" : "写一句话…（回车换行，最多 3 行）";
  const custom = !TEXT_COLORS[state.textColor];
  colorPick.style.outline = custom ? "2px solid var(--ink)" : "none"; colorPick.style.outlineOffset = "2px";
  if(custom) colorPick.value = state.textColor;
  $("#dateField").classList.toggle("dim-off", state.dateSource === "exif");
  $("#dateField label").textContent = state.dateSource === "exif" ? (ph ? `拍摄日期：${ph.date || "未知"}` : "拍摄日期（读取照片信息）") : "自定义日期";
  // doodles / film
  const dn = docOf(ph).doodles.length; $("#doodleNote").textContent = dn ? `这张 ${dn} 笔` : "";
  const film = [state.stamp && "印字", state.leak && "漏光", state.vignette && "暗角", state.fadeEdge && "褪色"].filter(Boolean);
  $("#filmNote").textContent = film.join(" · ");
  // export
  $("#bgOpts").classList.toggle("dim-off", !state.withBg);
  $("#export").textContent = photos.length > 1 ? "导出当前这张" : "导出图片";
  $("#exportAll").hidden = photos.length < 2; $("#exportAll").textContent = `导出全部 ${photos.length} 张（ZIP）`;
  $("#batchNote").hidden = photos.length < 2;
  $("#collageNote").textContent = photos.length ? `${Math.min(12, photos.length)} 张` : "";
  const ps = PRINT_SIZES[state.print.size];
  $("#printNote").textContent = `${ps.n}`;
  $("#printHint").textContent = `卡片宽度按实物 ${ps.w}mm 打印，高度随你的设计。想更接近实物比例，画幅可选 ${ps.hint}。打印时选「实际大小 / 100%」，不要缩放。`;
}

/* ================= toast ================= */
let toastT = 0;
function toast(msg){
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove("show"), 2600);
}

/* ================= host page (personal site header) ================= */
const siteHeader = document.getElementById("site-header");
if(siteHeader){
  const setH = () => document.documentElement.style.setProperty("--dr-head", siteHeader.offsetHeight + "px");
  setH(); window.addEventListener("resize", setH);
}

/* ================= boot ================= */
renderTpls();
syncControls();
syncUI();
refreshFonts();
draw();
commit();
})();
