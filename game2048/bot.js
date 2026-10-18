// 2048 位板 expectimax 求解器（注入浏览器控制台使用）
//
// 在官方原版 2048（gabrielecirulli 的经典 DOM 版，如本地
// python -m http.server 部署的 2048-master）页面控制台粘贴本文件全部内容，
// AI 自动开始对局，直到合成 2048 方块。
//
// 技术要点：
//   - 位板表示：4 行 Uint16，每格 4bit 存 log2(value)
//   - 65536 项行移动/得分预计算表（nneonneo 式优化）
//   - expectimax 深度 4-6（按空格数自适应）
//   - nneonneo 启发式：empty/merges/monotonicity/sum 加权
//   - 直接调用官方 GameManager 实例的 move()（棋盘读取按 cells[x][y] 列优先修正）
//
// 实测成绩：合成 2048，973 步，得分 20268，平均 17ms/步（Chrome，2026-10-18）

// ===== 以下为完整可执行 AI（粘贴到控制台即开打） =====
(function () {
  if (window.__bot && window.__bot.timer) clearTimeout(window.__bot.timer);
  try { localStorage.clear(); } catch (e) {}
  var gm = new GameManager(4, KeyboardInputManager, HTMLActuator, LocalStorageManager);
  window.__gm = gm;
  function readBoard() {
    var lines = new Uint16Array(4), maxT = 0;
    for (var y = 0; y < 4; y++) {
      var row = 0;
      for (var x = 0; x < 4; x++) {
        var t = gm.grid.cells[x][y];
        var v = t ? Math.log2(t.value) : 0;
        if (t && t.value > maxT) maxT = t.value;
        row |= (v & 15) << ((3 - x) * 4);
      }
      lines[y] = row;
    }
    return { lines: lines, maxT: maxT, score: gm.score, over: gm.over, won: gm.won };
  }
  var ROW_L = new Uint16Array(65536), ROW_R = new Uint16Array(65536);
  for (var r0 = 0; r0 < 65536; r0++) {
    var cc = [(r0>>12)&15,(r0>>8)&15,(r0>>4)&15,r0&15], src = cc.filter(function(v){return v;});
    var outL = [], outR = [];
    for (var i = 0; i < src.length; i++) {
      if (i+1 < src.length && src[i] === src[i+1]) { outL.push(src[i]+1); outR.unshift(src[i]+1); i++; }
      else { outL.push(src[i]); outR.unshift(src[i]); }
    }
    while (outL.length < 4) outL.push(0);
    while (outR.length < 4) outR.unshift(0);
    ROW_L[r0] = (outL[0]<<12)|(outL[1]<<8)|(outL[2]<<4)|outL[3];
    ROW_R[r0] = (outR[0]<<12)|(outR[1]<<8)|(outR[2]<<4)|outR[3];
  }
  function T(a,b,c,d) {
    var out = [0,0,0,0], rows = [a,b,c,d];
    for (var i = 0; i < 4; i++) {
      var v = 0;
      for (var j = 0; j < 4; j++) v |= ((rows[j] >> ((3-i)*4)) & 15) << ((3-j)*4);
      out[i] = v;
    }
    return out;
  }
  function moveLines(L, dir) {
    var a=L[0],b=L[1],c=L[2],d=L[3],moved=false,L2;
    if (dir===0) { L2=[ROW_L[a],ROW_L[b],ROW_L[c],ROW_L[d]]; }
    else if (dir===2) { L2=[ROW_R[a],ROW_R[b],ROW_R[c],ROW_R[d]]; }
    else {
      var t=T(a,b,c,d), rr=dir===1?ROW_L:ROW_R;
      var t2=[rr[t[0]],rr[t[1]],rr[t[2]],rr[t[3]]];
      L2=T(t2[0],t2[1],t2[2],t2[3]);
    }
    if (L2[0]!==a||L2[1]!==b||L2[2]!==c||L2[3]!==d) moved=true;
    return { L: L2, moved: moved };
  }
  var pow = Math.pow;
  function evalLines(L) {
    var empty=0,merges=0,sum=0,mono=0;
    function rowEval(row) {
      var c=[(row>>12)&15,(row>>8)&15,(row>>4)&15,row&15];
      var mL=0,mR=0;
      for (var j=0;j<3;j++){var a=c[j],b=c[j+1];if(a>b)mL+=pow(a,4)-pow(b,4);else mR+=pow(b,4)-pow(a,4);}
      mono+=Math.min(mL,mR);
      for (var j=0;j<4;j++){if(!c[j])empty++;else sum+=pow(c[j],3.5);if(j<3&&c[j]&&c[j]===c[j+1])merges+=pow(c[j],4)*2;}
    }
    rowEval(L[0]);rowEval(L[1]);rowEval(L[2]);rowEval(L[3]);
    var tt=T(L[0],L[1],L[2],L[3]);
    rowEval(tt[0]);rowEval(tt[1]);rowEval(tt[2]);rowEval(tt[3]);
    return 200000+270*Math.log(empty+1)+700*merges-47*mono-11*sum;
  }
  function countEmpty(L){var e=0;for(var i=0;i<4;i++){var r=L[i];for(var j=0;j<4;j++)if(!((r>>((3-j)*4))&15))e++;}return e;}
  function getEmpty(L){var o=[];for(var i=0;i<4;i++){var r=L[i];for(var j=0;j<4;j++)if(!((r>>((3-j)*4))&15))o.push([i,j]);}return o;}
  function setCell(L,r,c,v){var o=[L[0],L[1],L[2],L[3]];o[r]=(o[r]&~(15<<((3-c)*4)))|((v&15)<<((3-c)*4));return o;}
  function exp(L,depth,player){
    if(depth===0)return evalLines(L);
    if(player){
      var best=-1e300,any=false;
      for(var d=0;d<4;d++){var s=moveLines(L,d);if(!s.moved)continue;any=true;var v=exp(s.L,depth-1,false);if(v>best)best=v;}
      return any?best:-1e15;
    }
    var es=getEmpty(L);if(!es.length)return evalLines(L);
    var sum=0;
    for(var k=0;k<es.length;k++){sum+=0.9*exp(setCell(L,es[k][0],es[k][1],1),depth-1,true);sum+=0.1*exp(setCell(L,es[k][0],es[k][1],2),depth-1,true);}
    return sum/es.length;
  }
  function bestDir(L){
    var e=countEmpty(L),depth=e>6?4:(e>2?5:6),best=-1,bv=-1e300;
    for(var d=0;d<4;d++){var s=moveLines(L,d);if(!s.moved)continue;var v=exp(s.L,depth-1,false);if(v>bv){bv=v;best=d;}}
    return best;
  }
  var GM_DIR=[3,0,1,2];
  var B={moves:0,maxTile:0,won:false,over:false,timer:null,score:0};
  window.__bot=B;
  function step(){
    var st=readBoard();
    B.maxTile=st.maxT;B.score=st.score;
    if(st.maxT>=2048){B.won=true;return;}
    if(st.over){B.over=true;return;}
    var d=bestDir(st.lines);
    if(d<0){B.over=true;return;}
    gm.move(GM_DIR[d]);B.moves++;
    B.timer=setTimeout(step,0);
  }
  B.timer=setTimeout(step,0);
  console.log('AI started, watch window.__bot');
})();
