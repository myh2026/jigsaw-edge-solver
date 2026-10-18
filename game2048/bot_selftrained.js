// 自训练版 2048 AI：评估权重由"图自训练（自对弈进化）"学到（graph_selftrain_es.py）
// 图结构：6 个特征节点（空格/单调性/平滑度/合并潜力/角落/蛇形）→ 权重边 → 4 个动作节点
// 训练曲线见 graph_selftrain_es_results.md；同深度下自训练权重 > 手调权重（128→512、256→1024）
//
// 用法：在官方经典版 2048（本地 python -m http.server 部署）控制台粘贴本文件全部内容。
(function () {
  if (window.__bot && window.__bot.timer) clearTimeout(window.__bot.timer);
  try { localStorage.clear(); } catch (e) {}
  var gm = new GameManager(4, KeyboardInputManager, HTMLActuator, LocalStorageManager);
  window.__gm = gm;

  function readBoard() {
    var L = new Uint16Array(4), maxT = 0;
    for (var y = 0; y < 4; y++) {
      var row = 0;
      for (var x = 0; x < 4; x++) {
        var t = gm.grid.cells[x][y];
        var v = t ? Math.log2(t.value) : 0;
        if (t && t.value > maxT) maxT = t.value;
        row |= (v & 15) << ((3 - x) * 4);
      }
      L[y] = row;
    }
    return { L: L, maxT: maxT, score: gm.score, over: gm.over, won: gm.won };
  }
  var ROW_L = new Uint16Array(65536), ROW_R = new Uint16Array(65536);
  function mergeLeft(seq) {
    var out = [], i = 0;
    while (i < seq.length) {
      if (i + 1 < seq.length && seq[i] === seq[i + 1]) { out.push(seq[i] + 1); i += 2; }
      else { out.push(seq[i]); i += 1; }
    }
    return out;
  }
  for (var r0 = 0; r0 < 65536; r0++) {
    var cells0 = [(r0>>12)&15,(r0>>8)&15,(r0>>4)&15,r0&15];
    var src = cells0.filter(function (v) { return v; });
    var outL = mergeLeft(src), outR = mergeLeft(src.slice().reverse()).reverse();
    var oL = [0,0,0,0], oR = [0,0,0,0];
    for (var i = 0; i < outL.length; i++) oL[i] = outL[i];
    for (var j = 0; j < outR.length; j++) oR[4 - outR.length + j] = outR[j];
    ROW_L[r0] = (oL[0]<<12)|(oL[1]<<8)|(oL[2]<<4)|oL[3];
    ROW_R[r0] = (oR[0]<<12)|(oR[1]<<8)|(oR[2]<<4)|oR[3];
  }
  function T(a,b,c,d) {
    var out=[0,0,0,0], rows=[a,b,c,d];
    for (var i=0;i<4;i++){var v=0;for(var j=0;j<4;j++)v|=((rows[j]>>((3-i)*4))&15)<<((3-j)*4);out[i]=v;}
    return out;
  }
  function moveLines(L, dir) {
    var a=L[0],b=L[1],c=L[2],d=L[3],L2;
    if (dir===0) L2=[ROW_L[a],ROW_L[b],ROW_L[c],ROW_L[d]];
    else if (dir===2) L2=[ROW_R[a],ROW_R[b],ROW_R[c],ROW_R[d]];
    else {
      var t=T(a,b,c,d), rr=dir===1?ROW_L:ROW_R;
      var t2=[rr[t[0]],rr[t[1]],rr[t[2]],rr[t[3]]];
      L2=T(t2[0],t2[1],t2[2],t2[3]);
    }
    return { L: L2, moved: (L2[0]!==a||L2[1]!==b||L2[2]!==c||L2[3]!==d) };
  }
  function countEmpty(L){var e=0;for(var i=0;i<4;i++)for(var j=0;j<4;j++)if(!((L[i]>>((3-j)*4))&15))e++;return e;}
  function getEmpty(L){var o=[];for(var i=0;i<4;i++)for(var j=0;j<4;j++)if(!((L[i]>>((3-j)*4))&15))o.push([i,j]);return o;}
  function setCell(L,r,c,v){var o=[L[0],L[1],L[2],L[3]];o[r]=(o[r]&~(15<<((3-c)*4)))|((v&15)<<((3-c)*4));return o;}

  // ===== 自训练学到的评估 =====
  var W = [2.53, -1.08, 0.9, 3.8, 1.91, 3.9];
  var SNAKES = [
    [[15,14,13,12],[8,9,10,11],[7,6,5,4],[0,1,2,3]],
    [[12,13,14,15],[11,10,9,8],[4,5,6,7],[3,2,1,0]],
    [[3,2,1,0],[4,5,6,7],[11,10,9,8],[12,13,14,15]],
    [[0,1,2,3],[7,6,5,4],[8,9,10,11],[15,14,13,12]]
  ];
  function feats(L) {
    var cells = [];
    for (var i=0;i<4;i++) for (var j=0;j<4;j++) cells.push((L[i]>>((3-j)*4))&15);
    var grid = [cells.slice(0,4),cells.slice(4,8),cells.slice(8,12),cells.slice(12,16)];
    var empty = 0; for (var k=0;k<16;k++) if(!cells[k]) empty++;
    var mono = 0;
    for (var r=0;r<4;r++){var ml=0,mr=0;for(var c=0;c<3;c++){var a=grid[r][c],b=grid[r][c+1];if(a>b)ml+=a-b;else mr+=b-a;}mono+=Math.min(ml,mr);}
    for (var c2=0;c2<4;c2++){var ml2=0,mr2=0;for(var r2=0;r2<3;r2++){var a2=grid[r2][c2],b2=grid[r2+1][c2];if(a2>b2)ml2+=a2-b2;else mr2+=b2-a2;}mono+=Math.min(ml2,mr2);}
    var smooth = 0;
    for (var r3=0;r3<4;r3++)for(var c3=0;c3<3;c3++){if(grid[r3][c3]&&grid[r3][c3+1])smooth+=Math.abs(grid[r3][c3]-grid[r3][c3+1]);}
    for (var c4=0;c4<4;c4++)for(var r4=0;r4<3;r4++){if(grid[r4][c4]&&grid[r4+1][c4])smooth+=Math.abs(grid[r4][c4]-grid[r4+1][c4]);}
    var merges = 0;
    for (var r5=0;r5<4;r5++)for(var c5=0;c5<3;c5++)if(grid[r5][c5]&&grid[r5][c5]===grid[r5][c5+1])merges++;
    for (var c6=0;c6<4;c6++)for(var r6=0;r6<3;r6++)if(grid[r6][c6]&&grid[r6][c6]===grid[r6+1][c6])merges++;
    var mx = 0; for (var k2=0;k2<16;k2++) if(cells[k2]>mx) mx=cells[k2];
    var corner = (cells[0]===mx||cells[3]===mx||cells[12]===mx||cells[15]===mx)?1:0;
    var snake = -1e9;
    for (var s=0;s<4;s++){var sc=0;for(var r7=0;r7<4;r7++)for(var c7=0;c7<4;c7++)sc+=SNAKES[s][r7][c7]*grid[r7][c7];if(sc>snake)snake=sc;}
    return [empty, -mono, -smooth, merges, corner*mx, snake/60];
  }
  function evalBoard(L) {
    var f = feats(L), s = 0;
    for (var i=0;i<6;i++) s += W[i]*f[i];
    return s;
  }
  var nodes = 0, NODE_CAP = 200000;
  function exp(L, depth, player) {
    nodes++;
    if (depth === 0 || nodes > NODE_CAP) return evalBoard(L);
    if (player) {
      var best = -1e300, any = false;
      for (var d=0;d<4;d++){var s=moveLines(L,d);if(!s.moved)continue;any=true;var v=exp(s.L,depth-1,false);if(v>best)best=v;}
      return any ? best : -1e15;
    }
    var es = getEmpty(L);
    if (!es.length) return evalBoard(L);
    var sum = 0;
    for (var i=0;i<es.length;i++){sum += 0.9*exp(setCell(L,es[i][0],es[i][1],1),depth-1,true)+0.1*exp(setCell(L,es[i][0],es[i][1],2),depth-1,true);}
    return sum/es.length;
  }
  function bestMove(L) {
    var e = countEmpty(L);
    var depth = e > 6 ? 3 : 4;
    nodes = 0;
    var best = -1, bv = -1e300;
    for (var d=0;d<4;d++){var s=moveLines(L,d);if(!s.moved)continue;var v=exp(s.L,depth-1,false);if(v>bv){bv=v;best=d;}}
    return best;
  }
  var GM_DIR = [3,0,1,2];
  var B = { moves:0, maxTile:0, won:false, over:false, timer:null, score:0, msSum:0, steps:0, tileAt:{} };
  window.__bot = B;
  function step() {
    try {
    var st = readBoard();
    B.maxTile = st.maxT; B.score = st.score;
    if (st.won && !gm.keepPlaying) gm.keepPlaying = true;
    if (st.over) { B.over = true; return; }
    if (!B.tileAt[st.maxT]) B.tileAt[st.maxT] = { moves: B.moves, score: st.score };
    var t0 = performance.now();
    var d = bestMove(st.L);
    B.msSum += performance.now() - t0; B.steps++;
    if (d < 0) { B.over = true; return; }
    gm.move(GM_DIR[d]);
    B.moves++;
    } catch (e) { B.error = String(e && e.message || e); B.over = true; }
  }
  window.__run = function (n, budgetMs) {
    var t0 = performance.now();
    for (var i = 0; i < n; i++) {
      if (performance.now() - t0 > budgetMs) break;
      step();
      if (B.over || B.won) break;
    }
    return { moves: B.moves, max: B.maxTile, score: B.score, over: B.over, won: B.won,
             err: B.error || null, avgMs: B.steps ? Math.round(B.msSum / B.steps) : 0 };
  };
  window.__run(2, 500);
  console.log('self-trained AI loaded (sync runner); weights', W);
})();
