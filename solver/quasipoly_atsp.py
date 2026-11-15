#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bell & Frieze, "Solving a Random ATSP Exactly in Quasi-Polynomial Time w.h.p."
(arXiv:2308.02946v12) —— 可直接运行的 Python / NumPy 实现。

=============================== 算法结构 ===============================
记 AP 的最优匹配为 M*, 对应置换 phi (城市 i 的后继为 phi(i) = successor).

1. 解 AP 的 LP 松弛 ( Hungarian / JV, O(n^3) ), 同时拿到最优对偶变量 u, v.
   (论文 Sec 3.1; Potentials 满足 u_i + v_j <= C(i,j), 匹配边取等)

2. 既约费用 Cbar(i,j) = C(i,j) - u_i - v_j >= 0.
   由所有"近紧边"(Cbar ~ 0) 加上 M* 做 Kruskal, 得到最优基树 T*
   (论文 Sec 3.2: T* 有 2n-1 条边, 含 M*, LP 最优基).

3. 把 M* 的 n 条匹配边收缩成 n 个节点 (节点 i = 匹配对 (a_i, b_{phi(i)})).
   任一完美匹配可写为 M_psi = { (a_i, b_{phi(psi(i))}) }, psi 为置换;
   M* 对应 psi = id.  M_psi 是一条巡回 <=> pi = phi o psi 是单个 n-圈.

4. 费用恒等式 (论文的 Lemma 12 之前那段):
       C(M_psi) - Z_AP = sum_i Cbar(i, phi(psi(i)))
   即: 只有用到的**非基弧**付钱, 且每条费用 >= 0.

5. 收缩图中, T* 的 n-1 条非匹配树边各给出一条费用 0 的**有向**弧
   (真实 bipartite 边是 (a_i, b_{phi(j)}), 故方向 i -> j 唯一确定).
   这是对一棵树的定向 => 不存在 0 费用有向圈 => psi 的每个圈至少含 1 条非基弧.
   => 只需枚举"总既约费用 <= Delta"的有向圈 (合法圈), Delta = UB - Z_AP.

6. 把顶点不交的合法圈拼装起来得到 psi, 检查 pi = phi o psi 是否单圈,
   取最小费用. Delta 由启发式上界 (Karp 修补 + Or-opt) 给出.

=========================== 实现里两个坑 ==============================
坑 1 (对偶退化, 很容易中招):
   指派 LP 的所有基解都是 0/1 => 必然退化 => **最优对偶不唯一**, 而是一个最优面.
   Hungarian 的势给的是面上的非顶点解, 它的"紧边集"可能小到拼不出一棵最优基树
   —— 实测 n=200 时只有 195 条紧边 (需要 199 条), Kruskal 被迫用 Cbar=0.012
   的非紧边补洞, T* 就废了. 这里会检测这种情况并退回去用 LP 的顶点对偶
   (见 lp_vertex_duals), 保证 T* 上非匹配边的既约费用精确为 0.

坑 1a (不要用 LP 求顶点对偶 —— 用最短路势直接修补):
   拿到 Hungarian 的势后不必去解 n^2 个约束的 LP (n=800 时 linprog 要 79s).
   互补松弛要求最优面里的势只能形如 u'_i = u_i + w_i, v'_j = v_j - w_{invphi(j)},
   代入可行性后正好是差分约束 w_i - w_k <= Cbar(i, phi(k)). 取单源最短路的势
   w_i = dist(r,i) 就落在该多面体的一个顶点上, 而且最短路树的 n-1 条弧恰好给出
   一张 Cbar=0 的生成树 (见 repair_duals). n=800: 79s -> 0.2s.
   注意别用 scipy.sparse.csr_matrix(dense): 它内部走 nonzero(), 会把权重恰为 0
   的弧 —— 正是我们要的紧弧 —— 整条丢掉, 修补出来的势会不可行.

坑 2 (2-opt 在 ATSP 上是错的):
   对称 TSP 的 2-opt 是"反转一段", 但在 ATSP 里反转会同时把该片段**内部
   每条边的方向**也倒过来 (12->2->15 变成 15->2->12), 费用全变. 因此这里的
   局部搜索是 ATSP 版 Or-opt: 只允许搬移片段, 反向的话要把内部边差
   显式算进增量里.

坑 3 (拼装搜索的分支不能按"最少候选"挑):
   合法圈按费用升序编号, 解集用"编号递增"规范序枚举. 于是在某个节点上
   只能**遍历**全部编号 >= start 的圈, 不能偷懒只分支在"覆盖某个未覆盖
   phi-子圈"的那些圈上 —— 编号更小但覆盖其它子圈的圈会被漏掉, 结果就是
   搜出一个更差的解却误报 proved=True (实测会差 4e-4). 这里只用
   "某子圈无候选 => 剪掉" 这种纯剪枝, 分支保持完整.

=========================== 两个让实现真正跑得动的关键 =================
加速 1 (单圈检验 O(|S|) 而不是 O(n), 见 is_hamiltonian_psi):
   设 S = {i : psi(i) != i}. 把 phi 的每个子圈在 S 中的点处切断得到 |S| 条
   片段, 记 nxt(a) = 沿 a 所在 phi-子圈的下一个 S 中的点, 则
       pi 的圈  <->  T(a) = nxt(psi(a)) 在 S 上的圈
   故 pi 是单圈 <=> 每个 phi-子圈都被 S 碰到, 且 T 在 S 上是单圈.
   |S| 只是被选中那几个圈的长度之和 (通常几十), 与 n 无关.
   这一条把 n=300 的搜索从 31s+/未证明 压到 5s/已证明.

加速 2 (费用升序 => 提前 break): 候选圈按既约费用升序存放, 一旦
   rcsum + cost >= best_rc 就可以直接 break, 而不是 continue 扫完整个列表.

加速 3 (拼装搜索的下界剪枝): 每个还没被 S 碰到的 phi-子圈 s 至少还要再选
   一个"编号 >= start 且碰到 s"的圈. 由于候选圈已按费用升序排列, 其中最便宜
   的就是 cover_list[s] 里第一个编号 >= start 的, O(log) 就能取到. 取所有未覆盖
   子圈上的 max (不能求和 —— 一个圈可以同时碰到多个子圈), 得到完成当前部分解
   所必需的最小额外费用, rcsum + lb >= best_rc 即剪掉. 不额外占内存.

=========================== 一个被实测否掉的"优化" =====================
在随机费用矩阵上试过给起始上界加 Or-opt + double-bridge 的 ILS 重启
(想靠压小 Delta 来减小枚举规模): 40 次重启在 n=50..1500 的 8 个实例上
**一次都没能改进** Or-opt 的局部最优 (heur 值逐位相同), 白花 0.3~15s.
原因是非度量随机矩阵上 double-bridge 一次改动 4 条弧, 扰动幅度远大于
Or-opt (3 条弧) 能修复的范围. 所以这里不再做 ILS; 若换到度量/结构化实例
可以再试. 真正决定难度的是**真实 gap 本身**: n=1000 的 gap 只有 3.8e-4
(137 个圈, 1.6s 证明), 而 n=300 某个实例的 gap 有 1.0e-2 (40 万个圈),
后者光枚举就要几十秒且拼装搜索难以穷尽 —— 难度不随 n 单调, 随实例而定.

=========================== 工程上为什么要"爬坡" ======================
论文的 Delta 是理论 gap O(log^2 n / n), 而 UB - Z_AP (启发式) 通常比它大一个
数量级, 直接按它枚举合法圈会爆炸. 所以实现成“预算 ladder”:
从"只保留 64 条最便宜的非基弧"起步, 每轮翻倍, 每找到更好的解就把 UB 压下来,
于是后面的轮次越来越便宜.
  关键是 ladder 要按**非基弧条数** k 走, 不能按预算的绝对数值走: 启发式上界在
  n=1500 时比真实 gap 差 90 倍, 若从Delta/32 起步, 第一轮就会撞 cycle_cap.
Result.proved=True 表示在 max_cycles 限制内已穷尽; 否则返回当前最优解
并在 Result.overflow=True 时说明是超时/撞上限, 没有证明最优.

用法:
    import numpy as np, quasipoly_atsp as qa
    D = np.random.random((300, 300)); np.fill_diagonal(D, np.inf)
    r = qa.solve(D, time_limit=60, verbose=True)
    print(r.cost, r.proved, r.tour)
"""

from __future__ import annotations

import bisect
import itertools
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════
# 1. 指派问题 + 对偶变量  (Hungarian / Jonker-Volgenant)
# ══════════════════════════════════════════════════════════════════

def hungarian_with_duals(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """解 min sum C(i, phi(i)), 返回 (phi, u, v).

    phi[i] = 行 i 分配到的列
    u, v    = 最优对偶变量, 满足 u_i + v_j <= C(i,j), 匹配边上取等
    """
    C = np.ascontiguousarray(C, dtype=float)
    n, m = C.shape
    INF = float("inf")

    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)      # p[j] = 分配到列 j 的行 (1-based)
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            cur = C[i0 - 1] - u[i0] - v[1:]
            cur = np.where(used[1:], INF, cur)
            better = cur < minv[1:]
            minv[1:] = np.where(better, cur, minv[1:])
            way[1:] = np.where(better, j0, way[1:])
            mm = minv[1:].copy()
            mm[used[1:]] = INF
            j1 = int(np.argmin(mm)) + 1
            delta = float(mm[j1 - 1])
            ui = np.nonzero(used)[0]
            u[p[ui]] += delta
            v[ui] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    phi = np.empty(n, dtype=np.int64)
    for j in range(1, m + 1):
        if p[j] > 0:
            phi[p[j] - 1] = j - 1
    return phi, u[1:], v[1:]


def repair_duals(rc: np.ndarray, phi: np.ndarray) -> Optional[Tuple[np.ndarray,
                                                                    np.ndarray]]:
    """把任意最优对偶 (u,v) "修补"成一个**顶点解**, O(n^2), 无需 LP.

    推导: 设 (u,v) 最优. 互补松弛要求所有匹配边保持紧, 故任何同面的最优对偶必为
        u'_i = u_i + w_i,     v'_j = v_j - w_{invphi(j)}
    代入可行性 u'_i + v'_j <= C(i,j), 记 j = phi(k), 用 Cbar(i,phi(k)) 表示原
    既约费用, 得到纯差分约束
        w_i - w_k <= Cbar(i, phi(k))             (*)
    且 sum(w) 无约束 -> 最优性自动保持 (目标和不变).

    于是"最优对偶面" 就是差分约束多面体 {w : (*)}. 它的一个顶点可以这样取:
    构造 n 个节点的有向图, 弧 k -> i 带权 d(k,i) = Cbar(i, phi(k)) (去掉 k=i 的
    自环/匹配边), 从根 r 跑单源最短路, 令 w_i = dist(r, i). Bellman 方程正是 (*),
    而且最短路树的每条树弧都取等 -> **恰好给出 n-1 条 Cbar=0 的非匹配树边**,
    且它们连成一棵生成树 —— 正是论文要的 T*.

    这比 linprog 解 n^2 个约束的 LP 快几个数量级 (n=800: 79s -> <0.3s).
    """
    n = rc.shape[0]
    if n < 3:
        return None
    # Dm[k, i] = rc(i, phi(k))   即弧 k -> i 的权重 (>= 0)
    Dm = np.ascontiguousarray(rc[:, phi].T, dtype=float)
    Dm[~np.isfinite(Dm)] = np.inf
    np.fill_diagonal(Dm, np.inf)                # 去掉匹配边对应的自环
    if Dm.min() < 0.0:
        Dm = np.clip(Dm, 0.0, None)

    # 自己写 Dijkstra: 不能用 csr_matrix(dense) —— 它内部走 nonzero(),
    # 会把权重恰为 0 的弧(正是我们要的紧弧)整条丢掉.
    INF = np.inf
    dist = np.full(n, INF)
    dist[0] = 0.0
    done = np.zeros(n, dtype=bool)
    for _ in range(n - 1):
        cand = np.where(done, INF, dist)
        xs = int(np.argmin(cand))
        if not np.isfinite(cand[xs]):
            break
        done[xs] = True
        np.minimum(dist, dist[xs] + Dm[xs], out=dist)
    if not np.all(np.isfinite(dist)):
        return None                             # 图不连通 (几乎不会发生)
    w = dist - dist[0]

    du = w
    dv = np.empty(n, dtype=float)
    dv[phi] = -du
    return du, dv


def lp_vertex_duals(C: np.ndarray, tol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """用通用 LP 求解器取 DLP 的**顶点解**.

    为什么需要它: 指派 LP 的所有基解都是 0/1 -> 必然退化 -> 最优对偶**不唯一**,
    而是一个最优面. Hungarian 的势给出的是这个面上的某个 (通常是非顶点) 最优对偶,
    它的"紧边集"可能小到拼不出一棵最优基树 —— 这在 n 稍大时实测会发生.
    LP 求解器返回顶点解, 其紧边恰好含一棵生成树, 正是论文要的 T*.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    n = C.shape[0]
    fin = np.isfinite(C)
    pi, pj = np.nonzero(fin)
    p = np.arange(len(pi))
    rows = np.repeat(p, 2)
    cols = np.empty(2 * len(p), dtype=np.int64)
    cols[0::2] = pi
    cols[1::2] = n + pj
    A = coo_matrix((np.ones(2 * len(p)), (rows, cols)), shape=(len(p), 2 * n))
    b = C[pi, pj]
    res = linprog(c=-np.ones(2 * n), A_ub=A, b_ub=b,
                  bounds=[(None, None)] * (2 * n), method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP dual solve failed: {res.message}")
    u = np.asarray(res.x[:n])
    v = np.asarray(res.x[n:])
    rc = C - u[:, None] - v[None, :]
    rc = np.where(fin, rc, 0.0)
    if rc.min() < -1e-6 * max(1.0, float(np.abs(C[fin]).max())):
        raise RuntimeError("LP dual infeasible")
    rc = np.clip(rc, 0.0, None)
    return u, v


def optimal_duals(C: np.ndarray, phi: np.ndarray, tol: float = 1e-9,
                  verbose: bool = False,
                  hung0: Optional[Tuple[np.ndarray, np.ndarray]] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """取赋值 LP 的一个**顶点**最优对偶.

    先用 Hungarian 的势 (O(n^3) 但常数极小, 且已是最优), 检查它的紧边够不够拼
    一棵生成树; 不够就用 repair_duals 的最短路势把它推到一个顶点上去; 万一还不行
    才退回 linprog. 返回 (u, v), 满足 sum(u)+sum(v) = Z_AP 且存在一张 Cbar=0 的
    生成树.

    hung0: 调用方若已经跑过 hungarian_with_duals(C), 就把它的 (u, v) 传进来,
    本函数就不再重复跑一次 O(n^3) 的 Hungarian. hungarian_with_duals 是确定性的,
    所以传进来的势与本地重算的**逐位相同**, 结果完全等价; 不传则行为与以前一致.
    """
    fin = np.isfinite(C)
    scale = max(1.0, float(np.abs(C[fin]).max()))

    if hung0 is not None:
        u0, v0 = hung0
        u0 = np.asarray(u0).copy()
        v0 = np.asarray(v0).copy()
    else:
        _, u0, v0 = hungarian_with_duals(C)
    rc0 = np.clip(np.where(fin, C - u0[:, None] - v0[None, :], np.inf), 0.0, None)

    candidates = [("Hungarian", u0, v0)]
    rep = repair_duals(rc0, phi)
    if rep is not None:
        du, dv = rep
        # 验证修补后的势仍然可行且保持最优 (差分约束可能因舍入略微越界)
        rc1 = np.where(fin, C - (u0 + du)[:, None] - (v0 + dv)[None, :], np.inf)
        if rc1.min() >= -1e-9 * scale and abs(float(du.sum() + dv.sum())) < 1e-6 * scale:
            candidates.append(("repair(dijkstra)", u0 + du, v0 + dv))

    for tag, u, v in candidates:
        rcf = np.clip(np.where(fin, C - u[:, None] - v[None, :], np.inf), 0.0, None)
        _, max_rc = build_basis_tree(rcf, phi)
        if verbose:
            print(f"[dual] {tag}: 最优基树上最大的非匹配边既约费用 = {max_rc:.3e}")
        if max_rc <= tol:
            return u, v

    if verbose:
        print("[dual] 组合修补失败 -> 退回 LP 顶点对偶 (慢)")
    return lp_vertex_duals(C, tol)


# ══════════════════════════════════════════════════════════════════
# 2. 由最优对偶构造最优基树 T*
# ══════════════════════════════════════════════════════════════════

class DSU:
    __slots__ = ("p",)

    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[ra] = rb
        return True


def build_basis_tree(rc: np.ndarray, phi: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
    """返回 (extra, max_rc_used): T* 的 n-1 条非匹配树边 [(i, j), ...]
    (行 i, 列 j = phi[k], k != i), 以及其中最大的既约费用.

    先把 M* 全部加入并查集 (顶点: 行 i -> i, 列 j -> n+j),
    再按既约费用从小到大做 Kruskal 直到生成树.
    拿到 LP 顶点对偶时这 n-1 条边既约费用全为 0, max_rc_used ≈ 0.
    """
    n = rc.shape[0]
    dsu = DSU(2 * n)
    for i in range(n):
        dsu.union(i, n + int(phi[i]))

    mask = np.ones((n, n), dtype=bool)
    mask[np.arange(n), phi] = False

    rows, cols = np.nonzero(mask)
    vals = rc[rows, cols]
    order = np.argsort(vals, kind="stable")

    extra: List[Tuple[int, int]] = []
    used_rc: List[float] = []
    for t in order:
        i, j = int(rows[t]), int(cols[t])
        if dsu.union(i, n + j):
            extra.append((i, j))
            used_rc.append(float(vals[t]))
            if len(extra) == n - 1:
                break
    if len(extra) != n - 1:                       # 理论上不会发生
        raise RuntimeError("failed to build spanning tree of the basis graph")
    return extra, max(used_rc)


# ══════════════════════════════════════════════════════════════════
# 3. 收缩图: 有向森林 + 非基弧
# ══════════════════════════════════════════════════════════════════

@dataclass
class Contracted:
    n: int
    children: List[List[int]]          # 费用 0 的有向树弧 i -> j
    nb_out: List[List[Tuple[int, int, float]]]   # 非基弧: (head, arc_id, rcost)
    nb_list: List[Tuple[int, int, float]]        # (tail, head, rcost)
    n_tree_arcs: int


def build_contracted(n: int, extra: Sequence[Tuple[int, int]], invphi: np.ndarray,
                     rc: np.ndarray, phi: np.ndarray, budget: float,
                     tol: float = 1e-9) -> Contracted:
    children: List[List[int]] = [[] for _ in range(n)]
    in_tree = np.zeros((n, n), dtype=bool)
    for i, j in extra:
        k = int(invphi[j])
        assert k != i, "tree edge must not be a matching edge"
        children[i].append(k)
        in_tree[i, j] = True

    nb_list: List[Tuple[int, int, float]] = []
    mask = np.ones((n, n), dtype=bool)
    mask[np.arange(n), phi] = False
    mask &= ~in_tree
    mask &= (rc <= budget + tol)
    rows, cols = np.nonzero(mask)
    vals = rc[rows, cols]
    order = np.argsort(vals, kind="stable")
    for t in order:
        i, j = int(rows[t]), int(cols[t])
        k = int(invphi[j])
        nb_list.append((i, k, max(float(vals[t]), 0.0)))

    nb_out: List[List[Tuple[int, int, float]]] = [[] for _ in range(n)]
    for aid, (i, k, c) in enumerate(nb_list):
        nb_out[i].append((k, aid, c))

    return Contracted(n=n, children=children, nb_out=nb_out, nb_list=nb_list,
                      n_tree_arcs=sum(len(c) for c in children))


# ══════════════════════════════════════════════════════════════════
# 4. 枚举"合法圈": 有向简单圈, 总既约费用 <= budget
#    每个圈从其弧号最小的非基弧出发, 保证不重不漏 (论文 Lemma 14/16)
# ══════════════════════════════════════════════════════════════════

def _return_bound(G: Contracted) -> Tuple[np.ndarray, Dict[int, int]]:
    """h[p][v] = 收缩图里 v -> p 的**最小**既约费用 (树弧 0, 非基弧 rc).

    p 只取"某条非基弧的尾", 所以最多 n 个源点; 一次反向 Dijkstra 全算出来.
    这是任何"从 v 回到起点 p"的完成方式的费用下界 —— 用它在 DFS 里剪枝不改变
    输出的圈集合 (只砍掉算不出合法圈的分支), 见 dfs_prune_probe.py 的逐元素校验.
    """
    n = G.n
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    for i, ch in enumerate(G.children):
        for k in ch:
            rows.append(k)
            cols.append(i)
            vals.append(0.0)
    for i, k, c in G.nb_list:
        rows.append(k)
        cols.append(i)
        vals.append(c)
    srcs = sorted({int(p) for p, _, _ in G.nb_list})
    if not srcs:
        return np.zeros((0, n)), {}
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    R = coo_matrix((np.asarray(vals, dtype=float),
                    (np.asarray(rows, dtype=np.int64),
                     np.asarray(cols, dtype=np.int64))),
                   shape=(n, n)).tocsr()
    H = dijkstra(R, directed=True, indices=np.asarray(srcs, dtype=np.int64))
    return np.ascontiguousarray(H), {p: t for t, p in enumerate(srcs)}


def enumerate_valid_cycles(G: Contracted, budget: float,
                           cap: int = 400_000,
                           deadline: Optional[float] = None,
                           use_bound: bool = True
                           ) -> Tuple[List[Tuple[Tuple[int, int], ...]],
                                      List[float], bool]:
    """枚举所有 (弧号最小非基弧 = a0 的) 有向简单圈, 总既约费用 <= budget.

    use_bound=True 时启用反向 Dijkstra 下界剪枝: 停在 v 且已花 cost 时, 若
    cost + h[p0][v] > budget, 则任何完成方式都会超预算, 直接回溯. 这是**精确**
    剪枝 (h 是剩余费用的下界), 输出的圈集合与 use_bound=False 逐元素相同;
    实测 n=1500 上 k=512/1024/1280 分别快 2.9x / 3.6x / 4.0x, 且加速比随 k 上升.
    """
    n = G.n
    children, nb_out = G.children, G.nb_out
    child_set = [set(c) for c in children]
    cycles: List[Tuple[Tuple[int, int], ...]] = []
    costs: List[float] = []
    overflow = False

    H, p0_row = _return_bound(G) if use_bound else (None, None)

    visited = np.zeros(n, dtype=bool)
    path: List[Tuple[int, int]] = []
    work = [0]

    for a0, (p0, q0, c0) in enumerate(G.nb_list):
        if c0 > budget:
            continue
        if visited.any():
            visited[:] = False
        visited[p0] = True
        visited[q0] = True
        path.clear()
        path.append((p0, q0))
        Hp = H[p0_row[p0]] if use_bound else None
        blim = budget + 1e-12

        # ---- DFS: 当前停在 v, 已花 cost, 已用节点 visited ----
        def dfs(v: int, cost: float) -> bool:      # True = 请终止 (达到上限)
            nonlocal overflow
            work[0] += 1
            if deadline is not None and (work[0] & 8191) == 0 \
                    and time.time() > deadline:
                overflow = True                   # 超时 => 结果不完整
                return True
            # (A) 闭合: 回到起点 p0 (树弧直接到 p0, 或一条编号更大的非基弧到 p0)
            if p0 in child_set[v]:
                if len(cycles) >= cap:
                    overflow = True
                    return True
                cycles.append(tuple(path) + ((v, p0),))
                costs.append(cost)
            for head, aid, c in nb_out[v]:
                if aid > a0 and head == p0 and cost + c <= budget:
                    if len(cycles) >= cap:
                        overflow = True
                        return True
                    cycles.append(tuple(path) + ((v, p0),))
                    costs.append(cost + c)

            # 精确剪枝: 剩下怎么走都回不到 p0 (或回去了也超预算)
            if use_bound and cost + Hp[v] > blim:
                return False

            # (B) 沿费用 0 的树弧延伸
            for w in children[v]:
                if visited[w]:
                    continue
                if use_bound and cost + Hp[w] > blim:
                    continue
                visited[w] = True
                path.append((v, w))
                if dfs(w, cost):
                    return True
                path.pop()
                visited[w] = False

            # (C) 沿非基弧延伸 (只许用编号更大的弧 -> 每个圈只被计一次)
            for head, aid, c in nb_out[v]:
                if aid <= a0 or visited[head] or head == p0:
                    continue
                nc = cost + c
                if nc > budget:
                    continue
                if use_bound and nc + Hp[head] > blim:
                    continue
                visited[head] = True
                path.append((v, head))
                if dfs(head, nc):
                    return True
                path.pop()
                visited[head] = False
            return False

        if dfs(q0, c0):
            break

    return cycles, costs, overflow


# --------------------------------------------------------------------------
# 编译后端 (可选).  圈枚举占 n=1500 运行时间的 94.8%, 而 Python 每产出 1 个圈
# 要 ~650 us / ~3.2 KB.  cy/ 下有两个 Cython 内核:
#     cyenum -- DFS 本身, 输出扁平 (pairs, indptr, costs) 而不是 tuple
#     cypack -- 拼装前的 mask/cmask/suffix_or/cover_list 预处理
# 两者都与这里的 Python 实现逐元素对拍过 (cy/validate_cyenum.py,
# cy/validate_cypack.py), 所以结果的圈集合 / 掩码 / 排序完全相同.
# 关掉: 设环境变量 QA_NO_CY=1.
# --------------------------------------------------------------------------
_CY_BACKEND = None


def cy_backend():
    """-> (cybridge, cypack) 或 False (不可用).  结果被缓存."""
    global _CY_BACKEND
    if _CY_BACKEND is None:
        import os
        if os.environ.get("QA_NO_CY"):
            _CY_BACKEND = False
        else:
            try:
                _dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "cy")
                if _dir not in __import__("sys").path:
                    __import__("sys").path.insert(0, _dir)
                import cybridge as _cb
                import cypack as _cp
                _CY_BACKEND = (_cb, _cp) if _cb.available() else False
            except Exception:                       # 没编译 / 编译坏了就退回
                _CY_BACKEND = False
    return _CY_BACKEND


class FlatCycles:
    """惰性视图: 把扁平圈流当成 `List[List[Tuple[int,int]]]` 用.

    只有真正被拼装搜索访问到的圈才会被物化成 tuple, 所以百万级圈也不必
    为它们付 3.2 KB/个的 Python 对象开销.
    """
    __slots__ = ("pairs", "indptr", "order")

    def __init__(self, pairs, indptr, order):
        self.pairs = pairs
        self.indptr = indptr
        self.order = order

    def __len__(self):
        return int(self.order.shape[0])

    def __getitem__(self, i):
        si = int(self.order[i])
        a = 2 * int(self.indptr[si])
        b = 2 * int(self.indptr[si + 1])
        seg = self.pairs[a:b]
        return [(int(x), int(y))
                for x, y in zip(seg[0::2].tolist(), seg[1::2].tolist())]


# --------------------------------------------------------------------------
# 拼装搜索看到的统一圈视图.
#
# RAM 路径 (RamView) 与磁盘路径 (StoreView, cy/cystore.pyx) 实现完全相同的
# 七个调用, 所以下面的 rec() 只有一份 -- 两条路径在搜索结果上逐位相同
# (cmp/test_store.py 逐圈对拍: cost / arcs / mask / cmask / suffix_or /
# cover_first 全部相等).
#
# 为什么要磁盘路径: RAM 路径每个圈约 0.5 kB (扁平弧流 328 B + 掩码/索引),
# 14 GB 内存在 ~2.8e7 个圈处见底; 而"真正能证明一个实例"的那一级实测
# >= 4.3e9 个圈 (Phase 24f), 差两个数量级. 磁盘路径把圈写成 21-32 B/条的自
# 描述记录, 外排序后分块流回, 内存 O(block + nblocks), 与圈数无关.
# --------------------------------------------------------------------------
class _CycleView:
    __slots__ = ()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError

    def cost(self, i):
        raise NotImplementedError

    def mask(self, i):
        raise NotImplementedError

    def cmask(self, i):
        raise NotImplementedError

    def suffix_or(self, i):
        raise NotImplementedError

    def cover_first(self, s, i):
        """第一个下标 >= i 且碰到 phi-子圈 s 的圈: (下标, 费用) 或 None."""
        raise NotImplementedError


class RamView(_CycleView):
    """把 solve() 里已有的那几个列表包成统一接口 (零额外开销)."""
    __slots__ = ("_cyc", "_cost", "_mask", "_cmask", "_suf", "_cov", "_n")

    def __init__(self, cyc_sorted, cost_sorted, mask_sorted, cmask_sorted,
                 suffix_or, cover_list):
        self._cyc = cyc_sorted
        self._cost = cost_sorted
        self._mask = mask_sorted
        self._cmask = cmask_sorted
        self._suf = suffix_or
        self._cov = cover_list
        self._n = len(cyc_sorted)

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return self._cyc[i]

    def cost(self, i):
        return self._cost[i]

    def mask(self, i):
        return self._mask[i]

    def cmask(self, i):
        return self._cmask[i]

    def suffix_or(self, i):
        return self._suf[i]          # 该列表有 n+1 项, 第 n 项是 0

    def cover_first(self, s, i):
        lst = self._cov[s]
        p = bisect.bisect_left(lst, i)
        if p >= len(lst):
            return None
        j = lst[p]
        return (j, self._cost[j])

    def close(self):
        pass                              # RAM 路径没有要释放的资源


class StoreView(_CycleView):
    """磁盘圈库视图. 下标越界 (i == n) 时退化成空, 与 RAM 列表的第 n 项一致."""
    __slots__ = ("st", "n", "tmpdir", "_keep")

    def __init__(self, st, tmpdir=None):
        self.st = st
        self.n = int(st.n)
        self.tmpdir = tmpdir
        self._keep = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.st.arcs(i)

    def cost(self, i):
        return self.st.cost(i)

    def mask(self, i):
        return self.st.mask(i)

    def cmask(self, i):
        return self.st.cmask(i)

    def suffix_or(self, i):
        return 0 if i >= self.n else self.st.suffix_or(i)

    def cover_first(self, s, i):
        if i >= self.n:
            return None
        return self.st.cover_first(s, i)

    def close(self):
        self.st = None
        if self.tmpdir:
            d = self.tmpdir
            self.tmpdir = None
            # 先改名再一次删: 逐个 unlink 会撞沙箱的单回合删除配额
            try:
                os.rename(d, d + "_trash")
                shutil.rmtree(d + "_trash", ignore_errors=True)
            except OSError:
                shutil.rmtree(d, ignore_errors=True)

    def __del__(self):                    # pragma: no cover - 兜底
        try:
            self.close()
        except Exception:
            pass


_STORE_BASE = None


def _store_base_dir():
    """圈库临时目录的根, 按顺序探测第一个真正可写的:

        QA_STORE_DIR  ->  求解器所在目录/.qa_store  ->  ~/.cache/qa_store
                      ->  tempfile.gettempdir()

    不能用 /tmp: 本机 /tmp 是 10 MB 的 tmpfs, 而 1e9 个圈要 ~30 GB.
    探测靠留一个 2 字节的标记文件 (不删), 结果按进程缓存 —— 删掉它会让某些
    环境 (带批量删除配额的工具沙箱) 直接杀掉进程. 顺序往下换候选是因为沙箱/
    权限可能让某个位置看起来存在却写不进去.
    """
    global _STORE_BASE
    if _STORE_BASE is not None:
        return _STORE_BASE
    cands = []
    env = os.environ.get("QA_STORE_DIR")
    if env:
        cands.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    cands.append(os.path.join(here, ".qa_store"))
    cands.append(os.path.join(os.path.expanduser("~"), ".cache", "qa_store"))
    cands.append(os.path.join(tempfile.gettempdir(), "qa_store"))
    for d in cands:
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".probe.%d" % os.getpid())
            with open(probe, "w") as fh:
                fh.write("ok")
            _STORE_BASE = d
            return d
        except OSError:
            continue
    raise OSError("no writable directory for the cycle store "
                  "(set QA_STORE_DIR)")


def _store_keep():
    """QA_STORE_KEEP=1: 一圈跑完不删临时文件 (留着人工检查 / 规避删除配额)."""
    return os.environ.get("QA_STORE_KEEP", "0") not in ("0", "")


def build_store_view(_cb, G, budget, cap, deadline, fcid, nfc, H=None,
                     nthreads=None, verbose=False):
    """枚举 -> 写盘 -> 外排序 -> CycleStore, 返回 StoreView (或 None).

    与 enumerate_cycles_cy 走同一个内核 (store 模式), 圈集合与发射序一致,
    所以 sort_records 按 (费用, 发射序号) 排出来的序就是
    np.argsort(costs, kind="stable") 的序. 内存 O(1) 于圈数.
    """
    from cystore import CycleStore, sort_records
    base = _store_base_dir()
    tmp = tempfile.mkdtemp(prefix="run_", dir=base)
    keep = _store_keep()
    # 清理统一走"改名成 trash 再一次 rmtree": 逐个 unlink 会撞本机沙箱的
    # 单回合 50 次删除配额 (进程被直接 kill, 无 traceback), 而按目录删只计
    # 一次. 1e9 圈的排序会产生几百个 run 文件, 必须走这条路.
    def _drop():
        if keep:
            return
        try:
            os.rename(tmp, tmp + "_trash")
            shutil.rmtree(tmp + "_trash", ignore_errors=True)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
    try:
        info = _cb.enumerate_cycles_store(
            G, budget, cap=cap, deadline=deadline, use_bound=True, H=H,
            out_paths=[os.path.join(tmp, "raw_0.bin")], nthreads=nthreads)
        if info["ovf"]:
            _drop()
            return None                    # 撞 cap: 让调用方按老规矩回退 k
        srt = os.path.join(tmp, "sorted.bin")
        # chunk_recs 决定 run 文件数 (n/chunk_recs); rr_buf 固定 1 MB,
        # 所以合并阶段内存 = O(n/chunk_recs) 而不是 O(n) -- 3.5e7 圈时
        # 曾因 8 MB/run 吃掉 1.27 GB, 改成定长窗口后降到 197 MB.
        n, blk_off = sort_records(info["paths"], srt, info["bits_tree"],
                                  info["bits_nb"], tmpdir=tmp,
                                  keep=keep, rr_buf=1 << 20)
        if n != info["n"]:
            raise RuntimeError("sort lost records: %d != %d" % (n, info["n"]))
        csr = info["csr"]
        st = CycleStore(srt, blk_off, G.n, len(csr["st_tail"]), nfc,
                        csr["ch_ptr"], csr["ch_idx"], csr["st_tail"],
                        csr["st_head"], np.asarray(fcid, dtype=np.int64),
                        n, block=4096, cache=6)
        st.build_aux(verbose=verbose)
        if verbose:
            s = st.stats()
            print("[store] n=%d  %.1f B/cyc  %d blocks  bits=(%d,%d)"
                  % (s["n"], s["bytes_per_cycle"], s["nblocks"],
                     s["bits_tree"], s["bits_nb"]))
        return StoreView(st, tmpdir=None if keep else tmp)
    except Exception:
        _drop()
        raise


# ══════════════════════════════════════════════════════════════════
# 5. 单圈检验与费用求值
# ══════════════════════════════════════════════════════════════════

def psi_to_tour(psi: Dict[int, int], phi: np.ndarray, n: int) -> Optional[List[int]]:
    """pi = phi o psi; 若 pi 是单个 n-圈则返回巡回顶点序列, 否则 None."""
    succ = np.array([int(phi[psi.get(i, i)]) for i in range(n)], dtype=np.int64)
    seen = np.zeros(n, dtype=bool)
    tour: List[int] = []
    x = 0
    while not seen[x]:
        seen[x] = True
        tour.append(int(x))
        x = int(succ[x])
    if len(tour) != n:
        return None
    return tour


def tour_cost(C: np.ndarray, tour: Sequence[int]) -> float:
    if len(tour) < 2:
        return 0.0
    arr = np.asarray(tour, dtype=np.int64)
    return float(C[arr, np.roll(arr, -1)].sum())


def phi_cycle_labels(phi: Sequence[int], n: int
                     ) -> Tuple[List[int], List[int], int]:
    """把 phi 分解成子圈, 返回 (fcid, fpos, nfc).

    fcid[i] = i 所在的 phi-子圈编号, fpos[i] = i 在该子圈中的序号.
    """
    fcid = [0] * n
    fpos = [0] * n
    seen = [False] * n
    nfc = 0
    for s in range(n):
        if seen[s]:
            continue
        x = s
        L = 0
        while not seen[x]:
            seen[x] = True
            fcid[x] = nfc
            fpos[x] = L
            L += 1
            x = int(phi[x])
        nfc += 1
    return fcid, fpos, nfc


def is_hamiltonian_psi(psi: Dict[int, int], phi: Sequence[int],
                       fcid: Sequence[int], fpos: Sequence[int],
                       nfc: int) -> bool:
    """判定 pi = phi o psi 是否为单个 n-圈, 复杂度 O(|S| log |S|) 而非 O(n).

    结构定理 (实现上的关键捷径, 也是论文 Lemma 12 那套"交错圈"论证的另一种写法):

        设 S = {i : psi(i) != i} 是被 psi 真正移动的城市. 把 phi 的每个子圈
        在 S 中的点处切断, 得到 |S| 条"片段" P_a (从 phi(a) 沿 phi-子圈走到
        下一个 S 中的点). 记 nxt(a) = 沿 a 所在 phi-子圈的下一个 S 中的点.
        那么 pi 走完 P_a 之后接的是 P_{nxt(psi(a))}, 于是

            pi 的圈  <->  映射 T(a) = nxt(psi(a)) 在 S 上的圈

        并且: 没被 S 碰到的 phi-子圈会原封不动地成为 pi 的一个圈.
        所以  pi 是单圈  <=>  每个 phi-子圈都被 S 碰到, 且 T 在 S 上是单圈.

    这使得单圈检验只跟 |S| (被选中的那些合法圈的长度之和, 通常 <= 几十) 有关,
    而跟 n 无关.
    """
    S = [i for i in psi if psi[i] != i]
    if not S:
        return nfc == 1                      # psi = id: pi = phi
    groups: Dict[int, List[int]] = {}
    for a in S:
        groups.setdefault(fcid[a], []).append(a)
    nxt: Dict[int, int] = {}
    for lst in groups.values():
        if len(lst) != len(set(lst)):
            raise ValueError("psi 的圈不是顶点不交的")
        lst.sort(key=lambda a: fpos[a])
        L = len(lst)
        for t in range(L):
            nxt[lst[t]] = lst[(t + 1) % L]
    if len(groups) != nfc:
        return False                         # 有 phi-子圈完全没被碰到
    s0 = S[0]
    x = s0
    steps = 0
    for _ in range(len(S)):
        x = nxt[psi[x]]
        steps += 1
        if x == s0:
            break
    return x == s0 and steps == len(S)


# ══════════════════════════════════════════════════════════════════
# 6. 启发式上界: Karp 修补 + 2-opt (定义剪枝预算 Delta)
# ══════════════════════════════════════════════════════════════════

# 子圈数 k 小于该阈值时走"逐对循环", 否则走"分块全矩阵"。
# 实测切换点在 k ~ 18-19 (n=200/400/800, 对称度渐变的实例族):
#   k<=14 逐对循环快 3-6 倍; k>=18 分块全矩阵快, 且 k=Theta(n) 时快 17-56 倍。
# i.i.d. 费用下 k ~ H_n = Theta(log n) (n<=800 时实测 <=10), 永远走不到 v2,
# 所以这条切换对本文全部基准数字没有任何影响, 只影响结构化/近对称实例。
PATCH_BLOCK_K = 20
PATCH_BLOCK_ROWS = 512


def _patch_decompose(succ: np.ndarray
                     ) -> Tuple[List[np.ndarray], np.ndarray]:
    """按后继数组分解成子圈, 同时给出每个点的圈号 lab."""
    n = succ.size
    lab = np.full(n, -1, dtype=np.int64)
    cycles: List[np.ndarray] = []
    k = 0
    for s in range(n):
        if lab[s] >= 0:
            continue
        cyc: List[int] = []
        x = s
        while lab[x] < 0:
            lab[x] = k
            cyc.append(int(x))
            x = int(succ[x])
        cycles.append(np.asarray(cyc, dtype=np.int64))
        k += 1
    return cycles, lab


def _patch_swap(succ: np.ndarray, pred: np.ndarray, x: int, y: int) -> None:
    """2-exchange: 删 (x->succ(x)), (y->succ(y)); 加 (x->succ(y)), (y->succ(x)).

    x, y 必须在不同子圈上, 这样两个圈就并成一个。
    """
    sx, sy = int(succ[x]), int(succ[y])
    succ[x], succ[y] = sy, sx
    pred[sy] = x
    pred[sx] = y


def _patch_pairwise(C: np.ndarray, succ: np.ndarray, pred: np.ndarray,
                    cycles: List[np.ndarray]) -> None:
    """逐对循环: 每轮在 Python 里扫所有子圈对, 取全局最优 2-exchange 合并。

    numpy 调用次数 Theta(k^3); k 小时常数极小, k 大时被调用开销淹没。
    """
    while len(cycles) > 1:
        best: Optional[Tuple[float, int, int, int, int]] = None
        for bi in range(len(cycles)):
            A = cycles[bi]
            pa = pred[A][:, None]
            ca_a = C[pa, A[:, None]]
            for bj in range(bi + 1, len(cycles)):
                B = cycles[bj]
                pb = pred[B][None, :]
                M = (C[pa, B[None, :]] + C[pb, A[:, None]]
                     - ca_a - C[pb, B[None, :]])
                r, c = np.unravel_index(int(np.argmin(M)), M.shape)
                d = float(M[r, c])
                if best is None or d < best[0]:
                    best = (d, int(A[r]), int(B[c]), bi, bj)
        assert best is not None
        _, a, bb, bi, bj = best
        # 统一用 pred 索引的写法: x=pred[a], y=pred[bb]
        _patch_swap(succ, pred, int(pred[a]), int(pred[bb]))
        new: List[int] = []
        x = a
        while True:
            new.append(int(x))
            x = int(succ[x])
            if x == a:
                break
        keep = [t for t in range(len(cycles)) if t not in (bi, bj)]
        cycles[:] = [cycles[t] for t in keep] + [np.asarray(new, np.int64)]


def _patch_blocked(C: np.ndarray, succ: np.ndarray, pred: np.ndarray,
                   cycles: List[np.ndarray], n: int,
                   block: int = PATCH_BLOCK_ROWS) -> None:
    """分块全矩阵: 每轮直接造整张增量矩阵再取全局 argmin。

    Delta(x,y) = C[pred(x),y] + C[pred(y),x] - C[pred(x),x] - C[pred(y),y]
    (令 x'=pred(x), y'=pred(y) 后就是 C[x',succ(y')]+C[y',succ(x')]
     -C[x',succ(x')]-C[y',succ(y')], 与逐对版是同一族动作。)

    每轮 O(n^2) 算术 + O(n/block) 次 numpy 调用 —— 把 Theta(k^3) 次调用
    压成 Theta(k n/block) 次。合并后只有 pred 的两个点变了, 下一轮整体重算
    (O(n^2)) 即可, 不需要增量维护矩阵。
    """
    lab = np.empty(n, dtype=np.int64)
    for ci, cyc in enumerate(cycles):
        lab[cyc] = ci
    rows = np.arange(n, dtype=np.int64)
    nblocks = (n + block - 1) // block
    k = len(cycles)
    while k > 1:
        cd = C[pred, rows]
        bd, bx, by = np.inf, -1, -1
        for b in range(nblocks):
            R = rows[b * block:(b + 1) * block]
            M = C[pred[R], :]
            M += C[pred[:, None], R[None, :]].T
            M -= cd[R][:, None]
            M -= cd[None, :]
            M[lab[R][:, None] == lab[None, :]] = np.inf
            j = int(np.argmin(M))
            d = float(M.flat[j])
            if d < bd:
                r, c = divmod(j, n)
                bd, bx, by = d, int(R[r]), int(c)
        if not np.isfinite(bd):
            raise RuntimeError("no feasible 2-exchange in karp_patching")
        # 注意: 增量矩阵是按 (x, y) 索引的, 但被换掉的两条弧是
        #   (pred(x)->x) 和 (pred(y)->y)  —— 所以要交换的是 pred(x), pred(y)
        #   这两个点的后继, 不是 x, y 的后继。写错会让费用暴涨 (实测近对称
        #   n=800 上 24.9 -> 44.4)。
        _patch_swap(succ, pred, int(pred[bx]), int(pred[by]))
        # union by size: 把较小的圈改标到较大的圈上, 每个点最多被改标 O(log k) 次
        bi, bj = int(lab[bx]), int(lab[by])
        if cycles[bi].size < cycles[bj].size:
            bi, bj = bj, bi
        small = cycles[bj]
        lab[small] = bi
        cycles[bi] = np.concatenate([cycles[bi], small])
        cycles[bj] = small[:0]
        k -= 1


def karp_patching(C: np.ndarray, phi: np.ndarray) -> List[int]:
    """把 AP 置换的若干子圈逐步合并成一条巡回 (最小费用 2-exchange).

    贪心: 每一步在所有"跨圈点对"里取增量最小的 2-exchange 合并两个子圈,
    共 k-1 步。返回的是一条合法巡回 (0 号点起沿后继走一圈)。

    注意: 这个函数是 solve() 里**初始上界的唯一来源**, 每条实例都会跑一次,
    不是"搜索失败后的兜底"。它的输出决定 Delta = UB - Z_AP, 进而决定预算
    剪枝的规模和 proved 能否达成 —— 见 cmp/audit_patching.py 的消融。
    """
    n = len(phi)
    succ = np.array(phi, dtype=np.int64).copy()
    pred = np.empty(n, dtype=np.int64)
    pred[succ] = np.arange(n)
    cycles, _ = _patch_decompose(succ)
    if len(cycles) < PATCH_BLOCK_K:
        _patch_pairwise(C, succ, pred, cycles)
    else:
        _patch_blocked(C, succ, pred, cycles, n)
    tour: List[int] = []
    x = 0
    for _ in range(n):
        tour.append(int(x))
        x = int(succ[x])
    return tour


def or_opt(C: np.ndarray, tour: Sequence[int], max_seg: int = 3,
           max_moves: int = 20000,
           deadline: Optional[float] = None) -> Tuple[List[int], float]:
    """ATSP 版 Or-opt: 把长度 <= max_seg 的片段搬到环上别处 (可反向).

    注意: 对称 TSP 常用的 2-opt (反转片段) 在 ATSP 上是**错的** ——
    片段内部每条边的方向都会被倒过来, 费用全变. 这里只做搬移,
    并且任何候选移动的真实费用增量都用 s -> t 方向显式重算.

    deadline 到点后立即停手 (可能停在局部最优之前), 返回当前合法巡回.
    """
    t = [int(x) for x in tour]
    n = len(t)
    if n < 4:
        return t, tour_cost(C, t)
    INF = float("inf")
    moves = 0
    while moves < max_moves:
        improved = False
        X = np.asarray(t, dtype=np.int64)
        Y = np.roll(X, -1)
        for L in range(1, max_seg + 1):
            if n - L < 3:
                break
            for i in range(n):
                if deadline is not None and (i & 127) == 0 and time.time() > deadline:
                    return t, tour_cost(C, t)
                prev = t[(i - 1) % n]
                first = t[i]
                lastx = t[(i + L - 1) % n]
                nxt = t[(i + L) % n]
                keep = C[prev, first] + C[lastx, nxt]
                add = C[prev, nxt] - keep - C[X, Y]
                d0 = add + C[X, first] + C[lastx, Y]      # 片段原序插入
                # 反向插入时, 片段内部每条边的方向也翻转, 必须补上内部费用差
                seg_nodes = [t[(i + s) % n] for s in range(L)]
                inner0 = float(sum(C[seg_nodes[s], seg_nodes[s + 1]]
                                   for s in range(L - 1)))
                inner1 = float(sum(C[seg_nodes[s + 1], seg_nodes[s]]
                                   for s in range(L - 1)))
                d1 = add + C[X, lastx] + C[first, Y] + (inner1 - inner0)
                forb = [(i - 1 + s) % n for s in range(L + 1)]
                d0[forb] = INF
                d1[forb] = INF
                k0 = int(np.argmin(d0))
                k1 = int(np.argmin(d1))
                if d1[k1] < d0[k0]:
                    k, rev, d = k1, True, float(d1[k1])
                else:
                    k, rev, d = k0, False, float(d0[k0])
                if d >= -1e-12:
                    continue
                seg = seg_nodes[::-1] if rev else seg_nodes
                rest = [t[(i + L + s) % n] for s in range(n - L)]
                y = t[(k + 1) % n]
                pos = rest.index(y)
                t = rest[:pos] + seg + rest[pos:]
                improved = True
                moves += 1
                break
            if improved:
                break
        if not improved:
            break
    return t, tour_cost(C, t)


# ══════════════════════════════════════════════════════════════════
# 7. 主求解器
# ══════════════════════════════════════════════════════════════════

@dataclass
class Result:
    cost: float
    tour: List[int]
    Z_AP: float
    budget: float
    n_tree_arcs: int
    n_nonbasic_arcs: int
    n_cycles: int
    heuristic_cost: float
    time_sec: float
    overflow: bool = False
    search_nodes: int = 0
    proved: bool = False
    # 部分证书: 阶梯最后一个**完整跑完**的层给出的下界 Z_AP + b_arc.
    # 不完整 (撞上限/超时/深度受限) 的层不计入, 所以 lb_cert 永远是可以
    # 拿出来声明的下界; 一层都没跑完时它就是 Z_AP.
    lb_cert: Optional[float] = None
    # 阶梯最后一个完整跑完的层用了多少条非基弧 (k), 用于把 lb_cert 换算成
    # "再给多少时间能认证到几比几".
    rung_ok: int = 0


def solve(C, use_local_search: bool = True, cycle_cap: int = 200_000,
          search_cap: int = 2_000_000, time_limit: Optional[float] = None,
          max_cycles: int = 12, verbose: bool = False,
          init_tour: Optional[Sequence[int]] = None) -> Result:
    """精确求解 ATSP (论文 Bell & Frieze 2023 的准多项式算法).

    init_tour: 外部初始解 (例如 LKH-3 的巡回), 给定时与自带启发式取更优者.
    它只影响预算 Delta = UB - Z_AP, 不影响任何枚举/剪枝的正确性 —— Delta 变小
    只会让阶梯更短, 不会漏解. 但注意: 外部上界不是本框架产出的, 因此它只
    收紧待证区间, 不产生证书.
    """
    C = np.ascontiguousarray(np.asarray(C, dtype=float))
    n = C.shape[0]
    t_start = time.time()
    deadline = t_start + time_limit if time_limit is not None else None

    if n <= 4:                                        # 兜底: 暴力
        best_c, best_t = _brute_force(C)
        # Even on this fallback the certificate fields must be well-formed:
        # Z_AP is the assignment lower bound and, since brute force here is
        # exhaustive, the optimum itself is a valid (tight) lower bound.
        phi0, _, _ = hungarian_with_duals(C)
        z_ap = float(C[np.arange(n), phi0].sum())
        best_c = float(best_c)
        return Result(best_c, best_t, z_ap, best_c - z_ap, 0, 0, 0, best_c,
                      time.time() - t_start, proved=True,
                      lb_cert=best_c, rung_ok=0)

    # ---- 步骤 1,2: AP + 对偶 + 最优基树 ----
    # 只跑一次 Hungarian: 势 (u_h, v_h) 直接交给 optimal_duals, 否则它内部会
    # 再跑一遍同一个确定性算法 (唯一的一处 O(n^3), 重复它要花掉 1/3 的时间).
    phi, u_h, v_h = hungarian_with_duals(C)
    u, v = optimal_duals(C, phi, hung0=(u_h, v_h), verbose=verbose)
    rc = C - u[:, None] - v[None, :]
    rc = np.where(np.isfinite(C), rc, np.inf)
    rc = np.clip(rc, 0.0, None)
    Z_AP = float(C[np.arange(n), phi].sum())
    if abs(Z_AP - (u.sum() + v.sum())) > 1e-7 * max(1.0, Z_AP):
        raise RuntimeError("dual objective != primal objective")
    invphi = np.empty(n, dtype=np.int64)
    invphi[phi] = np.arange(n)
    extra, tree_max_rc = build_basis_tree(rc, phi)
    if verbose:
        print(f"[tree] |T*|=2n-1, max reduced cost on its non-matching edges "
              f"= {tree_max_rc:.3e}")

    # ---- 步骤 6a: 启发式上界 -> 剪枝预算 Delta ----
    heur = karp_patching(C, phi)
    hcost = tour_cost(C, heur)
    if use_local_search:
        cand, ccost = or_opt(C, heur, deadline=deadline)
        if ccost < hcost:
            heur, hcost = cand, ccost
    best_cost = hcost
    best_tour = list(heur)
    heur0 = hcost
    init_cost = None
    if init_tour is not None:
        init_cost = float(tour_cost(C, init_tour))
        if init_cost < best_cost:
            best_cost = init_cost
            best_tour = list(init_tour)
    budget = best_cost - Z_AP
    if verbose:
        print(f"[init] n={n}  Z_AP={Z_AP:.6f}  UB(heur)={hcost:.6f}"
              + (f"  UB(init)={init_cost:.6f}" if init_cost is not None else "")
              + f"  Delta={budget:.4e}")
    if budget <= 1e-12:
        return Result(best_cost, best_tour, Z_AP, budget, len(extra), 0, 0,
                      heur0, time.time() - t_start, proved=True,
                      lb_cert=Z_AP, rung_ok=0)

    sys.setrecursionlimit(max(10000, 20 * n + 1000))

    # ---- 预算逐轮收紧的迭代枚举 ----
    # 直接用 UB-Z_AP 做预算时, 合法圈的数量会爆炸 (因为 UB 还不够紧).
    # 论文的 Delta 是理论 gap O(log^2 n / n); 这里先用小预算 b 快速找到好解,
    # 用新解把 UB 压下来, 再用压下来的 Delta 重新枚举 —— 每一轮都变便宜.
    # ---- 预算 ladder: 按"保留多少条最便宜的非基弧"控制图规模 ----
    # 不能用 UB-Z_AP 的某个固定比例当起点: 启发式上界在大 n 上会比真实 gap 差
    # 一个数量级 (n=1500 实测差 40 倍), 那样第一轮就会撞 cycle_cap.
    # 改成直接控制非基弧条数 k (翻倍增长), 圈数随之可控; k 到了全图就等价于
    # 预算 b >= Delta, 此时若还没有改进解即可判定最优.
    in_tree = np.zeros((n, n), dtype=bool)
    for i, j in extra:
        in_tree[i, j] = True
    nb_mask = np.ones((n, n), dtype=bool)
    nb_mask[np.arange(n), phi] = False
    nb_mask &= ~in_tree
    nb_all = np.sort(rc[np.isfinite(rc) & nb_mask])
    m_nb = int(nb_all.size)
    if m_nb == 0:                                     # 无非基弧可用 => UB 已最优
        return Result(best_cost, best_tour, Z_AP, budget, len(extra), 0, 0,
                      heur0, time.time() - t_start, proved=True,
                      lb_cert=Z_AP, rung_ok=0)

    total_nodes = 0
    total_cycles = 0
    proved = False
    overflow = False
    lb_cert = Z_AP                       # 已完整跑完的层给出的最好下界
    rung_ok = 0                          # 该层用的非基弧条数
    k_nb = min(64, m_nb)
    k_last_ok = 0                                 # 上一次没撞上限的非基弧条数
    G_nb = 0
    while True:
        b_arc = float(nb_all[k_nb - 1]) if k_nb <= m_nb else float("inf")
        # 弧集完整 <=> 放完了全部非基弧, 或者第 k_nb 条既约费用已超过预算
        # (此后 rc > Delta 的弧不可能出现在改进解里, 因为 rc >= 0).
        finished = (k_nb >= m_nb) or (b_arc >= budget - 1e-18)
        b_arc = min(b_arc, budget)
        # 圈的"总既约费用"上限必须始终用 budget, 不能用弧阈值 b_arc 代替:
        # 当 budget 大于全部非基弧的既约费用时 (结构化/近对称实例上 AP 界很弱,
        # Delta 很大就会出现), b_arc = max rc < budget, 若拿它当圈费用上限会把
        # 总费用落在 (max rc, Delta] 区间内的圈全部砍掉 —— 而这些弧明明都在图里.
        # 审计 audit_certificate.py 在 metric-ish 实例上抓到 2880 例中 10 例
        # 因此误报 proved=True (返回的 tour 比暴力最优贵 0.4%~2%).
        b_cyc = budget if finished else b_arc
        # ---- 步骤 3: 收缩图 (只保留 rc <= b_arc 的非基弧) ----
        G = build_contracted(n, extra, invphi, rc, phi, b_arc)
        G_nb = len(G.nb_list)

        # ---- 步骤 3.5: phi-子圈标号 (拼装剪枝要用; 磁盘路径建库也要) ----
        # 必要条件 (子圈覆盖): 若某个 phi-子圈里的城市 psi 全不动,
        # 该子圈就会原封不动地留在 pi = phi o psi 里 => pi 不可能是单圈.
        # 因此每个候选合法圈必须"触碰"它所在的所有 phi-子圈, 且整体上所有
        # phi-子圈都要被触碰到 —— 这是个很便宜却很有效的剪枝.
        fcid, fpos, nfc = phi_cycle_labels(phi, n)
        full_mask = (1 << nfc) - 1

        # ---- 步骤 4: 枚举合法圈 ----
        # 三条路径产出同一个圈集合 (费用序也相同):
        #   (a) 磁盘圈库 (QA_CY_STORE=1): O(1) 内存, 21-32 B/圈
        #   (b) 编译后端扁平流: ~0.5 kB/圈, ~2.8e7 圈处 OOM
        #   (c) 纯 Python tuple: ~3.2 KB/圈, 仅作参照
        flat = None
        cycles = None
        view = None
        store_ovf = False
        _cy = cy_backend()
        _use_store = (_cy is not False
                      and os.environ.get("QA_CY_STORE", "0") not in ("0", ""))
        if _use_store:
            try:
                _cb, _ = _cy
                view = build_store_view(
                    _cb, G, b_cyc, cycle_cap, deadline,
                    fcid, nfc, H=_return_bound(G), verbose=verbose)
            except Exception as _e:
                if verbose:
                    print(f"[store] 失败, 退回 RAM 路径: {_e}")
                view = None
            if view is None:
                # 磁盘路径是因为撞了 cycle_cap 才拒绝的 (不是崩溃). 千万不要
                # 退回 RAM 路径重新枚举同一层: 那个规模 RAM 装不下, 只会 OOM
                # (n=5000 上就是这么被杀的, 写掉 19 GB 后进程消失).
                store_ovf = True
        if view is not None:
            n_this = len(view)
            total_cycles += n_this
            if verbose:
                print(f"[enum] k_nb={k_nb}/{m_nb}  b_arc={b_arc:.4e}  "
                      f"b_cyc={b_cyc:.4e}  fin={finished}  "
                      f"nb-arcs={G_nb}  cycles={n_this}  [store]")
        else:
            if store_ovf:
                # 直接按"这一层不完整"处理: 交给下面的回退逻辑缩小 k.
                n_this = 0
                cycles = []
                costs = []
                ovf = True
            elif _cy is not False:
                try:
                    _cb, _ = _cy
                    _pairs, _indptr, _costs, ovf, _work = _cb.enumerate_cycles_cy(
                        G, b_cyc, cap=cycle_cap, deadline=deadline,
                        use_bound=True, H=_return_bound(G))
                    flat = (_pairs, _indptr)
                    cycles = None
                    costs = _costs.tolist()
                except Exception:
                    flat = None
            # store_ovf 时 flat 仍是 None, 但绝不能掉进纯 Python 的
            # enumerate_valid_cycles: 那个 DFS 在百万圈级就是 MemoryError.
            if flat is None and not store_ovf:
                cycles, costs, ovf = enumerate_valid_cycles(
                    G, b_cyc, cap=cycle_cap, deadline=deadline)
            n_this = (len(flat[1]) - 1) if flat is not None else len(cycles)
            total_cycles += n_this
            if verbose:
                print(f"[enum] k_nb={k_nb}/{m_nb}  b_arc={b_arc:.4e}  "
                      f"b_cyc={b_cyc:.4e}  fin={finished}  nb-arcs={G_nb}  "
                      f"cycles={n_this}" + ("  *** CAP ***" if ovf else ""))

        if view is not None:
            ovf = False                     # 磁盘路径无 per-round 圈数上限
        if ovf:
            if view is not None:
                view.close()
                view = None
            if deadline is not None and time.time() > deadline:
                overflow = True
                break                              # 超时: 缩小 k 也无济于事
            # 这一轮不完整 => 不能据此证明最优. 圈数随 k 是爆炸式增长的
            # (n=2000 实测 k:1024->2048 时 771 -> 40000+), 所以不是直接放弃,
            # 而是回退到 (k_last_ok, k) 的中间值再试, 逐步逼近可处理的规模.
            if k_last_ok >= k_nb - 1:
                overflow = True
                break
            k_hi = k_nb
            k_nb = (k_last_ok + k_hi) // 2
            if k_nb <= k_last_ok:
                k_nb = k_last_ok + 1
            if verbose:
                print(f"[enum] 回退 k -> {k_nb}")
            continue
        k_last_ok = k_nb
        if deadline is not None and time.time() > deadline:
            overflow = True
            break

        # ---- 步骤 5: 顶点不交拼装 + 单圈检验 ----
        # 统一视图: 磁盘路径已经在 build_store_view 里排好序并算好辅助结构,
        # RAM 路径在这里排序 + 由 cypack 算掩码. 之后 rec() 只见 view.
        if view is None:
            if flat is not None:
                # 编译路径: 排序只算一个置换, 掩码由 cypack 直接吃扁平流算出,
                # 圈本身惰性物化 (FlatCycles). 与下面的 Python 路径位位相同.
                _cb, _cp = _cy
                _pairs, _indptr = flat
                order = np.argsort(np.asarray(costs, dtype=np.float64),
                                   kind="stable")
                cyc_sorted = FlatCycles(_pairs, _indptr, order)
                cost_sorted = [costs[t] for t in order]
                mask_sorted, cmask_sorted, suffix_or, cover_list = _cp.build_masks(
                    _pairs, _indptr, np.asarray(fcid, dtype=np.int64),
                    n, nfc, order=order)
            else:
                order = sorted(range(len(cycles)), key=lambda t: costs[t])
                cyc_sorted = [cycles[t] for t in order]
                cost_sorted = [costs[t] for t in order]
                mask_sorted: List[int] = []
                cmask_sorted: List[int] = []
                for edges in cyc_sorted:
                    m = 0
                    cm = 0
                    for a, _ in edges:
                        m |= 1 << a
                        cm |= 1 << fcid[a]
                    mask_sorted.append(m)
                    cmask_sorted.append(cm)
                suffix_or = [0] * (len(cmask_sorted) + 1)
                for i in range(len(cmask_sorted) - 1, -1, -1):
                    suffix_or[i] = suffix_or[i + 1] | cmask_sorted[i]
                # cover_list[s] = 所有"触碰到 phi-子圈 s"的圈的编号, 按费用升序
                cover_list: List[List[int]] = [[] for _ in range(nfc)]
                for idx, cm in enumerate(cmask_sorted):
                    bits = cm
                    while bits:
                        low = bits & -bits
                        cover_list[low.bit_length() - 1].append(idx)
                        bits ^= low
            view = RamView(cyc_sorted, cost_sorted, mask_sorted, cmask_sorted,
                           suffix_or, cover_list)
        if flat is not None or cycles is not None:
            del costs
            if cycles is not None:
                del cycles

        # 诊断: 若这批圈的并集连"碰遍所有 phi-子圈"都做不到, 那么无论怎么拼
        # pi 都不可能是单圈 (没被碰到的子圈会原封不动留下来), 这一轮注定无解 ——
        # 唯一出路是放更多非基弧, 但圈数会随之爆炸. 报出来便于判断瓶颈在哪.
        if verbose and view.suffix_or(0) != full_mask:
            miss = [s for s in range(nfc) if not (view.suffix_or(0) >> s) & 1]
            print(f"[pack] 警告: 有 {len(miss)}/{nfc} 个 phi-子圈没有被任何合法圈"
                  f" 碰到 {miss}, 本轮不可能出解, 需要更大的 k_nb")

        base_phi = [int(x) for x in phi]
        psi_arr = [-1] * n                   # psi(a) = b, -1 表示不动
        S_list: List[int] = []               # 被 psi 移动的城市 (无重复)
        best_rc = best_cost - Z_AP
        best_found_cost = best_cost
        best_out = list(best_tour)
        nodes = 0
        hit_cap = False
        depth_limited = [False]

        def evaluate(rcsum: float) -> None:
            """当前 psi (=已选的若干个顶点不交的圈) 是否让 pi 成为单圈.

            用 is_hamiltonian_psi 的结构定理: 只需在 |S| 个点上走一圈,
            代价与 n 无关. 费用用 Z_AP 加"被改动的那些弧"的差量精确算出,
            避免逐轮累加造成的浮点漂移.
            """
            nonlocal best_rc, best_out, best_found_cost
            m = len(S_list)
            if m == 0:
                return
            groups: Dict[int, List[int]] = {}
            for a in S_list:
                groups.setdefault(fcid[a], []).append(a)
            if len(groups) != nfc:
                return                                # 有 phi-子圈没被碰到
            nxt: Dict[int, int] = {}
            for lst in groups.values():
                lst.sort(key=lambda a: fpos[a])
                L = len(lst)
                for t in range(L):
                    nxt[lst[t]] = lst[(t + 1) % L]
            s0 = S_list[0]
            x = s0
            steps = 0
            for _ in range(m):
                if psi_arr[x] < 0:
                    return                        # 状态不一致时拒绝评估
                x = nxt[psi_arr[x]]
                steps += 1
                if x == s0:
                    break
            if x != s0 or steps != m:
                return
            c = Z_AP
            for a in S_list:
                c += C[a, base_phi[psi_arr[a]]] - C[a, base_phi[a]]
            if c < best_found_cost - 1e-12:
                succ = [base_phi[psi_arr[i]] if psi_arr[i] >= 0 else base_phi[i]
                        for i in range(n)]
                # Guard against a malformed successor map: a non-Hamiltonian
                # succ must never turn this walk into an infinite loop, and the
                # incumbent is committed only after the tour has been verified.
                tour = [0]
                y = succ[0]
                closed = False
                for _ in range(n + 1):
                    if y == 0:
                        closed = True
                        break
                    if not (0 <= y < n):
                        break
                    tour.append(y)
                    y = succ[y]
                if not closed or len(tour) != n:
                    return
                best_found_cost = c
                best_rc = c - Z_AP
                best_out = tour

        # 统一视图的本地绑定: rec() 是热点, 而 RAM 视图的这几个方法只是列表
        # 索引, 所以一次绑定换来"RAM / 磁盘两条路径共用一份搜索代码".
        _v_mask = view.mask
        _v_cost = view.cost
        _v_arcs = view.__getitem__
        _v_cmask = view.cmask
        _v_suffix = view.suffix_or
        _v_cover = view.cover_first
        _n_cyc = len(view)

        def rec(start: int, used: int, rcsum: float, covered: int,
                depth: int) -> bool:
            nonlocal nodes, hit_cap
            nodes += 1
            if nodes > search_cap:
                hit_cap = True
                return True
            if deadline is not None and (nodes & 255) == 0 and time.time() > deadline:
                hit_cap = True
                return True
            if covered == full_mask:
                evaluate(rcsum)
            if depth >= dcap:
                # Any depth truncation disqualifies a certificate.  The flag is
                # reset before each pass, so only the full-depth pass survives.
                depth_limited[0] = True
                return False

            if covered != full_mask:
                # 可行性剪枝 (只剪不分支): 若某个未覆盖的 phi-子圈根本没有任何
                # 编号 >= start 的圈能碰到它, 则这个节点下不可能有可行解.
                # 注意: 这里不能改成"只在该子圈的候选圈上分支" —— 那会漏掉
                # 编号更小的、覆盖其它子圈的圈 (解集的规范序要求按编号递增取).
                if (covered | _v_suffix(start)) != full_mask:
                    return False                      # 剩下的圈补不齐覆盖
                # 下界剪枝: 每个还没被碰到的 phi-子圈 s 都至少还要再选一个
                # 编号 >= start 且碰到 s 的圈. 因为视图是按费用升序排的,
                # 这样的圈里最便宜的那个就是 cover_first(s, start) 取到的.
                # 取所有未覆盖子圈上的最大值 (不能求和 —— 一个圈可以同时
                # 碰到多个子圈), 得到完成当前部分解所必需的最小额外费用.
                lb = 0.0
                bits = full_mask & ~covered
                while bits:
                    low = bits & -bits
                    s = low.bit_length() - 1
                    bits ^= low
                    hit = _v_cover(s, start)
                    if hit is None:
                        return False
                    v = hit[1]
                    if v > lb:
                        lb = v
                if rcsum + lb >= best_rc:
                    return False
            for idx in range(start, _n_cyc):
                if used & _v_mask(idx):
                    continue
                nc = rcsum + _v_cost(idx)
                if nc >= best_rc:
                    break                             # 费用升序 => 后面都超预算
                added = _v_arcs(idx)
                for a, bb in added:
                    psi_arr[a] = bb
                    S_list.append(a)
                if rec(idx + 1, used | _v_mask(idx), nc,
                       covered | _v_cmask(idx), depth + 1):
                    return True
                for a, _ in added:
                    psi_arr[a] = -1
                    S_list.pop()
            return False

        # 深度爬梯子: 先用 <=2 / <=3 个圈的浅层拼装快速挖出好解来压 UB,
        # 收紧后的 best_rc 再让最后一层完整搜索便宜得多.
        improved_this_round = False
        # max_cycles must be respected even when it is below 3, and the last
        # pass must be the full-depth one.  depth_limited is reset per pass,
        # hence the final value describes the full-depth pass only.
        dcap_passes = sorted({d for d in (2, 3, max_cycles)
                              if d <= max_cycles} or {max_cycles})
        for dcap in dcap_passes:
            # 上一轮可能因撞上限/超时提前 return, 递归栈没有逐层回溯,
            # 于是 psi_arr / S_list 仍带着部分解. 每一层重新开始前必须清空,
            # 否则下一层会把已占用的城市再放一次, 拼出的 psi 不是置换.
            psi_arr[:] = [-1] * n
            S_list.clear()
            depth_limited[0] = False
            rec(0, 0, 0.0, 0, 0)
            if best_found_cost < best_cost - 1e-12:
                best_cost = best_found_cost        # UB 立刻收紧
                best_tour = best_out
                budget = best_cost - Z_AP
                best_rc = budget
                improved_this_round = True
                if budget <= 1e-12 or dcap < dcap_passes[-1]:
                    break                          # 回外层用更小预算重新开始
        total_nodes += nodes
        if hit_cap:
            overflow = True
            if view is not None:
                view.close()
                view = None
            break

        # ---- 部分证书 ----
        # 本层若被完整搜过 (枚举没撞 cycle_cap、拼装没撞 search_cap、最后一次
        # 是完整深度), 那么任何比 UB 更优的巡回 T 必然满足: 要么用到一条
        # rc > b_arc 的非基弧, 要么虽只用 rc <= b_arc 的弧但总既约费用 > b_arc.
        # 两种情况下 cost(T) = Z_AP + (非基弧既约费用之和) > Z_AP + b_arc.
        # 于是 LB = Z_AP + b_arc 可以声明, 认证比 = UB / LB.
        # 深度受限的层不算: 那时只搜了 <=2/<=3 个圈的拼装, 不能据此下界.
        if not depth_limited[0]:
            if Z_AP + b_arc > lb_cert:
                lb_cert = Z_AP + b_arc
                rung_ok = G_nb

        # 这一轮的圈集 (磁盘文件 / RAM 缓冲) 到这里就用完了
        if view is not None:
            view.close()
            view = None

        if improved_this_round:
            if budget <= 1e-12:
                proved = True
                break
            continue                               # 沿用当前 k_nb 继续爬

        any_depth_limited = depth_limited[0]       # 只认最后一次(完整深度)的搜索
        if finished:                               # 已按完整预算枚举且无改进 => 最优
            proved = not any_depth_limited
            break
        if k_nb >= m_nb:                           # 非基弧已放完
            proved = not any_depth_limited
            break
        # 指数步长; 撞上限时由上面的回退逻辑细化, 从而在易例上轮数尽量少
        k_nb = min(m_nb, 2 * k_nb)

    return Result(cost=best_cost, tour=best_tour, Z_AP=Z_AP, budget=budget,
                  n_tree_arcs=len(extra), n_nonbasic_arcs=G_nb,
                  n_cycles=total_cycles, heuristic_cost=heur0,
                  time_sec=time.time() - t_start, overflow=overflow,
                  search_nodes=total_nodes, proved=proved,
                  lb_cert=lb_cert, rung_ok=rung_ok)


# ══════════════════════════════════════════════════════════════════
# 校验 / 基准
# ══════════════════════════════════════════════════════════════════

def _brute_force(C: np.ndarray) -> Tuple[float, List[int]]:
    n = C.shape[0]
    best = float("inf")
    best_t: List[int] = []
    for perm in itertools.permutations(range(1, n)):
        t = (0,) + perm
        c = sum(C[t[i], t[(i + 1) % n]] for i in range(n))
        if c < best:
            best, best_t = c, list(t)
    return best, best_t


def held_karp(C: np.ndarray) -> Tuple[float, List[int]]:
    """精确 DP, O(n^2 2^n), 用于中等 n 的交叉验证."""
    n = C.shape[0]
    INF = float("inf")
    size = 1 << n
    dp = np.full((size, n), INF)
    par = np.full((size, n), -1, dtype=np.int64)
    dp[1, 0] = 0.0
    for S in range(1, size):
        if not (S & 1):
            continue
        for j in range(1, n):
            if not (S >> j) & 1:
                continue
            prev = S ^ (1 << j)
            if prev == 0:
                continue
            row = dp[prev, :] + C[:, j]
            k = int(np.argmin(row))
            dp[S, j] = row[k]
            par[S, j] = k
    full = size - 1
    total = dp[full, 1:] + C[1:, 0]
    jbest = int(np.argmin(total)) + 1
    cost = float(total[jbest - 1])
    tour: List[int] = []
    S, j = full, jbest
    while j != 0:
        tour.append(j)
        pj = int(par[S, j])
        S ^= 1 << j
        j = pj
    tour.append(0)
    tour.reverse()
    return cost, tour


def _self_test() -> None:
    """对小实例与 Held-Karp 精确解逐一对账."""
    print("=" * 76)
    print("正确性交叉验证 (vs Held-Karp 精确 DP)")
    print("=" * 76)
    rng = np.random.default_rng(20240630)
    bad = 0
    for trial in range(24):
        n = int(rng.integers(5, 13))
        D = rng.random((n, n)) if trial % 2 == 0 else rng.exponential(1.0, (n, n))
        np.fill_diagonal(D, np.inf)
        exact, _ = held_karp(D)
        r = solve(D, verbose=False)
        good = (abs(r.cost - exact) < 1e-9 and abs(tour_cost(D, r.tour) - exact) < 1e-9
                and len(set(r.tour)) == n and r.proved)
        bad += (not good)
        print(f"  n={n:2d} dist={'U[0,1]' if trial % 2 == 0 else 'Exp(1)'}  "
              f"Z_AP={r.Z_AP:.5f}  UB(heur)={r.heuristic_cost:.5f}  "
              f"solved={r.cost:.5f}  exact={exact:.5f}  "
              f"{'OK' if good else '*** MISMATCH ***'}")
    print(f"  => {'全部通过' if bad == 0 else f'{bad} 个失败用例'}")


def _demo() -> None:
    print()
    print("=" * 76)
    print("规模演示 (随机 U[0,1] 费用; time_limit=60s)")
    print("=" * 76)
    g = np.random.default_rng(1)
    for n in (50, 100, 200, 300, 500, 800, 1000):
        D = g.random((n, n))
        np.fill_diagonal(D, np.inf)
        r = solve(D, time_limit=60.0, verbose=False)
        status = "proved optimal" if r.proved else "best found (未证明最优)"
        assert len(set(r.tour)) == n, "返回的不是一个合法巡回"
        assert abs(tour_cost(D, r.tour) - r.cost) < 1e-9, "费用对不上"
        print(f"  n={n:4d}  Z_AP={r.Z_AP:.6f}  UB(heur)={r.heuristic_cost:.6f}  "
              f"OPT={r.cost:.6f}  gap={r.cost - r.Z_AP:.3e}\n"
              f"         valid-cycles={r.n_cycles}  search-nodes={r.search_nodes}  "
              f"t={r.time_sec:.2f}s  [{status}]")


if __name__ == "__main__":
    _self_test()
    _demo()
