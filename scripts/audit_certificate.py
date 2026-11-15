"""审计 proved=True 的可靠性: 证书到底依赖什么?

四组可机器检查的条件:
  C1 对偶可行     : min_{i,j} rc_ij >= -eps           (下界 Z_AP 合法)
  C2 零对偶间隙   : |sum(u)+sum(v) - Z_AP| <= eps     (分解式成立)
  C3 树弧全紧     : max_{(i,j) in T*} rc_ij <= eps    (枚举里树弧按 0 计价, 必须真的为 0)
  C4 完备性       : proved=True 时 cost == 暴力真值最优 (且从不 cost > OPT)

C4 是唯一无法靠等式检查、只能用独立真值验证的一条 —— 它正是
"交错圈分解是否穷尽了所有排列" 这条结构性引理的实证检验。
刻意使用退化实例(整数代价/大量并列)去压这条引理。
"""
import os
import sys, itertools, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "solver"))
import numpy as np
import quasipoly_atsp as qa

TOL = 1e-7


def brute_force_opt(C):
    n = C.shape[0]
    best = np.inf
    for p in itertools.permutations(range(1, n)):
        t = (0,) + p
        c = sum(C[t[k], t[(k + 1) % n]] for k in range(n))
        if c < best:
            best = c
    return float(best)


def gen(kind, rng, n):
    if kind == "uniform":
        C = rng.random((n, n))
    elif kind == "exponential":
        C = rng.exponential(1.0, (n, n))
    elif kind == "ties":          # 重度退化: 只有 4 种代价
        C = rng.integers(1, 5, (n, n)).astype(float)
    elif kind == "ties2":         # 极端退化: 0/1
        C = rng.integers(0, 2, (n, n)).astype(float)
    elif kind == "planted":       # 已知最优圈被人为压低
        C = rng.random((n, n))
        perm = rng.permutation(n)
        for k in range(n):
            C[perm[k], perm[(k + 1) % n]] *= 0.01
    elif kind == "metric-ish":
        P = rng.random((n, 2))
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        C = d + rng.random((n, n)) * 0.05
    np.fill_diagonal(C, np.inf)
    return C


def check_conditions(C):
    """返回 (C1,C2,C3) 三个数值指标"""
    from scipy.optimize import linear_sum_assignment
    _, phi = linear_sum_assignment(C)
    phi = phi.astype(int)
    Z = float(C[np.arange(len(phi)), phi].sum())
    u, v = qa.optimal_duals(C, phi)
    fin = np.isfinite(C)
    rc_full = np.where(fin, C - u[:, None] - v[None, :], np.inf)
    c1 = float(rc_full[fin].min())
    c2 = abs(float(u.sum() + v.sum()) - Z)
    rc = np.clip(rc_full, 0.0, None)
    extra, max_rc = qa.build_basis_tree(rc, phi)
    c3 = float(max_rc)
    return c1, c2, c3, Z


def main():
    kinds = ["uniform", "exponential", "ties", "ties2", "planted", "metric-ish"]
    ns = [6, 7, 8, 9]
    NSEED = 120
    # 确定性: 绝不用 hash() (PYTHONHASHSEED 随机化会让结果不可复现)

    stats = {k: {"n": 0, "proved": 0, "wrong": 0, "c1": [], "c2": [], "c3": [],
                 "worse": []} for k in kinds}
    global_wrong = []
    t0 = time.time()
    for ki, kind in enumerate(kinds):
        for n in ns:
            for s in range(NSEED):
                rng = np.random.default_rng(1_000_000 * (ki + 1) + 10_000 * n + s)
                C = gen(kind, rng, n)
                try:
                    c1, c2, c3, Z = check_conditions(C)
                except Exception as e:
                    print("  dual check raised", kind, n, s, e)
                    continue
                res = qa.solve(C, time_limit=20)
                opt = brute_force_opt(C)
                st = stats[kind]
                st["n"] += 1
                st["c1"].append(c1); st["c2"].append(c2); st["c3"].append(c3)
                # 交叉校验: 报告的 cost 必须等于它自己返回的 tour 的真实费用
                tc = qa.tour_cost(C, res.tour)
                if abs(tc - res.cost) > 1e-9:
                    global_wrong.append(("COST-MISMATCH", kind, n, s,
                                         res.cost, tc, opt))
                    st["wrong"] += 1
                if res.proved:
                    st["proved"] += 1
                    if abs(res.cost - opt) > 1e-7:
                        st["wrong"] += 1
                        global_wrong.append((kind, n, s, res.cost, opt,
                                             res.proved, c1, c2, c3, tc))
                st["worse"].append(res.cost - opt)
    print(f"耗时 {time.time()-t0:.1f}s\n")
    print(f"{'分布':13s} {'实例':>5} {'proved':>7} {'错误':>5} "
          f"{'C1 min rc':>12} {'C2 间隙':>11} {'C3 树弧max':>11} {'超优值':>10}")
    tot = totp = totw = 0
    for k in kinds:
        st = stats[k]
        tot += st["n"]; totp += st["proved"]; totw += st["wrong"]
        print(f"{k:13s} {st['n']:5d} {st['proved']:7d} {st['wrong']:5d} "
              f"{min(st['c1']):12.2e} {max(st['c2']):11.2e} "
              f"{max(st['c3']):11.2e} {max(st['worse']):10.2e}")
    print(f"{'合计':13s} {tot:5d} {totp:7d} {totw:5d}")
    print()
    print("结论: C4 违例(证明最优却不是最优) =", totw)
    if global_wrong:
        print("违例明细:", global_wrong[:10])
    # 最坏 C3
    allc3 = max(max(stats[k]["c3"]) for k in kinds)
    allc1 = min(min(stats[k]["c1"]) for k in kinds)
    allc2 = max(max(stats[k]["c2"]) for k in kinds)
    print(f"\nC1 最差 min rc       = {allc1:.3e}  (需 >= -1e-9)")
    print(f"C2 最差对偶间隙      = {allc2:.3e}  (需 ~0)")
    print(f"C3 最差树弧既约费用  = {allc3:.3e}  (需 ~0, 否则枚举计价偏小)")


if __name__ == "__main__":
    main()
