"""Runtime scaling fit for the corrected Bell-Frieze solver.
   Compares  t = a * n^b            (polynomial)
   against   t = exp(c*(ln n)^2 + d)(quasi-polynomial, the theoretical form)
   on log-log least squares, i.i.d. U[0,1] costs (the Bell-Frieze regime).
"""
import os
import sys, time, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "solver"))
import numpy as np
import quasipoly_atsp as qa

SIZES = [20, 30, 50, 80, 100, 150, 200, 300, 500, 800]
SEEDS = [1, 2, 3]
TL = 45.0
# 单次计时在小 n 上几乎全是 OS 抖动 (实测 n=20 的中位数在两次运行间能差 40%),
# 拟合出的指数和 R^2 因此不可复现. 改成重复计时取最小值 (best-of), 这是噪声下的
# 标准估计量; 慢实例本身已经跑了几秒, 单次即可, 否则总时长会失控.
REPS_FAST, FAST_CUTOFF = 5, 1.0

rows = []
for n in SIZES:
    for s in SEEDS:
        rng = np.random.default_rng(1000 * n + s)
        C = rng.random((n, n))
        np.fill_diagonal(C, np.inf)
        ts = []
        for rep in range(REPS_FAST):
            t0 = time.perf_counter()
            r = qa.solve(C, time_limit=TL)
            ts.append(time.perf_counter() - t0)
            if ts[-1] >= FAST_CUTOFF:
                break
        el = min(ts)
        rows.append(dict(n=n, seed=s, t=el, reps=len(ts), proved=bool(r.proved),
                         cost=float(r.cost), Z_AP=float(r.Z_AP),
                         nodes=int(r.search_nodes), cycles=int(r.n_cycles)))
        print(f"n={n:4d} seed={s}  t={el:9.4f}s (best of {len(ts)})  "
              f"proved={bool(r.proved)}  "
              f"cost={r.cost:.6f} Z_AP={r.Z_AP:.6f} gaps={100*(r.cost-r.Z_AP)/r.Z_AP:6.3f}%  "
              f"nodes={r.search_nodes}", flush=True)

json.dump(rows, open(os.path.join(_ROOT, "results", "fit_runtime.json"), "w"), indent=1)

ok = [r for r in rows if r["proved"]]
print(f"\nproved {len(ok)}/{len(rows)}")

def r2(y, yhat):
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - (y.mean())) ** 2)


def fit(agg):
    """agg: 'mean' | 'median'; 返回 (b, ln a, c, d, r2_poly, r2_quasi, ns, vals)"""
    ns = sorted(set(r["n"] for r in ok))
    nm = np.array(ns, float)
    yv = np.array([(np.mean if agg == "mean" else np.median)(
        [r["t"] for r in ok if r["n"] == k]) for k in ns])
    lx, ly = np.log(nm), np.log(yv)
    b, la = np.polyfit(lx, ly, 1)
    u = lx ** 2
    c, d = np.polyfit(u, ly, 1)
    return b, la, c, d, r2(ly, la + b * lx), r2(ly, c * u + d), ns, yv


print()
for agg in ("mean", "median"):
    b, la, c, d, rp, rq, ns, yv = fit(agg)
    print(f"[{agg:6s}] poly   t = {np.exp(la):.3e} * n^{b:.3f}   R^2 = {rp:.4f}")
    print(f"[{agg:6s}] quasi  ln t = {c:+.4f} (ln n)^2 {d:+.4f}    R^2 = {rq:.4f}")
    print(f"{'-'*58}\n{'n':>5} {'t (s)':>10} {'poly fit':>12} {'quasi fit':>12}")
    for k, t in zip(ns, yv):
        print(f"{int(k):5d} {t:10.4f} {np.exp(la)*k**b:12.4f} "
              f"{np.exp(c*np.log(k)**2+d):12.4f}")
    print()
