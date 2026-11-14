# Implementation notes

The theoretical recipe leaves several things to the implementer. Three of them change
the output, not just the speed, and the certificate depends on getting them right.
This note records what the code does and why; the inline comments in
`quasipoly_atsp.py` carry the longer version.

## 1. Tree membership is tested on the bipartite index

The assignment instance is the bipartite graph `K_{A,B}` with row nodes
`a_0..a_{n-1}` and column nodes `b_0..b_{n-1}`. Arc `(i,j)` of the ATSP is the edge
`{i, j+n}`. The basis tree `T*` is a spanning tree of that graph, so membership must
be tested as `(i, j+n) in T*`.

Testing `(i,j)` instead makes every arc look non-basic: the subsequent reduced-cost
pruning then removes nothing, the enumeration degenerates, and the solver still
returns feasible tours that are a few percent suboptimal. Nothing crashes. This is
the failure mode that makes the "bipartite index" choice a correctness issue rather
than a style issue.

## 2. The dual must be an extreme point of the optimal face

The assignment LP is highly degenerate, so its optimal dual is a *face*, not a point.
The potentials returned by a Hungarian/Jonker–Volgenant routine need not be a vertex
of that face, and their tight-arc set can be too small to contain a spanning tree.
At `n = 200` one instance had 195 tight arcs where 199 non-matching tree arcs are
needed.

`optimal_duals()` therefore does one of three things, in order:

1. accept the Hungarian potentials if the tight arcs already span;
2. otherwise repair them by a shortest-path computation on the difference
   constraints `w_i - w_k <= rc(i, phi(k))`, which lands on a vertex of the face in
   `O(n^2 log n)` (79 s -> 0.2 s at `n = 800` compared with solving the dual LP);
3. fall back to the dual LP only if the repair fails.

The solver then asserts `rho = max_{(i,j) in T*} rc_ij <= 1e-9`. This is what
licenses pricing every tree arc at zero inside the enumeration; without it the
identity `cost(H) = Z_AP + sum_{H \ M*} rc` is off by a small amount and the
certificate is invalid.

## 3. 2-opt is not a valid move on the ATSP

Reversing a segment also reverses the direction of every arc inside it, so the cost
of the segment changes and the usual symmetric-TSP delta is wrong. The local search
here is an ATSP Or-opt: it moves segments and recomputes the delta in the `s -> t`
direction. All experiments are on non-metric matrices, where this matters.

## 4. The budget ladder, and when a certificate may be issued

`Delta = UB - Z_AP`. Any improving tour uses only non-basic arcs of reduced cost
`< Delta`, so a round that keeps the `k` cheapest non-basic arcs is *complete* when
`k >= m` or `rc_(k) >= Delta`, where `m` is the number of finite non-basic reduced
costs. Only a complete round may issue `proved=True`.

On a complete round the per-cycle cap must be `Delta` itself, not the arc threshold
`b_arc` used to build the contracted graph; conflating the two silently discards
cycles whose total reduced cost lies in `(max rc, Delta]`, which is exactly the
regime where the assignment bound is weak.

Three hard caps bound the search: 12 cycles per packing, `2e5` cycles per round,
`2e6` nodes per run, plus the wall-clock deadline. Any of them binding sets
`proved=False`. The solver never returns a patched tour as optimal: patching supplies
the incumbent `UB`, and a patched tour carries no certificate because an arc
introduced by patching lies outside the enumerated set.

A cap that fires aborts the recursion without unwinding it level by level, so the
partial packing stays in `psi`/`S`. Each depth pass therefore resets both before it
starts; without that reset the next pass can place an already-used city a second
time and the reconstructed `psi` is no longer a permutation.

## 5. What is optional

* **Cython kernel** (`solver/cy/`): the DFS and the packing pre-processing are
  reimplemented in Cython. The kernel is validated element-by-element against the
  Python path (`validate_cyenum.py`, `validate_cypack.py`) and is bit-identical in
  every output it feeds. Disable it with `QA_NO_CY=1`.
* **Disk-backed cycle store** (`QA_CY_STORE=1`): stores cycles as move sequences
  (21–37 bytes each) instead of Python tuples. It is bit-identical to the RAM path on
  cost, arc list, vertex mask and subtour mask, and moves the memory ceiling from RAM
  to disk.
* **Compiled enumeration** is not required for any number in the paper's main tables;
  the interpreted path produces the same tours and certificates.

## 6. Numerical tolerances

Dual feasibility, zero duality gap and tree-arc tightness are checked at `1e-9` on
every instance. The certificate audit in `scripts/audit_certificate.py` measures the
worst violations over 2 880 instances; they stay six orders of magnitude inside the
tolerance. The audit also checks the one condition that is not an equality: that
`proved=True` implies the returned cost equals the brute-force optimum.
