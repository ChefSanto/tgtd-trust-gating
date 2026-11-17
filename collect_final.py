import json, glob, statistics as st
from scipy import stats
rows = []
for f in sorted(glob.glob("/workspace/runs_final/eval/eval_*.json")):
    if any(s in f for s in ("_mask", "_shuffled")): continue
    name = f.split("eval_")[1].replace(".json","")
    parts = name.split("_")
    d = json.load(open(f))
    rows.append(dict(ds=parts[0], ep=int(parts[1].replace("ep","")), src=parts[2],
                     seed=int(parts[3].replace("seed","")), clean=d["clean"],
                     text=d["text_corrupt"], sm=d["high_chaos_softmax"],
                     tw=d["gate_params"]["text_w"]))
def ci(v):
    m = st.mean(v); n = len(v)
    if n < 2: return m, 0.0
    return m, stats.t.ppf(0.975, n-1) * st.stdev(v) / (n ** 0.5)
for r in sorted(rows, key=lambda r:(r['ds'],r['ep'],r['src'],r['seed'])):
    print(f"{r['ds']:<6}{r['ep']:>4}ep {r['src']:<9} seed{r['seed']:<6} clean {r['clean']:.4f}  text {r['text']:.4f}   sm({r['sm'][0]:.2f},{r['sm'][1]:.2f},{r['sm'][2]:.2f})")
print("\n--- advantage (detector - zeros, text MAE) ---")
for ds in ("mosi","mosei"):
    for ep in (40,80,100,120):
        z = {r["seed"]: r for r in rows if r["ds"]==ds and r["ep"]==ep and r["src"]=="zeros"}
        d = {r["seed"]: r for r in rows if r["ds"]==ds and r["ep"]==ep and r["src"]=="detector"}
        common = sorted(set(z) & set(d))
        if len(common) >= 2:
            adv = [d[s]["text"] - z[s]["text"] for s in common]
            m, h = ci(adv); nf = sum(1 for a in adv if a < 0)
            print(f"{ds} {ep:>3}ep n={len(common):>2}  {m:+.4f} +/- {h:.4f}   {nf}/{len(common)} favor gate")
