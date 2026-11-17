"""Measure base-vs-gate inference overhead with repeats and CIs.
Methodology: per config, 3 seeds; per seed: 3 warmup + 500 timed forward
passes (batch 128, seq len 50); report mean +/- 95% CI across seeds.
Run from repo root on the (oldpy) env: python3 measure_resources3.py"""
import sys, json, time, statistics as st
import numpy as np, torch
from scipy import stats as stt
from XlNet_test.configs import args as args0
from XlNet_test.src.utils import set_up_data_loader
from XlNet_test.src.train_async import initiate

def time_config(use_asynchrony, tag, seeds=(42, 123, 2024), reps=500):
    per_seed, mems = [], []
    for seed in seeds:
        sys.argv = ["m", "--dataset", "mosi", "--seed", str(seed),
                    "--use_asynchrony", str(use_asynchrony).lower(),
                    "--use_corruption", "false", "--chaos_source", "zeros"]
        import importlib, XlNet_test.configs as cfg
        importlib.reload(cfg)
        a = cfg.args
        a.device = torch.device("cuda:0")
        torch.manual_seed(seed)
        np.random.seed(seed)
        s = initiate(a, num_train_optimization_steps=1)
        model = s["model"].to(a.device).eval()
        _, _, sl, _ = set_up_data_loader(a)
        batch = next(iter(sl))
        input_ids, visual, acoustic, input_mask, segment_ids, _ = [t.to(a.device) for t in batch]
        visual = visual.squeeze(1)
        acoustic = acoustic.squeeze(1)
        chaos = torch.zeros(input_ids.size(0), device=a.device) if use_asynchrony else None
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids, visual, acoustic, token_type_ids=segment_ids,
                          attention_mask=input_mask, chaos_score=chaos)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                _ = model(input_ids, visual, acoustic, token_type_ids=segment_ids,
                          attention_mask=input_mask, chaos_score=chaos)
            torch.cuda.synchronize()
        per_seed.append((time.perf_counter() - t0) / reps)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        with torch.no_grad():
            _ = model(input_ids, visual, acoustic, token_type_ids=segment_ids,
                      attention_mask=input_mask, chaos_score=chaos)
        mems.append(torch.cuda.max_memory_allocated() / 2**30)
        del model
        torch.cuda.empty_cache()
    m = st.mean(per_seed)
    h = stt.t.ppf(0.975, len(per_seed) - 1) * st.stdev(per_seed) / len(per_seed) ** 0.5
    return {"config": tag, "batch128_ms_mean": round(m * 1000, 2),
            "batch128_ms_95ci": round(h * 1000, 2),
            "per_seed_ms": [round(x * 1000, 2) for x in per_seed],
            "inference_peak_mem_GB_max": round(max(mems), 2)}

out = {"methodology": "3 warmup + 500 timed fwd passes, repeated over seeds 42/123/2024; RTX 3090",
       "base": time_config(False, "base_encoder"),
       "gate": time_config(True, "with_gate")}
json.dump(out, open("resource_report3.json", "w"), indent=2)
print(json.dumps(out, indent=2))
