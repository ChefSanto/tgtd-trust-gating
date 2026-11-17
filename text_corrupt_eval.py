"""Clean + text-corruption MAE for a final checkpoint, chaos source ACTIVE.
Corruption applied BEFORE the detector (matches train-time order).
Non-zero labels only. Flags: --mask_fraction (graded masking), --shuffle_chaos
(permutation ablation), --zero_audio/--zero_video (modality ablation).
NOTE (harness fix, 2026-11): the internal args now receive --chaos_source; the
pre-fix version defaulted it to 'detector', so uniform-fusion runs were
evaluated with chaos_score=None (dormant gate path). See paper Sect. 4.4."""
import argparse, json, sys
import numpy as np
import torch
from tqdm import tqdm

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="mosi")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--chaos_source", default="detector", choices=["detector", "zeros"])
p.add_argument("--model_path", required=True)
p.add_argument("--out", default="/tmp/eval.json")
p.add_argument("--mask_fraction", type=float, default=1.0)
p.add_argument("--shuffle_chaos", action="store_true")
p.add_argument("--zero_audio", action="store_true")
p.add_argument("--zero_video", action="store_true")
cli = p.parse_args()

sys.argv = ["text_corrupt_eval", "--dataset", cli.dataset,
            "--seed", str(cli.seed), "--use_asynchrony", "true",
            "--use_corruption", "false", "--chaos_source", cli.chaos_source]
from XlNet_test.configs import args, TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM
from XlNet_test.src.utils import set_up_data_loader
from XlNet_test.src.train_async import initiate, _resolve_chaos_source
from XlNet_test.src.asynchrony_detector import AsynchronyDetector

assert args.chaos_source == cli.chaos_source, f"PROPAGATION BUG: args={args.chaos_source} cli={cli.chaos_source}"
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(cli.seed); np.random.seed(cli.seed)
SPECIAL = torch.tensor([3, 4, 5])

def eval_pass(model, loader, detector, mask_fraction, shuffle_chaos):
    model.eval(); preds, labels = [], []
    g = torch.Generator(device=args.device).manual_seed(cli.seed)
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = visual.squeeze(1); acoustic = acoustic.squeeze(1)
            if cli.zero_audio: acoustic = torch.zeros_like(acoustic)
            if cli.zero_video: visual = torch.zeros_like(visual)
            if mask_fraction > 0:
                is_special = torch.isin(input_ids, SPECIAL.to(input_ids.device))
                mask = (torch.rand(input_ids.shape, generator=g,
                                   device=args.device) < mask_fraction) & (~is_special)
                input_ids = torch.where(mask, torch.full_like(input_ids, 6), input_ids)
            chaos = None
            if detector is not None:
                text_emb = model.transformer.word_embedding(input_ids)
                chaos = detector(text_emb, acoustic, visual, input_mask)
                if shuffle_chaos:
                    perm = torch.randperm(chaos.size(0), generator=g, device=args.device)
                    chaos = chaos[perm]
            chaos = _resolve_chaos_source(args, chaos, None,
                                          input_ids.size(0), args.device)
            logits, _ = model(input_ids, visual, acoustic,
                              token_type_ids=segment_ids,
                              attention_mask=input_mask, chaos_score=chaos)
            preds.extend(np.squeeze(logits.cpu().numpy()).tolist())
            labels.extend(np.squeeze(label_ids.cpu().numpy()).tolist())
    preds, labels = np.array(preds), np.array(labels)
    nz = labels != 0
    return float(np.mean(np.abs(preds[nz] - labels[nz])))

print(f"[eval] {cli.dataset} seed={cli.seed} src={cli.chaos_source} "
      f"mask={cli.mask_fraction} shuffle={cli.shuffle_chaos} "
      f"za={cli.zero_audio} zv={cli.zero_video} ckpt={cli.model_path}")
train_loader, _, test_loader, _ = set_up_data_loader(args)
settings = initiate(args, num_train_optimization_steps=1)
model = settings["model"]
model.load_state_dict(torch.load(cli.model_path, map_location=args.device))
model.to(args.device).eval()

detector = None
if cli.chaos_source == "detector":
    detector = AsynchronyDetector(TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM,
                                projection_dim=128, detach_output=True,
                                learnable_weights=True).to(args.device)
    def _hook(input_ids, input_mask, _m=model):
        with torch.no_grad():
            return _m.transformer.word_embedding(input_ids)
    detector.fit(train_loader, text_hook=_hook, device=args.device)
    detector.eval()

res = {"dataset": cli.dataset, "seed": cli.seed, "chaos_source": cli.chaos_source,
       "ckpt": cli.model_path, "mask_fraction": cli.mask_fraction,
       "shuffle_chaos": cli.shuffle_chaos,
       "zero_audio": cli.zero_audio, "zero_video": cli.zero_video}
res["clean"] = eval_pass(model, test_loader, detector, 0.0, False)
res["text_corrupt"] = eval_pass(model, test_loader, detector, cli.mask_fraction, shuffle_chaos=cli.shuffle_chaos)
res["delta"] = res["text_corrupt"] - res["clean"]
gmod = model.transformer.AMG
res["gate_params"] = {"low": float(gmod.chaos_threshold_low_raw.item()),
                      "high": float(gmod.chaos_threshold_high_raw.item()),
                      "text_w": float(gmod.text_weight_raw.item()),
                      "audio_w": float(gmod.audio_weight_raw.item()),
                      "visual_w": float(gmod.visual_weight_raw.item())}
import torch.nn.functional as F
res["high_chaos_softmax"] = F.softmax(torch.tensor(
    [res["gate_params"][k] for k in ("text_w", "audio_w", "visual_w")]), dim=0).tolist()
print(json.dumps(res, indent=2))
json.dump(res, open(cli.out, "w"), indent=2)
print(f"[eval] wrote {cli.out}")
