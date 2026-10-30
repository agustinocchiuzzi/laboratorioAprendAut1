"""Inference-only recording of fixed, predefined seeds. Never trains."""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import gymnasium as gym
import torch
from torch import nn
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/comparison_20261027T001236_958528Z'
OUT = ROOT / 'recordings' / datetime.now(timezone.utc).strftime('greedy_seed42_%Y%m%dT%H%M%S_%fZ')
OUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(1)
rows = []
for policy in ('A0', 'A1'):
    source = SOURCE / f'{policy}_seed42'
    config = json.loads((source / 'config.json').read_text())
    assert config['training_seed'] == 42
    dims = (8, *config['hidden_dims'], 4)
    layers = []
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims)-2:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    weights = source / 'online_weights.pt'
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    net.eval()
    for seed in (20000, 20001, 20002):
        stem = f'{policy}_train42_env{seed}_greedy'
        env = gym.make(config['environment'], continuous=config['continuous'],
                       enable_wind=config['enable_wind'], render_mode='rgb_array',
                       max_episode_steps=config['max_steps'])
        frames, trace = [], []
        obs, _ = env.reset(seed=seed)
        total = 0.0
        def frame(step, state):
            im = Image.fromarray(env.render())
            canvas = Image.new('RGB', (im.width, im.height+42), 'black')
            canvas.paste(im, (0,42))
            ImageDraw.Draw(canvas).text((8,4), f'{policy} | training seed 42 | env seed {seed} | GREEDY\nstep {step} | return {total:.2f} | {state}', fill='white')
            return canvas
        frames.append(frame(0, 'running'))
        try:
            with torch.inference_mode():
                for step in range(1, config['max_steps']+1):
                    action = int(net(torch.as_tensor(obs, dtype=torch.float32).reshape(1,-1)).argmax(1).item())
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)
                    trace.append(dict(step=step, action=action, reward=float(reward), observation=obs.tolist(), terminated=bool(terminated), truncated=bool(truncated)))
                    status = ('terminated' if terminated else 'time limit' if truncated else 'running')
                    frames.append(frame(step, status))
                    if terminated or truncated:
                        break
        finally:
            env.close()
        row = dict(policy=policy, training_seed=42, seed=seed, episode_return=total,
                   length=step, success=total >= 200, success_criterion='return >= 200',
                   terminated=bool(terminated), truncated=bool(truncated), reached_time_limit=bool(truncated),
                   recording=stem+'.gif', checkpoint=str(weights.relative_to(ROOT)), checkpoint_sha256=digest)
        (OUT / (stem+'.json')).write_text(json.dumps(dict(**row, config=config, trace=trace), indent=2)+'\n')
        frames[0].save(OUT / (stem+'.gif'), save_all=True, append_images=frames[1:], duration=20, loop=0, optimize=False)
        # Contact sheet preserves chronology for visual inspection; recordings retain every frame.
        indices = sorted(set([0, *range(49, len(frames), 50), len(frames)-1]))
        sheet = Image.new('RGB', (900, ((len(indices)+2)//3)*221), 'white')
        for j, idx in enumerate(indices):
            sheet.paste(frames[idx].resize((300,221)), ((j%3)*300,(j//3)*221))
        sheet.save(OUT / (stem+'_contact.png'))
        rows.append(row)
        with (OUT / 'episodes.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
        print(json.dumps(row), flush=True)
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == digest
print('OUTPUT', OUT, flush=True)
