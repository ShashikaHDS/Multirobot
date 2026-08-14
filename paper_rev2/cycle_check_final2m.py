"""Re-verify the livelock claim: every deterministic-protocol failure of the
tag-`final` models is an exact joint-state cycle (positions + known_map
revisited), which under a memoryless deterministic policy in a deterministic
env implies looping until truncation.  Mirrors eval_paper.py's deterministic
protocol exactly (20 maps, seeds 10000+i, model.predict(deterministic=True)).
"""
import sys, json
from pathlib import Path

BASE = Path(r"D:\RL\Teal_rl\RL\RL_V2.0\paper_rev2")
sys.path.insert(0, str(BASE))

from eval_paper import discover_runs, env_from_run, EVAL_SEED_BASE
from stable_baselines3 import PPO

runs, skipped = discover_runs(BASE / "runs_paper", "final2m")
assert not skipped, skipped
print(f"runs: {len(runs)}")

tot_fail = 0
cyc_fail = 0
per_run = {}
for run in runs:
    env = env_from_run(run)
    model = PPO.load(str(run["model"]), device="cpu")
    nf = nc = ns = 0
    for ep in range(20):
        obs, _ = env.reset(seed=EVAL_SEED_BASE + ep)
        seen = {(env.positions.tobytes(), env.known_map.tobytes())}
        cycled = False
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            key = (env.positions.tobytes(), env.known_map.tobytes())
            if key in seen:
                cycled = True          # exact state revisit -> livelock
            seen.add(key)
        ok = bool(info.get("is_success", False))
        if ok:
            ns += 1
        else:
            nf += 1
            nc += cycled
    env.close()
    tag = f"N{run['n']}_M{run['m']} seed{run['seed']}"
    per_run[tag] = (ns, nf, nc)
    print(f"{tag}: success {ns}/20, failures {nf}, cycled {nc}", flush=True)
    tot_fail += nf
    cyc_fail += nc

print(f"\nTOTAL: failures {tot_fail}, exact-state cycles {cyc_fail} "
      f"({cyc_fail}/{tot_fail})")
