"""hymotion-local: Tencent HY-Motion 1.0 text-to-motion on one consumer GPU (12 GB is enough), made to be
driven by an AI agent or a person: prompt in, BVH/FBX clips plus a contact sheet and numbers out.

  python motion.py gen "A person ..." [--name N] [--dur 3] [--seeds 0,1,2,3 | --n 8] [--cfg 5]
         [--steps 50] [--solver euler] [--game] [--smooth 1] [--head steady] [--tf32] [--fbx] [--cols 8]
         [--out DIR]
  python motion.py batch jobs.json [--out DIR] [--fbx]   (a list of jobs: {"prompt", "name", any flag above})
  python motion.py sheet JOB_DIR [--seed S] [--cols N]  (redraw a job's sheets; one take: a close-up)
  python motion.py fbx JOB_DIR                          (FBX copies of a job's BVH takes, through Blender)
  python motion.py doctor                               (check the install, GPU and weights)

Out: <out>/<name>/<name>_<seed>.bvh (+ .npz: world joints, SMPL-H poses), sheet*.png, job.json (the
settings and each take's numbers). <out> is --out, else $HYMOTION_OUT, else ./motion_out.
  --dur     seconds, 0.67-12 (the model works at 30 fps; under 5 s is its sweet spot)
  --seeds   the takes; the same seed and settings give the same take (noise is drawn on the CPU)
  --cfg     how hard the text is followed (1 = ignored; 5 default; 7+ stiff and overdone)
  --steps   flow-matching steps (50 default; more = finer, slower)
  --solver  torchdiffeq method: euler (default), midpoint, rk4, dopri5 (adaptive: ignores --steps)
  --game    the model's "game animation" source token (a small style shift)
  --smooth  slerp smoothing sigma of the joints (1 default; 0 = raw joints and an unsmoothed root)
  --head    steady (default): the model's stray twist of the neck and head about the neck taken out (no
            joint moves); raw: as the model made it
  --tf32    TF32 matmuls: ~25% faster on RTX 30xx+, joints within ~0.2 deg of the full fp32 take
  --fbx     also write <take>.fbx (needs Blender: $BLENDER, on PATH, or its default install folder)
The text encoder (Qwen3-8B, bf16, 16 GB) runs once per new prompt with the layers that don't fit the
GPU held in RAM; its output is cached, so new seeds, cfg or steps for a known prompt skip it.
"""
import argparse
import gc
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parent


def hy_home():
    """The HY-Motion checkout holding ckpts/, .venv/ and cache/: $HYMOTION_HOME, else config.json's
    "home" (written by install.py), else ./HY-Motion-1.0 next to this file."""
    if os.environ.get("HYMOTION_HOME"):
        return Path(os.environ["HYMOTION_HOME"])
    cfg = TOOL / "config.json"
    if cfg.exists():
        return Path(json.loads(cfg.read_text())["home"])
    return TOOL / "HY-Motion-1.0"


HY = hy_home()
VENV = HY / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if VENV.exists() and Path(sys.executable).resolve() != VENV.resolve():
    sys.exit(subprocess.call([str(VENV), str(Path(__file__).resolve()), *sys.argv[1:]]))

import numpy as np

MODEL = HY / "ckpts/tencent/HY-Motion-1.0"
CACHE = HY / "cache"
FPS = 30
SMPLH = ["Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle", "Spine3",
         "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow",
         "R_Elbow", "L_Wrist", "R_Wrist"]
# (the BVH joints carry Mixamo's names: most retargeters and engines map them without setup)
MIXAMO = ["Hips", "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg", "Spine1", "LeftFoot", "RightFoot",
          "Spine2", "LeftToeBase", "RightToeBase", "Neck", "LeftShoulder", "RightShoulder", "Head", "LeftArm",
          "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand"]
MIXAMO += [f"{s}Hand{f}{i}" for s in ("Left", "Right") for f in ("Index", "Middle", "Pinky", "Ring", "Thumb")
           for i in (1, 2, 3)]
DEFAULTS = dict(name=None, dur=3.0, seeds=None, n=4, cfg=5.0, steps=50, solver="euler", game=False, smooth=1.0,
                head="steady", tf32=False, fbx=False, model=str(MODEL), cols=8)


def skeleton():
    d = HY / "scripts/gradio/static/assets/dump_wooden"
    j = np.fromfile(d / "j_template.bin", dtype=np.float32).reshape(-1, 3).astype(np.float64)
    parents = np.fromfile(d / "kintree.bin", dtype=np.int32)
    sole = np.fromfile(d / "v_template.bin", dtype=np.float32).reshape(-1, 3)[:, 1].min()
    return j, parents, float(sole)


def encode(prompts):
    """{prompt: its text features}, from the cache or Qwen3-8B + CLIP-L (loaded only when one is new)."""
    import torch
    CACHE.mkdir(exist_ok=True)
    key = {p: CACHE / (hashlib.sha1(f"128|{p}".encode()).hexdigest()[:16] + ".pt") for p in prompts}
    new = [p for p in prompts if not key[p].exists()]
    if new:
        from transformers import AutoModelForCausalLM
        from hymotion.network.text_encoders import text_encoder as te

        free = torch.cuda.mem_get_info()[0] / 2**30

        class Offloaded:  # (Qwen3-8B in bf16 as the reference runs it, the layers past the GPU's room in RAM)
            @staticmethod
            def from_pretrained(path, **kw):
                return AutoModelForCausalLM.from_pretrained(
                    path, device_map="auto", max_memory={0: f"{max(free - 2.0, 1.0):.1f}GiB", "cpu": "26GiB"}, **kw)
        te.LLM_ENCODER_LAYOUT["qwen3"]["text_encoder_class"] = Offloaded
        print(f">> text encoder for {len(new)} new prompt(s), {free:.1f} GiB GPU free", flush=True)
        enc = te.HYTextModel(llm_type="qwen3", max_length_llm=128)
        enc.sentence_emb_text_encoder.to("cuda")
        for p in new:
            vtxt, ctxt, n = enc.encode([p])
            torch.save({"text_vec_raw": vtxt[0].cpu(), "text_ctxt_raw": ctxt[0].cpu(),
                        "text_ctxt_raw_length": n.cpu()}, key[p])
        del enc
        gc.collect()
        torch.cuda.empty_cache()
    return {p: torch.load(key[p]) for p in prompts}


def load_pipeline(model):
    import torch
    import yaml
    from hymotion.utils.loaders import load_object
    cfg = yaml.safe_load(open(Path(model) / "config.yml"))
    pipe = load_object(cfg["train_pipeline"], cfg["train_pipeline_args"], network_module=cfg["network_module"],
                       network_module_args=cfg["network_module_args"])
    pipe.load_in_demo(str(Path(model) / "latest.ckpt"), build_text_encoder=False)
    return pipe.to(torch.device("cuda")).eval()


def write_bvh(path, poses, trans, j, parents, sole):
    """SMPL-H poses (frames, 52*3 axis-angle) and root translation as a BVH: Y up, the character facing +Z
    standing on y = 0, metres, 30 fps."""
    from scipy.spatial.transform import Rotation
    n = len(poses)
    kids = {i: [k for k in range(len(parents)) if parents[k] == i] for i in range(len(parents))}
    lines, order = ["HIERARCHY"], []

    def joint(i, depth):
        pad = "  " * depth
        off = (j[0] - [0, sole, 0]) if i == 0 else j[i] - j[parents[i]]
        lines.append(f"{pad}{'ROOT' if i == 0 else 'JOINT'} mixamorig:{MIXAMO[i]}")
        lines.append(pad + "{")
        lines.append(f"{pad}  OFFSET {off[0]:.5f} {off[1]:.5f} {off[2]:.5f}")
        lines.append(pad + "  CHANNELS " + ("6 Xposition Yposition Zposition " if i == 0 else "3 ")
                     + "Zrotation Xrotation Yrotation")
        order.append(i)
        for k in kids[i]:
            joint(k, depth + 1)
        if not kids[i]:
            e = [0, 0.18, 0] if MIXAMO[i] == "Head" else (j[i] - j[parents[i]]) * 0.6
            lines.extend([f"{pad}  End Site", pad + "  {", f"{pad}    OFFSET {e[0]:.5f} {e[1]:.5f} {e[2]:.5f}",
                          pad + "  }"])
        lines.append(pad + "}")
    joint(0, 0)
    eul = Rotation.from_rotvec(poses.reshape(-1, 3)).as_euler("ZXY", degrees=True).reshape(n, -1, 3)
    eul = np.degrees(np.unwrap(np.radians(eul), axis=0))  # (no 360 jumps frame to frame: resampled, they spun)
    root = j[0] + trans
    lines += ["MOTION", f"Frames: {n}", f"Frame Time: {1 / FPS:.7f}"]
    for f in range(n):
        vals = list(root[f]) + [v for i in order for v in eul[f, i]]
        lines.append(" ".join(f"{v:.4f}" for v in vals))
    Path(path).write_text("\n".join(lines) + "\n")


def fk(poses, trans, j, parents):
    """World joint positions (frames x 52 x 3) of SMPL-H poses on the template skeleton, as the BVH has
    them (HY's own keypoints3d leave the root's travel out)."""
    from scipy.spatial.transform import Rotation
    n = len(poses)
    rot = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(n, -1, 3, 3)
    g, k = np.zeros_like(rot), np.zeros((n, len(parents), 3))
    g[:, 0], k[:, 0] = rot[:, 0], j[0] + trans
    for i in range(1, len(parents)):
        p = parents[i]
        g[:, i] = g[:, p] @ rot[:, i]
        k[:, i] = k[:, p] + g[:, p] @ (j[i] - j[p])
    return k


def steady_head(poses, j):
    """The neck's and head's turn about the neck taken out (frames x 156 SMPL-H axis-angle; a copy), their
    tilt kept. HY-Motion learnt them from joint positions, which a turn about the neck leaves where they
    are: its takes carry a stray 20-50 deg twist there, the face looking off to one side."""
    from scipy.spatial.transform import Rotation
    p = np.asarray(poses, dtype=np.float64).reshape(len(poses), -1, 3).copy()
    a = (j[15] - j[12]) / np.linalg.norm(j[15] - j[12])
    for i in (12, 15):
        q = Rotation.from_rotvec(p[:, i]).as_quat()  # (x, y, z, w)
        t = np.concatenate([a * (q[:, :3] @ a)[:, None], q[:, 3:]], axis=1)  # (its twist about the neck)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        p[:, i] = (Rotation.from_quat(q) * Rotation.from_quat(t).inv()).as_rotvec()
    return p.reshape(len(poses), -1)


def facing(k):
    """The way a body faces (frames x 3, level unit vectors), from its hips and shoulders together as the
    training data defines it (the hips alone sit ~40 deg off the chest in a sword stance)."""
    across = (k[..., 2, :] - k[..., 1, :]) + (k[..., 17, :] - k[..., 16, :])
    f = np.cross([0, 1, 0], across)
    f[..., 1] = 0
    return f / np.linalg.norm(f, axis=-1, keepdims=True)


def measure(k):
    """A take's numbers (world joints: frames x 52 x 3, Y up): where it went, its turn, its feet, its hands."""
    f0 = facing(k[0])
    side = np.cross(f0, [0, 1, 0])  # (its right)
    hips = k[:, 0]
    d = hips[-1] - hips[0]
    far = np.linalg.norm((hips - hips[0])[:, [0, 2]], axis=1).max()
    fw = facing(k)
    yaw = np.degrees(np.unwrap(np.arctan2(fw[:, 0], fw[:, 2])))
    yaw -= yaw[0]
    lead = [(k[i, 8] - k[i, 7]) @ f0 for i in (0, -1)]  # (the right ankle ahead of the left, start and end)
    skate = 0.0  # (a foot down = its ankle and toe both within 2 cm of their lowest: a low toe alone counted
    for ankle, toe in ((7, 10), (8, 11)):  # the swing foot skimming the floor, 2-4 m "slide" on a clean walk)
        down = (k[:, ankle, 1] < k[:, ankle, 1].min() + 0.02) & (k[:, toe, 1] < k[:, toe, 1].min() + 0.02)
        step = np.linalg.norm(np.diff(k[:, toe][:, [0, 2]], axis=0), axis=1)
        skate += float(step[down[1:] & down[:-1]].sum())
    grip = np.linalg.norm(k[:, 20] - k[:, 21], axis=1)
    return dict(secs=round(len(k) / FPS, 2), fwd=round(float(d @ f0), 2), right=round(float(d @ side), 2),
                far=round(float(far), 2), yaw=round(float(yaw[-1]), 1), yaw_max=round(float(np.abs(yaw).max()), 1),
                lead=[round(float(v), 2) for v in lead], skate=round(skate, 2), grip=round(float(np.median(grip)), 2),
                hips=[round(float(hips[:, 1].min()), 2), round(float(hips[:, 1].max()), 2)])


def describe(m):
    lead = "/".join(("R" if v > 0 else "L") + f"{abs(v):.2f}" for v in m["lead"])
    return (f"fwd {m['fwd']:+.2f} right {m['right']:+.2f} (far {m['far']:.2f}) m, yaw {m['yaw']:+.0f} "
            f"(max {m['yaw_max']:.0f}), lead foot {lead}, skate {m['skate']:.2f} m, hands {m['grip']:.2f} m apart, "
            f"hips {m['hips'][0]:.2f}-{m['hips'][1]:.2f}")


def draw(job_dir, cols=8, seed=None):
    """Contact sheets of a job: per take a front row (facing us, its right hand on our left) and a side row
    (seen from its right, forward to the right); red = right side, blue = left. Each cell is centred on
    the hips; the floor ticks every 0.5 m show the travel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    job = json.loads((job_dir / "job.json").read_text())
    _, parents, _ = skeleton()
    takes = [t for t in job["takes"] if seed is None or t["seed"] == seed]
    big = 1.6 if seed is not None else 1.0  # (a close-up of one take: bigger cells)
    paths = []
    for page in range(0, len(takes), 4):
        group = takes[page:page + 4]
        fig, axes = plt.subplots(2 * len(group), cols, figsize=(cols * 1.4 * big, len(group) * 3.0 * big + 0.6),
                                 squeeze=False)
        for t, take in enumerate(group):
            k = np.load(job_dir / take["npz"])["keypoints"]
            f0 = facing(k[0])
            yaw = np.arctan2(f0[0], f0[2])
            c, s = np.cos(-yaw), np.sin(-yaw)
            k = k - [k[0, 0, 0], 0, k[0, 0, 2]]
            k = np.stack([c * k[..., 0] + s * k[..., 2], k[..., 1], -s * k[..., 0] + c * k[..., 2]], -1)  # (start facing +Z)
            frames = np.linspace(0, len(k) - 1, cols).round().astype(int)
            for row, (h, name) in enumerate(((0, "front"), (2, "side"))):
                for col, f in enumerate(frames):
                    ax = axes[2 * t + row, col]
                    cx = k[f, 0, h]
                    ax.set_xlim(cx - 1.0, cx + 1.0)
                    ax.set_ylim(-0.08, 1.92)
                    ax.set_aspect("equal")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.axhline(0, color="0.6", lw=0.8)
                    for x in np.arange(np.floor((cx - 1) * 2) / 2, cx + 1, 0.5):
                        ax.plot([x, x], [-0.08, 0], color="0.6", lw=0.8)
                    ax.plot(k[:f + 1, 0, h], k[:f + 1, 0, 1] * 0 + 0.01, color="0.75", lw=1)
                    for i in range(1, 22):
                        p = parents[i]
                        nm = SMPLH[i]
                        colr = "#c0392b" if nm.startswith("R_") else "#2e6fd1" if nm.startswith("L_") else "0.15"
                        ax.plot(k[f, [p, i], h], k[f, [p, i], 1], color=colr, lw=2.2, solid_capstyle="round")
                    ax.plot(k[f, 15, h], k[f, 15, 1] + 0.06, "o", color="0.15", ms=5)
                    if t == 0 and row == 0:
                        ax.set_title(f"{f / FPS:.2f}s", fontsize=8)
                    if col == 0:
                        ax.set_ylabel(f"s{take['seed']} {name}", fontsize=8)
            axes[2 * t, 0].text(0, 1.02 if t else 1.2, f"s{take['seed']}: " + describe(take["m"]), fontsize=7,
                                transform=axes[2 * t, 0].transAxes, ha="left", va="bottom").set_in_layout(False)
        fig.suptitle(job["prompt"][:150], fontsize=8, y=0.999)
        fig.tight_layout(pad=0.4, h_pad=1.2, rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
        p = job_dir / (f"take{seed}.png" if seed is not None else f"sheet{page // 4 + 1}.png")
        fig.savefig(p, dpi=90)
        plt.close(fig)
        paths.append(p)
    return paths


def blender():
    found = os.environ.get("BLENDER") or shutil.which("blender")
    if not found:
        pats = ["C:/Program Files/Blender Foundation/Blender */blender.exe",
                "/Applications/Blender.app/Contents/MacOS/Blender", "/usr/bin/blender", "/snap/bin/blender"]
        hits = sorted((h for p in pats for h in glob.glob(p)), key=lambda h: [int(x) for x in
                      "".join(c if c.isdigit() else " " for c in h).split()] or [0])
        found = hits[-1] if hits else None
    return found


FBX_SCRIPT = """
import bpy, sys
for src in sys.argv[sys.argv.index("--") + 1:]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_anim.bvh(filepath=src, global_scale=1.0, update_scene_fps=True,
                            update_scene_duration=True, rotate_mode="QUATERNION", axis_forward="-Z", axis_up="Y")
    arm = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
    a, z = (int(round(v)) for v in arm.animation_data.action.frame_range)
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = a, z  # (else the default 1-250 is baked)
    bpy.ops.export_scene.fbx(filepath=src[:-4] + ".fbx", add_leaf_bones=False, bake_anim=True,
                             bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False,
                             bake_anim_simplify_factor=0.0, axis_forward="-Z", axis_up="Y")
    print("fbx", src[:-4] + ".fbx")
"""


def to_fbx(bvhs):
    exe = blender()
    if not exe:
        print("   no Blender found (set $BLENDER to blender's executable): FBX skipped", flush=True)
        return
    subprocess.run([exe, "-b", "--factory-startup", "--python-expr", FBX_SCRIPT, "--", *map(str, bvhs)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"   fbx: {len(bvhs)} take(s) via {exe}", flush=True)


def run(jobs, out):
    import torch
    for job in jobs:
        for k, v in DEFAULTS.items():
            job.setdefault(k, v)
        job["name"] = job["name"] or hashlib.sha1(job["prompt"].encode()).hexdigest()[:8]
        if isinstance(job["seeds"], str):
            job["seeds"] = [int(s) for s in job["seeds"].split(",") if s.strip()]
        job["seeds"] = job["seeds"] or list(range(job["n"]))
    feats = encode(sorted({j["prompt"] for j in jobs}))
    from hymotion.pipeline.body_model import construct_smpl_data_dict
    j, parents, sole = skeleton()
    pipes = {}
    for job in jobs:
        if job["model"] not in pipes:
            pipes.clear()
            torch.cuda.empty_cache()
            pipes[job["model"]] = load_pipeline(job["model"])
        pipe = pipes[job["model"]]
        torch.backends.cuda.matmul.allow_tf32 = job["tf32"]
        pipe.validation_steps = job["steps"]
        pipe._noise_scheduler_cfg = {"method": job["solver"]}
        sm = job["smooth"]
        cls = type(pipe)
        pipe.smooth_with_slerp = (lambda x, sigma=1.0: cls.smooth_with_slerp(x, sigma=sm)) if sm > 0 else (lambda x, sigma=1.0: x)
        pipe.smooth_with_savgol = cls.smooth_with_savgol if sm > 0 else (lambda x, window_length=9, polyorder=5: x)
        print(f">> {job['name']}: {job['prompt'][:90]} | {job['dur']}s seeds {job['seeds']} cfg {job['cfg']} "
              f"steps {job['steps']} {job['solver']}{' game' if job['game'] else ''}{' tf32' if job['tf32'] else ''} "
              f"smooth {sm}{' raw head' if job['head'] == 'raw' else ''}", flush=True)
        try:
            res = pipe.generate(job["prompt"], job["seeds"], job["dur"], cfg_scale=job["cfg"],
                                use_special_game_feat=job["game"],
                                hidden_state_dict={k: v.cuda() for k, v in feats[job["prompt"]].items()})
        except Exception as err:  # (one bad job doesn't end a long batch)
            print(f"   FAILED: {err!r}", flush=True)
            torch.cuda.empty_cache()
            continue
        d = out / job["name"]
        d.mkdir(parents=True, exist_ok=True)
        takes = []
        for b, seed in enumerate(job["seeds"]):
            smpl = construct_smpl_data_dict(res["rot6d"][b].clone(), res["transl"][b].clone())
            poses = np.asarray(smpl["poses"]) if job["head"] == "raw" else steady_head(smpl["poses"], j)
            k = fk(poses, smpl["trans"], j, parents)
            stem = f"{job['name']}_{seed}"
            np.savez(d / f"{stem}.npz", keypoints=k, poses=poses, trans=smpl["trans"], fps=FPS)
            write_bvh(d / f"{stem}.bvh", poses, smpl["trans"], j, parents, sole)
            m = measure(k)
            takes.append(dict(seed=seed, bvh=f"{stem}.bvh", npz=f"{stem}.npz", m=m))
            print(f"   s{seed}: {describe(m)}", flush=True)
        (d / "job.json").write_text(json.dumps({**job, "takes": takes}, indent=1))
        for p in draw(d, job["cols"]):
            print(f"   sheet {p}", flush=True)
        if job["fbx"]:
            to_fbx([d / t["bvh"] for t in takes])


def doctor():
    """The install, GPU and weights, then forward kinematics against HY's own body model (no weights needed)."""
    ok = True

    def check(cond, what, fix=""):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + what + ("" if cond or not fix else f"  -> {fix}"))
        ok = ok and bool(cond)
    check(HY.exists(), f"HY-Motion code at {HY}", "run install.py")
    check(Path(sys.executable).resolve() == VENV.resolve(), f"running in {VENV}", "run install.py")
    try:
        import torch
        cuda = torch.cuda.is_available()
        check(cuda, f"torch {torch.__version__} with CUDA", "install a CUDA build of torch (install.py does)")
        if cuda:
            free, total = torch.cuda.mem_get_info()
            check(total / 2**30 > 11, f"GPU {torch.cuda.get_device_name(0)}: {total / 2**30:.1f} GiB "
                  f"({free / 2**30:.1f} free)", "needs ~12 GB of VRAM")
    except ImportError:
        check(False, "torch", "run install.py")
    for p, what in ((MODEL / "latest.ckpt", "motion model"), (HY / "ckpts/Qwen3-8B/model.safetensors.index.json", "Qwen3-8B text encoder"),
                    (HY / "ckpts/clip-vit-large-patch14/model.safetensors", "CLIP-L text encoder"),
                    (HY / "stats/Mean.npy", "motion stats (git-lfs file)")):
        check(p.exists() and p.stat().st_size > 1000, f"{what}: {p}", "run install.py --accept-license")
    exe = blender()
    print(f"  {'ok  ' if exe else 'info'} Blender for --fbx: {exe or 'not found (optional; set $BLENDER)'}")
    if HY.exists():
        import torch
        from hymotion.pipeline.body_model import WoodenMesh, construct_smpl_data_dict
        from hymotion.utils.geometry import rotation_matrix_to_rot6d
        from scipy.spatial.transform import Rotation
        j, parents, _ = skeleton()
        rng = np.random.default_rng(0)
        rot = torch.tensor(Rotation.from_rotvec(rng.normal(0, 0.5, (8 * 22, 3))).as_matrix(), dtype=torch.float32)
        r6 = rotation_matrix_to_rot6d(rot).reshape(8, 22, 6)
        trans = torch.tensor(rng.normal(0, 0.3, (8, 3)), dtype=torch.float32)
        ref = WoodenMesh().forward({"rot6d": r6, "trans": trans})["keypoints3d"].numpy()[:, :22]
        smpl = construct_smpl_data_dict(r6, trans)
        mine = fk(smpl["poses"], smpl["trans"], j, parents)[:, :22]
        err = np.abs((mine - mine[:, :1]) - (ref - ref[:, :1])).max()
        check(err < 1e-4, f"forward kinematics matches HY's body model (max error {err:.1e} m)")
        steady = fk(steady_head(smpl["poses"], j), smpl["trans"], j, parents)[:, :22]
        moved = np.abs(steady - mine).max()
        check(moved < 1e-6, f"steady_head moves no joint (max {moved:.1e} m)")
    print("ready" if ok else "not ready")
    return ok


def main():
    ap = argparse.ArgumentParser(description="HY-Motion 1.0 text-to-motion, locally")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("prompt")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            g.add_argument(f"--{k}", action="store_true")
        else:
            g.add_argument(f"--{k}", type=type(v) if v is not None else str, default=v)
    b = sub.add_parser("batch")
    b.add_argument("file")
    b.add_argument("--fbx", action="store_true")
    for p in (g, b):
        p.add_argument("--out", default=os.environ.get("HYMOTION_OUT", "motion_out"))
    sh = sub.add_parser("sheet")
    sh.add_argument("job")
    sh.add_argument("--seed", type=int)
    sh.add_argument("--cols", type=int)
    sub.add_parser("fbx").add_argument("job")
    sub.add_parser("doctor")
    a = ap.parse_args()
    a.cwd = os.getcwd()
    if a.cmd in ("sheet", "fbx"):
        d = Path(a.job).resolve()
        job = json.loads((d / "job.json").read_text())
        if a.cmd == "fbx":
            return to_fbx([d / t["bvh"] for t in job["takes"]])
        for t in job["takes"]:  # (the numbers again, so a redraw after an update of measure() shows new ones)
            t["m"] = measure(np.load(d / t["npz"])["keypoints"])
        (d / "job.json").write_text(json.dumps(job, indent=1))
        for p in draw(d, a.cols or job.get("cols", 8), a.seed):
            print(p)
        return
    if HY.exists():
        os.chdir(HY)  # (HY's code reads its stats, skeleton and encoders from paths under its own folder)
        sys.path.insert(0, str(HY))
    if a.cmd == "doctor":
        sys.exit(0 if doctor() else 1)
    os.chdir(a.cwd)
    out = Path(a.out).resolve()
    if a.cmd == "gen":
        jobs = [{k: getattr(a, k) for k in ("prompt", *DEFAULTS)}]
    else:
        jobs = json.loads(Path(a.file).read_text())
        for job in jobs:
            job.setdefault("fbx", a.fbx)
    os.chdir(HY)
    run(jobs, out)


if __name__ == "__main__":
    main()
