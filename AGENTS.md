# hymotion-local for AI agents

You can make character animation from text with this tool: prompt in, BVH/FBX takes plus a picture out.
These notes cover installing it, using it well, and the common mistakes. Details are in [README.md](README.md).

## 1. Install (once per machine)

1. Check the hardware. Run `nvidia-smi`: you need an NVIDIA GPU with 12 GB or more. You also need 32 GB of
   RAM and about 28 GB of free disk. If any is missing, stop and tell the user.
2. Tell the user that the weights are under the Tencent HY-Motion 1.0 Community License. It does not
   apply in the EU, the UK or South Korea, and products over 1M monthly users need Tencent's permission.
   Get their OK before you pass `--accept-license`. The download is about 22 GB.
3. Install:
   ```bash
   git clone https://github.com/rottenvia/hymotion-local && cd hymotion-local
   python install.py --accept-license        # add --cuda cu128 on RTX 50xx
   ```
   Run it in the background, because the download takes a while. It ends with `motion.py doctor`, which
   must print `ready`. Re-running is safe: finished parts are skipped.
4. If the user already has a HY-Motion checkout with `ckpts/`, use `python install.py --home <that folder>`.

## 2. Generate

```bash
python motion.py gen "A person ..." --name NAME --dur 2 --n 4     # 4 takes
python motion.py batch jobs.json --out DIR                           # many prompts in one run
```

- One job of 4 takes runs about 40-45 s on an RTX 3060. The first new prompt of a run adds 1-2 min to load
  the text encoder. Run batches in the background and wait for them to exit; don't poll in a loop.
- Run one generation at a time. Two runs at once don't fit in 12 GB.
- A batch file is a JSON list of jobs: `{"name": "hit", "prompt": "A person ...", "dur": 1.5, "seeds": [0,1,2,3], "cfg": 5}`.
  Any flag of `gen` works as a key. A failing job is reported and skipped, and the rest still run.
- The same prompt and seed with the same settings gives the same take. Prompt encodings are cached, so
  re-running a prompt with other seeds or settings is cheap.

## 3. Look before you choose

After each job, open `<out>/<name>/sheet1.png`; it is an image, so read it.

- Each take has a front row and a side row; red is the character's right side.
- The line above each take gives its numbers: `fwd`/`right` travel, `yaw` turn, `lead foot`, foot `skate`,
  `hands` apart, and `hips` height range.
- For a close-up of one take: `python motion.py sheet <out>/<name> --seed S --cols 10`, then read `take<S>.png`.
- Choose by both the pictures and the numbers. Watch for:
  - turns or travel nobody asked for (`yaw`, `far`);
  - the wrong foot leading;
  - big `skate`;
  - hands far apart on a two-handed grip (`hands` > 0.3 m);
  - a stance that doesn't match the start or end pose you need;
  - for reactions in a fight: a turn away from the attacker (`yaw` max over about 30 deg), or a fall that
    travels towards them (`fwd` over about 0.5 m). Pick another take, or take out part of the turn or
    travel when you retarget.
- If no take is right, change the prompt (see 4) or try more seeds (`--n 8`). Changing `--cfg` (3-7) helps
  less.

## 4. Prompting that works

- Write a caption the way the training data does: "A person holding a two-handed sword is hit in the chest,
  flinches backward, takes a small step back with the left foot, and recovers into a ready stance."
- Use plain English and the character's own left and right. Order the clauses in time. Name body parts and
  directions. Keep it under about 60 words, with `--dur` under 5 s.
- Good subjects: reactions, staggers, falls and deaths, dodges, guards and parries, gestures, walking and
  running, everyday actions.
- Unreliable: exact foot placement, exact weapon lines in fast swings (big swings twist the whole body),
  facing held over a long motion, and seamless loops or in-place motion. Fix these in post (trim, root
  lock) or choose another take.
- Loops: generate a walk, run or idle of 4-5 s, then `python motion.py loop <out>/<name>` (idles:
  `--min 2 --max 4`). It writes `<name>_loop/`, one seamless loop per take. Choose by its numbers: `seam`
  under about 5 cm, `speed` near the speed your game moves at, and `skate` low.
- No scene, camera, emotion, clothing or second person.

## 5. Hand the take over

- The BVH uses Mixamo bone names (`mixamorig:Hips` ...), Y up, faces +Z, metres, 30 fps, feet on y = 0.
  Blender imports it with forward -Z and up Y.
- `--fbx` (or `python motion.py fbx <job dir>`) writes FBX through Blender, for Unity, Unreal, Roblox or
  any Mixamo retargeter. Set `$BLENDER` if Blender isn't found.
- `<take>.npz` has `keypoints` (frames x 52 x 3 world joints) for your own retargeting or checks.
- The head's stray twist is already removed (`--head steady`, the default). If a retargeted face still looks
  away, aim the head from the neck-to-head line of `keypoints`, not from the head bone's rotation.
- Tell the user which take you chose and why, and show them its sheet.
