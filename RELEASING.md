# Releasing (videos, GitHub, Pages)

Videos NEVER go into git (302 MB raw; history keeps blobs forever). They travel
two routes: three featured videos via GitHub's markdown drag-drop (inline
players), the gallery via Release assets. Web-compressed copies live in
`~/molib/retargeted/web/` (18 files, 33 MB, h264 crf30) plus
`~/molib/retargeted/hero_01_01_web.mp4` (2.8 MB).

## 0. Install gh (one-time)

```bash
sudo mkdir -p -m 755 /etc/apt/keyrings
wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update && sudo apt install gh -y
gh auth login        # choose GitHub.com -> HTTPS -> login with browser (device code)
```

(No-gh fallback: create the repo on github.com, then
`git remote add origin https://github.com/<USER>/gmr-gpu.git && git push -u origin main`;
create the release on github.com -> Releases -> "Draft a new release" and drag
the 18 web/ videos into the assets box.)

## 1. Push the repo

```bash
cd ~/molib/gmr-gpu
gh repo create gmr-gpu --public --source . --push
```

## 2. Inline video players — THREE drag-drops (web editor only)

GitHub renders an inline player ONLY for videos uploaded through its web UI
(max 10 MB each; all three are under). On github.com, open README.md ->
pencil (edit), then for each placeholder comment, delete it and drag the file
from the file manager into the editor at that spot. GitHub inserts a
`https://github.com/.../assets/...` URL — leave it on its own line. One commit
for all three.

| placeholder (README comment) | file to drag |
|---|---|
| hero, top of page | `~/molib/retargeted/hero_01_01_web.mp4` (2.8 MB) |
| featured failure pair, first | `~/molib/retargeted/web/hero_102_28.mp4` |
| featured failure pair, second | `~/molib/retargeted/web/hero_94_01.mp4` |

(This cannot be done from the CLI; asset URLs are minted only by the web UI.
WSL tip: `explorer.exe .` in the video dir opens Windows Explorer for dragging.)

## 3. Gallery as Release assets (one command)

```bash
gh release create v0.1.0 ~/molib/retargeted/web/hero_*.mp4 \
    --title "gmr-gpu 0.1.0" \
    --notes "Full-CMU benchmark renders: 3-pane (cold | seq | mink) follow-cam. Featured: 102_28 and 94_01 show the production pipeline's warm-start chain failing while gmr-gpu tracks."
```

Each video is then at
`https://github.com/<USER>/gmr-gpu/releases/download/v0.1.0/hero_<clip>.mp4`.
Replace the gallery placeholder in README.md with a link list (release links
render as links, not players — the Pages site handles inline playback):

```markdown
## Gallery
18 side-by-side renders across CMU subjects:
[01_01](.../hero_01_01.mp4) · [102_28](.../hero_102_28.mp4) · ...
```

## 4. GH Pages site (real <video> players)

`docs/` will hold a static page whose `<video src=...>` tags point at the
release-asset URLs (they support direct linking) — repo stays light, videos
stream. Enable: repo Settings -> Pages -> deploy from branch -> main /docs.
The site itself: ask Claude once the release URLs exist.

## 5. Announce, then upstream

Open an issue on YanjieZe/GMR offering `--backend gpu`, linking the table, the
61-vs-2 reliability finding, and the parity proofs.
