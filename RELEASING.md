# Releasing (videos, GitHub, Pages)

Videos NEVER go into git (302 MB raw; history keeps blobs forever). They travel
two routes: the hero via GitHub's markdown video attachment, the gallery via
Release assets. Web-compressed copies live in `~/molib/retargeted/web/`
(h264 crf30, ~10x smaller, still crisp).

## 1. Push the repo

```bash
gh repo create gmr-gpu --public --source . --push        # or git remote add + push
```

## 2. Hero video in the README (inline player)

GitHub renders an inline player ONLY for videos uploaded through its web UI
(max 10 MB; `hero_01_01_web.mp4` is 2.8 MB):

1. On github.com, open README.md -> pencil (edit).
2. Delete the `<!-- VIDEO PLACEHOLDER: hero ... -->` comment.
3. Drag `hero_01_01_web.mp4` from the file manager into the editor.
4. GitHub uploads it and inserts a `https://github.com/.../assets/...` URL.
   Leave that URL on its own line -> it renders as a player. Commit.

(This cannot be done from the CLI; asset URLs are minted only by the web UI.)

## 3. Gallery as Release assets (one command)

```bash
gh release create v0.1.0 ~/molib/retargeted/web/hero_*.mp4 \
    --title "gmr-gpu 0.1.0" \
    --notes "Full-CMU benchmark renders: 3-pane (cold | seq | mink) follow-cam."
```

Each video is then at
`https://github.com/<USER>/gmr-gpu/releases/download/v0.1.0/hero_<clip>.mp4`.
Replace the gallery placeholder in README.md with a link list (players are not
rendered for release links -- that's what the Pages site is for):

```markdown
## Gallery
20 side-by-side renders across CMU subjects:
[01_01](.../hero_01_01.mp4) · [102_28](.../hero_102_28.mp4) · ...
```

## 4. GH Pages site (real <video> players)

`docs/` will hold a static page whose `<video src=...>` tags point at the
release-asset URLs (they support direct linking) -- repo stays light, videos
stream. Enable: repo Settings -> Pages -> deploy from branch -> /docs.

## 5. Announce, then upstream

Open an issue on YanjieZe/GMR offering `--backend gpu`, linking the table and
the parity proofs.
