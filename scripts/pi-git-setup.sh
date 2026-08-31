#!/usr/bin/env bash
# Run this ONCE, directly on a (new) Pi (not from the laptop).
#
# Creates one bare git repo for the whole combined project under ~/repos,
# with a post-receive hook that checks the pushed commit out straight into
# the live working directory and initializes submodules (rpi-lorawan).
# After this, `git push pi main` from the laptop replaces scp for deploying
# detector_ui/, drone_detect/, Raspberry_Recorder/, and scripts/ together.
#
# Untracked files in the working directory (.venv/, "Trained models/data",
# etc.) are left alone by `git checkout -f` -- it only touches tracked paths.
set -euo pipefail

REPO_BASE="$HOME/repos"
BARE="$REPO_BASE/drone_detector.git"
WORKTREE="$HOME/workspace/drone_detector"

mkdir -p "$REPO_BASE"

if [ -d "$BARE" ]; then
    echo "drone_detector: bare repo already exists at $BARE, skipping"
    exit 0
fi

mkdir -p "$WORKTREE"
git init --bare -b main "$BARE" >/dev/null

cat > "$BARE/hooks/post-receive" <<HOOK
#!/bin/sh
git --git-dir="$BARE" --work-tree="$WORKTREE" checkout -f main
git --git-dir="$BARE" --work-tree="$WORKTREE" submodule update --init --recursive || \
    echo "Warning: submodule update failed (no internet/SSH key on the Pi?)"
echo "Deployed drone_detector to $WORKTREE"
HOOK
chmod +x "$BARE/hooks/post-receive"
echo "drone_detector: bare repo created at $BARE -> checks out into $WORKTREE"

echo
echo "Done. On the laptop, point the repo's 'pi' remote at this Pi, e.g.:"
echo "  git remote add pi pi@<host>:repos/drone_detector.git"
echo "Then deploy with: git push pi main"
