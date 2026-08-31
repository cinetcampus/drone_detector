#!/usr/bin/env bash
# Run this ONCE, directly on the Pi (not from the laptop).
#
# Creates a bare git repo per project under ~/repos, each with a
# post-receive hook that checks the pushed commit out straight into the
# live working directory. After this, `git push pi main` from the laptop
# replaces scp for deploying detector_ui/, drone_detect/, and scripts/.
#
# Untracked files in the working directories (.venv/, "Trained models/",
# etc.) are left alone by `git checkout -f` -- it only touches tracked paths.
set -euo pipefail

REPO_BASE="$HOME/repos"
WORK_BASE="$HOME/workspace/drone_detector"

mkdir -p "$REPO_BASE"

setup_repo() {
    local name="$1"
    local bare="$REPO_BASE/$name.git"
    local worktree="$WORK_BASE/$name"

    if [ -d "$bare" ]; then
        echo "$name: bare repo already exists at $bare, skipping"
        return
    fi

    mkdir -p "$worktree"
    git init --bare -b main "$bare" >/dev/null

    cat > "$bare/hooks/post-receive" <<HOOK
#!/bin/sh
git --git-dir="$bare" --work-tree="$worktree" checkout -f main
echo "Deployed $name to $worktree"
HOOK
    chmod +x "$bare/hooks/post-receive"
    echo "$name: bare repo created at $bare -> checks out into $worktree"
}

setup_repo "detector_ui"
setup_repo "drone_detect"
setup_repo "scripts"

echo
echo "Done. On the laptop, point each repo's 'pi' remote at this Pi, e.g.:"
echo "  git -C detector_ui  remote add pi pi@<host>:repos/detector_ui.git"
echo "  git -C drone_detect remote add pi pi@<host>:repos/drone_detect.git"
echo "  git -C scripts      remote add pi pi@<host>:repos/scripts.git"
echo "Then deploy with: git push pi main"
