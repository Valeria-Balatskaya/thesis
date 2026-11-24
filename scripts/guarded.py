# scripts/guarded.py
# Run a resumable long job so that one stalled case cannot cost hours (the 5.3 h stall in the
# E1 trustmark pilot): the child is restarted when its progress files have not changed for
# --stall minutes. Every test in tests/test_owner_*.py skips finished rows, so a restart loses
# at most the photo in flight.
#
#   caffeinate -dimsu python scripts/guarded.py --watch output/owner_forensics/pilot/wam.csv \
#       --stall 25 --log output/logs/x.log -- env VICTIM=wam SET=pilot .venv/bin/python tests/test_owner_forensics.py
#
# --watch  file(s) whose mtime or size is the progress signal (the CSV the job appends to; repeatable)
# --stall  minutes without progress before kill + restart
# --max-restarts  give up after this many restarts (exit 3)
# The wrapper itself writes <log>.guard (start / stall / restart / exit with timestamps and the
# wall-clock jump, which also shows whether the machine slept).

import argparse
import os
import signal
import subprocess
import sys
import time


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def state(paths):
    out = []
    for p in paths:
        try:
            st = os.stat(p)
            out.append((st.st_mtime, st.st_size))
        except FileNotFoundError:
            out.append((0, 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="append", default=[])
    ap.add_argument("--stall", type=float, default=25.0)
    ap.add_argument("--max-restarts", type=int, default=12)
    ap.add_argument("--log", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    guard = open(a.log + ".guard", "a", buffering=1)

    def note(msg):
        guard.write(f"{stamp()}  {msg}\n")

    restarts = 0
    while True:
        note(f"start (restart #{restarts}): {' '.join(cmd)}")
        out = open(a.log, "a")
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        last, last_change, last_tick = state(a.watch), time.time(), time.time()
        while proc.poll() is None:
            time.sleep(20)
            now = time.time()
            if now - last_tick > 120:                  # the loop sleeps 20 s: a big jump = the machine slept
                note(f"wall-clock jump of {now - last_tick:.0f} s (machine slept?)")
                last_change = now                      # the sleep is not a stall of the job
            last_tick = now
            cur = state(a.watch)
            if cur != last:
                last, last_change = cur, now
            elif now - last_change > a.stall * 60:
                note(f"no progress for {(now - last_change) / 60:.0f} min -> kill and restart")
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                break
        else:
            rc = proc.returncode
            note(f"exit {rc}")
            return rc
        restarts += 1
        if restarts > a.max_restarts:
            note("too many restarts, giving up")
            return 3


if __name__ == "__main__":
    sys.exit(main())
