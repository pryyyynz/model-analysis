import subprocess
import threading
import time

import psutil


class GpuMemoryMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                )
                for line in out.strip().splitlines():
                    val = float(line.strip())
                    if val > self.peak_mb:
                        self.peak_mb = val
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.peak_mb = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self._stop.clear()

    @property
    def peak(self):
        return self.peak_mb


def system_ram_mb():
    return psutil.virtual_memory().used / 1024 / 1024
