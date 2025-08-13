import requests, secrets, threading

class QRNGClient:
    """API-backed client (ANU QRNG). Thread-safe buffer with batched fetches.
    Modes:
      - 'quantum': fetch from ANU QRNG API (0..255 uints scaled to [0,1]).
      - 'pseudo' : secrets.SystemRandom()
      - 'deterministic': always 0.5
    """
    def __init__(self, mode='quantum', batch_size=4096):
        self.mode = mode
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._buf = []
        self._sys = secrets.SystemRandom()

    def _refill(self, n=None):
        if n is None:
            n = self.batch_size
        url = f"https://qrng.anu.edu.au/API/jsonI.php?length={n}&type=uint8"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        js = r.json()
        data = js.get("data", [])
        # scale 0..255 to [0,1]
        self._buf.extend([v/255.0 for v in data])

    def next(self) -> float:
        if self.mode == 'deterministic':
            return 0.5
        if self.mode == 'pseudo':
            return self._sys.random()
        # quantum via API
        with self._lock:
            if not self._buf:
                self._refill()
            return self._buf.pop()
