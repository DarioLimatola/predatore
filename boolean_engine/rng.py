"""V7 episode generator: xoshiro256** seeded through SplitMix64.

Bit-identical to ``tesseract_core::rng::Rng`` (Tesseract repository), which the
native episode kernel uses. Python and Rust therefore draw the same numbers for
the same seed. V7 is a declared new experimental line: V6 episodes used Python's
Mersenne Twister and are compared with V7 only statistically.
"""
M64 = (1 << 64) - 1


def _splitmix(state):
    state = (state + 0x9E3779B97F4A7C15) & M64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return state, z ^ (z >> 31)


def _rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & M64


def seed64(seed):
    """Any Python integer (also negative) mapped to an unsigned 64-bit seed."""
    return int(seed) % (1 << 64)


class Rng:
    __slots__ = ('s',)

    def __init__(self, seed):
        sm = seed64(seed)
        s = []
        for _ in range(4):
            sm, value = _splitmix(sm)
            s.append(value)
        self.s = s

    def next_u64(self):
        s = self.s
        result = (_rotl((s[1] * 5) & M64, 7) * 9) & M64
        t = (s[1] << 17) & M64
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = _rotl(s[3], 45)
        return result

    def below(self, n):
        """Uniform integer in 0..n, unbiased rejection sampling; 0 when n == 0."""
        if n <= 0:
            return 0
        zone = M64 - (M64 % n)
        while True:
            v = self.next_u64()
            if v < zone:
                return v % n

    def choice(self, items):
        return items[self.below(len(items))]

    def sample2(self, items):
        """Two distinct elements, in draw order."""
        n = len(items)
        if n < 2:
            raise ValueError('sample2 needs at least two items')
        i = self.below(n)
        j = self.below(n - 1)
        if j >= i:
            j += 1
        return items[i], items[j]

    def getstate(self):
        return tuple(self.s)
