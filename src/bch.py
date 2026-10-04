# src/bch.py
# Binary BCH error-correcting code (pure Python, no dependencies).
#
# Used to protect the watermark payload: the neural decoder never recovers
# every bit after a screenshot or JPEG, so the product ID is encoded with
# redundancy and up to t bit errors are corrected at detection time.
#
# Default: BCH(63, 30), t = 6 — 30 information bits in a 63-bit codeword,
# any 6 bit errors corrected (raw BER up to 9.5%).
#
# References:
#   Bose, R.C. & Ray-Chaudhuri, D.K. (1960). On a class of error correcting
#   binary group codes. Information and Control, 3(1), 68-79.
#   Lin, S. & Costello, D.J. (2004). Error Control Coding, 2nd ed., ch. 6
#   (Berlekamp-Massey decoding, Chien search).

import numpy as np

# Primitive polynomials for GF(2^m), as bit masks (x^6 + x + 1 -> 0b1000011)
_PRIMITIVE = {4: 0b10011, 5: 0b100101, 6: 0b1000011, 7: 0b10001001}


class BCH:
    def __init__(self, m: int = 6, t: int = 6):
        self.m, self.t = m, t
        self.n = (1 << m) - 1

        # GF(2^m) exp/log tables
        self._exp = [0] * (2 * self.n)
        self._log = [0] * (self.n + 1)
        x = 1
        for i in range(self.n):
            self._exp[i] = x
            self._log[x] = i
            x <<= 1
            if x & (1 << m):
                x ^= _PRIMITIVE[m]
        for i in range(self.n, 2 * self.n):
            self._exp[i] = self._exp[i - self.n]

        # Generator polynomial g(x) = lcm of minimal polynomials of a^1..a^2t.
        # GF(2)[x] polynomials are stored as ints (bit i = coefficient of x^i).
        self.g = 1
        covered = set()
        for i in range(1, 2 * t + 1):
            if i in covered:
                continue
            conjugates, c = [], i
            while c not in conjugates:
                conjugates.append(c)
                c = (c * 2) % self.n
            covered.update(conjugates)
            self.g = self._gf2_mul(self.g, self._minimal_poly(conjugates))
        self.k = self.n - (self.g.bit_length() - 1)

    # ─── field / polynomial helpers ───────────────────────────────

    def _mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self._exp[self._log[a] + self._log[b]]

    def _inv(self, a: int) -> int:
        return self._exp[self.n - self._log[a]]

    def _minimal_poly(self, conjugates: list[int]) -> int:
        """prod (x - a^c) over the conjugacy class; coefficients end up in GF(2)."""
        poly = [1]  # poly[i] = coefficient of x^i, in GF(2^m)
        for c in conjugates:
            root = self._exp[c]
            nxt = [0] * (len(poly) + 1)
            for i, coef in enumerate(poly):
                nxt[i + 1] ^= coef
                nxt[i] ^= self._mul(coef, root)
            poly = nxt
        return sum(1 << i for i, coef in enumerate(poly) if coef)

    @staticmethod
    def _gf2_mul(a: int, b: int) -> int:
        out = 0
        while b:
            if b & 1:
                out ^= a
            a <<= 1
            b >>= 1
        return out

    def _gf2_mod(self, a: int) -> int:
        deg_g = self.g.bit_length() - 1
        while a.bit_length() - 1 >= deg_g:
            a ^= self.g << (a.bit_length() - 1 - deg_g)
        return a

    # Bit arrays are ordered highest-degree first, so a codeword array is
    # [k message bits | n-k parity bits].
    def _to_int(self, bits) -> int:
        out = 0
        for b in bits:
            out = (out << 1) | int(b)
        return out

    def _to_bits(self, value: int, length: int) -> np.ndarray:
        return np.array([(value >> (length - 1 - i)) & 1 for i in range(length)],
                        dtype=np.uint8)

    # ─── encode / decode ──────────────────────────────────────────

    def encode(self, message_bits) -> np.ndarray:
        """Systematic encoding: k message bits -> n codeword bits."""
        if len(message_bits) != self.k:
            raise ValueError(f"expected {self.k} message bits, got {len(message_bits)}")
        shifted = self._to_int(message_bits) << (self.n - self.k)
        return self._to_bits(shifted ^ self._gf2_mod(shifted), self.n)

    def decode(self, received_bits) -> tuple[np.ndarray | None, int]:
        """
        Correct up to t errors.
        Returns (message_bits, n_errors_corrected), or (None, -1) when the
        word is more than t errors away from every codeword (decoding failure).
        """
        if len(received_bits) != self.n:
            raise ValueError(f"expected {self.n} bits, got {len(received_bits)}")
        r = self._to_int(received_bits)
        positions = [i for i in range(self.n) if (r >> i) & 1]

        # Syndromes S_j = r(a^j), j = 1..2t
        synd = []
        for j in range(1, 2 * self.t + 1):
            s = 0
            for i in positions:
                s ^= self._exp[(i * j) % self.n]
            synd.append(s)
        if not any(synd):
            return self._to_bits(r >> (self.n - self.k), self.k), 0

        # Berlekamp-Massey: error locator polynomial sigma(x)
        C, B = [1], [1]
        L, shift, b = 0, 1, 1
        for i in range(2 * self.t):
            d = synd[i]
            for j in range(1, L + 1):
                if j < len(C):
                    d ^= self._mul(C[j], synd[i - j])
            if d == 0:
                shift += 1
                continue
            coef = self._mul(d, self._inv(b))
            T = C[:]
            C = C + [0] * (len(B) + shift - len(C))
            for j, bj in enumerate(B):
                C[j + shift] ^= self._mul(coef, bj)
            if 2 * L <= i:
                L, B, b, shift = i + 1 - L, T, d, 1
            else:
                shift += 1
        if L > self.t:
            return None, -1

        # Chien search: error at position i  <=>  sigma(a^-i) = 0
        errors = []
        for i in range(self.n):
            val = 0
            for j, cj in enumerate(C[:L + 1]):
                if cj:
                    val ^= self._exp[(self._log[cj] + j * (self.n - i)) % self.n]
            if val == 0:
                errors.append(i)
        if len(errors) != L:
            return None, -1

        for i in errors:
            r ^= 1 << i
        if self._gf2_mod(r) != 0:
            return None, -1
        return self._to_bits(r >> (self.n - self.k), self.k), L
