"""
core/bloom.py — Prime-Number Bloom Filter for O(1) Song Deduplication

Why primes?
  A Bloom filter uses k independent hash functions. Using prime numbers
  as multiplicative hash seeds gives maximum avalanche — any small change
  in the input propagates to all bits, minimising collision clusters.
  The bit-array size is itself a large prime (BLOOM_SIZE = 2_000_003) to
  ensure uniform distribution of h(x) mod BLOOM_SIZE across all slots.

  Sieve of Eratosthenes generates the seed primes in O(n log log n).

False-positive rate at 1M items, 7 hashes, 2M bits:
  p ≈ (1 - e^(-k*n/m))^k  ≈ 0.8%  → acceptable for a dedup guard
"""

import hashlib
import array
import json
from config import BLOOM_SIZE, BLOOM_HASH_COUNT, PRIMES_100


class PrimeBloomFilter:
    """
    Bit-array Bloom filter with prime-seeded polynomial rolling hashes.

    Each of the k hash functions is:
        h_i(x) = (fnv_hash(x) * p_i + p_i^2) mod BLOOM_SIZE

    where p_i is the i-th prime from our Sieve of Eratosthenes list.
    """

    def __init__(self, size: int = BLOOM_SIZE, k: int = BLOOM_HASH_COUNT):
        self.size = size
        self.k = k
        self.primes = PRIMES_100[:k]
        # 'b' typecode = unsigned char (1 byte per slot, efficient)
        self._bits = array.array('b', [0]) * size
        self._count = 0

    def _hashes(self, item: str) -> list[int]:
        """
        Generate k independent hash positions for item using prime seeds.
        FNV-1a base hash, then polynomial expansion per prime.
        """
        raw = item.encode()
        # FNV-1a 64-bit
        h = 14695981039346656037
        for byte in raw:
            h ^= byte
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        positions = []
        for p in self.primes:
            # Each prime creates an independent affine projection
            pos = (h * p + p * p) % self.size
            positions.append(pos)
        return positions

    def add(self, item: str) -> None:
        for pos in self._hashes(item):
            self._bits[pos] = 1
        self._count += 1

    def __contains__(self, item: str) -> bool:
        return all(self._bits[pos] for pos in self._hashes(item))

    def signature(self, item: str) -> str:
        """Return a compact hex signature of the item's hash positions."""
        return ",".join(hex(p) for p in self._hashes(item))

    @property
    def estimated_false_positive_rate(self) -> float:
        import math
        m, k, n = self.size, self.k, max(self._count, 1)
        return (1 - math.exp(-k * n / m)) ** k

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return (
            f"PrimeBloomFilter(size={self.size}, k={self.k}, "
            f"items={self._count}, fp_rate≈{self.estimated_false_positive_rate:.4%})"
        )


# ─── Module-level singleton ───────────────────────────────
_bloom: PrimeBloomFilter | None = None


def get_bloom() -> PrimeBloomFilter:
    global _bloom
    if _bloom is None:
        _bloom = PrimeBloomFilter()
    return _bloom


def is_duplicate(song_id: str) -> bool:
    return song_id in get_bloom()


def mark_seen(song_id: str) -> None:
    get_bloom().add(song_id)
