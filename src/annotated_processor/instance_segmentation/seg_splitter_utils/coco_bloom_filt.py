import math
from typing import Any, Dict, List

import mmh3
from bitarray import bitarray


class COCOAnnotationBloomFilter:

    def __init__(
            self,
            capacity: int,
            false_positive_rate: float = 0.001,
            use_double_hashing: bool = True
    ):
        self._validate_capacity(capacity)
        self._validate_fp_rate(false_positive_rate)

        self.capacity = capacity
        self.fp_rate = false_positive_rate
        self.m = self._calculate_size(capacity, false_positive_rate)
        self.k = self._calculate_hash_count(self.m, capacity)
        self.bit_array = bitarray(self.m)
        self.bit_array.setall(0)
        self.items_count = 0
        self.use_double_hashing = use_double_hashing

    def _validate_capacity(self, capacity: int) -> None:
        is_invalid = capacity <= 0
        if is_invalid:
            raise ValueError(f"Capacity must be positive, got {capacity}")

    def _validate_fp_rate(self, false_positive_rate: float) -> None:
        is_too_low = false_positive_rate <= 0
        is_too_high = false_positive_rate >= 1
        is_invalid = is_too_low or is_too_high
        if is_invalid:
            raise ValueError(
                f"False positive rate must be in (0,1), got {false_positive_rate}"
            )

    @staticmethod
    def _calculate_size(n: int, p: float) -> int:
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(math.ceil(m))

    @staticmethod
    def _calculate_hash_count(m: int, n: int) -> int:
        k = (m / n) * math.log(2)
        return max(1, int(math.ceil(k)))

    def _get_positions_double_hash(self, item_str: str) -> List[int]:
        h1 = mmh3.hash(item_str, seed=0)
        h2 = mmh3.hash(item_str, seed=1)
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def _get_positions_multi_hash(self, item_str: str) -> List[int]:
        return [mmh3.hash(item_str, seed=i) % self.m for i in range(self.k)]

    def _get_positions(self, item: Any) -> List[int]:
        item_str = str(item)
        if self.use_double_hashing:
            return self._get_positions_double_hash(item_str)
        return self._get_positions_multi_hash(item_str)

    def add(self, item: Any) -> bool:
        positions = self._get_positions(item)
        is_new = any(self.bit_array[pos] == 0 for pos in positions)

        for pos in positions:
            self.bit_array[pos] = 1

        if is_new:
            self.items_count += 1

        return is_new

    def might_exist(self, item: Any) -> bool:
        positions = self._get_positions(item)
        return all(self.bit_array[pos] == 1 for pos in positions)

    def definitely_new(self, item: Any) -> bool:
        exists = self.might_exist(item)
        is_new = exists == False
        return is_new

    def __contains__(self, item: Any) -> bool:
        return self.might_exist(item)

    def get_stats(self) -> Dict[str, Any]:
        bits_set = self.bit_array.count()
        fill_ratio = self._calculate_fill_ratio(bits_set)
        actual_fp_rate = fill_ratio ** self.k

        return {
            'capacity': self.capacity,
            'items_added': self.items_count,
            'utilization': self._calculate_utilization(),
            'bit_array_size': self.m,
            'bits_set': bits_set,
            'fill_ratio': fill_ratio,
            'hash_functions': self.k,
            'target_fp_rate': self.fp_rate,
            'actual_fp_rate': actual_fp_rate,
            'memory_bytes': self.m // 8,
            'memory_mb': (self.m // 8) / (1024 * 1024),
            'bits_per_item': self._calculate_bits_per_item()
        }

    def _calculate_fill_ratio(self, bits_set: int) -> float:
        has_bits = self.m > 0
        if has_bits:
            return bits_set / self.m
        return 0

    def _calculate_utilization(self) -> float:
        has_capacity = self.capacity > 0
        if has_capacity:
            return self.items_count / self.capacity
        return 0

    def _calculate_bits_per_item(self) -> float:
        has_capacity = self.capacity > 0
        if has_capacity:
            return self.m / self.capacity
        return 0

    def should_resize(self, threshold: float = 0.7) -> bool:
        limit = self.capacity * threshold
        return self.items_count >= limit

    def export_state(self) -> Dict[str, Any]:
        return {
            'capacity': self.capacity,
            'false_positive_rate': self.fp_rate,
            'use_double_hashing': self.use_double_hashing,
            'items_count': self.items_count,
            'bit_array': self.bit_array.tolist()
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'COCOAnnotationBloomFilter':
        bf = cls(
            capacity=state['capacity'],
            false_positive_rate=state['false_positive_rate'],
            use_double_hashing=state['use_double_hashing']
        )
        bf.items_count = state['items_count']
        bf.bit_array = bitarray(state['bit_array'])
        return bf


def calculate_bloom_parameters(
        num_annotations: int,
        false_positive_rate: float = 0.001
) -> Dict[str, Any]:
    m = COCOAnnotationBloomFilter._calculate_size(num_annotations, false_positive_rate)
    k = COCOAnnotationBloomFilter._calculate_hash_count(m, num_annotations)

    return {
        'num_annotations': num_annotations,
        'false_positive_rate': false_positive_rate,
        'bit_array_size': m,
        'hash_functions': k,
        'memory_bytes': m // 8,
        'memory_mb': (m // 8) / (1024 * 1024),
        'bits_per_annotation': m / num_annotations
    }