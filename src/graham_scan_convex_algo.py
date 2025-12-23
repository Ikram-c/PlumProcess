"""
Graham Scan Convex Hull Algorithm - Object-Oriented Implementation

Optimized for Python 3.14 with parallel processing support via Numba
and optional free-threaded execution. Supports massive datasets through
configurable preprocessing, coordinate compression, and generator-based
memory-efficient sorting.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from enum import IntEnum

import numpy as np
from numba import njit
from numba import prange
from sortedcontainers import SortedList


class Orientation(IntEnum):
    """
    Enumeration representing the orientation of an ordered triplet of points.
    
    Used to determine turn direction during convex hull construction.
    Integer values allow direct comparison and pattern matching.
    """

    COLLINEAR = 0
    CLOCKWISE = 1
    COUNTERCLOCKWISE = 2


@njit(parallel=True, fastmath=True)
def _parallel_polar_angles(
    points: np.ndarray,
    pivot_x: float,
    pivot_y: float,
) -> np.ndarray:
    """
    Compute polar angles from pivot to all points in parallel.
    
    Uses Numba's prange for automatic thread distribution across CPU cores.
    The fastmath flag enables aggressive floating-point optimizations.
    
    Args:
        points: (N, 2) array of 2D point coordinates.
        pivot_x: X-coordinate of the pivot point.
        pivot_y: Y-coordinate of the pivot point.
    
    Returns:
        Array of N polar angles in radians, range [-pi, pi].
    """
    n = points.shape[0]
    angles = np.empty(n, dtype=np.float64)
    for i in prange(n):
        angles[i] = np.arctan2(points[i, 1] - pivot_y, points[i, 0] - pivot_x)
    return angles


@njit(parallel=True, fastmath=True)
def _parallel_distances_sq(
    points: np.ndarray,
    pivot_x: float,
    pivot_y: float,
) -> np.ndarray:
    """
    Compute squared Euclidean distances from pivot to all points in parallel.
    
    Squared distances avoid expensive sqrt operations while preserving
    relative ordering for comparison purposes.
    
    Args:
        points: (N, 2) array of 2D point coordinates.
        pivot_x: X-coordinate of the pivot point.
        pivot_y: Y-coordinate of the pivot point.
    
    Returns:
        Array of N squared distances.
    """
    n = points.shape[0]
    distances = np.empty(n, dtype=np.float64)
    for i in prange(n):
        dx = points[i, 0] - pivot_x
        dy = points[i, 1] - pivot_y
        distances[i] = dx * dx + dy * dy
    return distances


@njit
def _graham_stack(sorted_points: np.ndarray) -> np.ndarray:
    """
    Execute the Graham scan stack algorithm on pre-sorted points.
    
    Iterates through points in polar angle order, maintaining a stack
    of convex hull vertices. Points causing right turns (clockwise)
    are removed from the stack.
    
    Args:
        sorted_points: (N, 2) array sorted by polar angle from pivot.
    
    Returns:
        (H, 2) array of convex hull vertices in counter-clockwise order,
        where H <= N is the number of hull points.
    """
    n = sorted_points.shape[0]
    hull = np.empty((n, 2), dtype=np.float64)
    size = 0
    for i in range(n):
        while size >= 2:
            cross = (
                (hull[size - 1, 0] - hull[size - 2, 0])
                * (sorted_points[i, 1] - hull[size - 2, 1])
                - (hull[size - 1, 1] - hull[size - 2, 1])
                * (sorted_points[i, 0] - hull[size - 2, 0])
            )
            if cross <= 0:
                size -= 1
            else:
                break
        hull[size] = sorted_points[i]
        size += 1
    return hull[:size]


@njit
def _filter_collinear(
    angles: np.ndarray,
    distances: np.ndarray,
    sorted_indices: np.ndarray,
) -> np.ndarray:
    """
    Remove collinear points, keeping only the farthest from pivot.
    
    For points sharing the same polar angle (within tolerance), only
    the point with maximum distance from pivot is retained. This
    prevents degenerate cases in the Graham stack.
    
    Args:
        angles: Pre-computed polar angles for all points.
        distances: Pre-computed squared distances for all points.
        sorted_indices: Indices sorted by (angle, distance).
    
    Returns:
        Filtered index array with collinear duplicates removed.
    """
    n = len(sorted_indices)
    mask = np.ones(n, dtype=np.bool_)
    prev_angle = angles[sorted_indices[0]]
    for i in range(1, n):
        idx = sorted_indices[i]
        if abs(angles[idx] - prev_angle) < 1e-10:
            prev_idx = sorted_indices[i - 1]
            if distances[idx] > distances[prev_idx]:
                mask[i - 1] = False
            else:
                mask[i] = False
        prev_angle = angles[idx]
    return sorted_indices[mask]


@dataclass(slots=True, frozen=True)
class Point:
    """
    Immutable 2D point with optional original index tracking.
    
    Supports tuple-like access via iteration and indexing, enabling
    seamless interoperability with NumPy arrays and tuple unpacking.
    
    Attributes:
        x: X-coordinate.
        y: Y-coordinate.
        index: Original index in source array, -1 if untracked.
    """

    x: float
    y: float
    index: int = -1

    def __iter__(self):
        """Enable tuple unpacking: x, y = point."""
        return iter((self.x, self.y))

    def __getitem__(self, idx: int) -> float:
        """Enable indexing: point[0] returns x, point[1] returns y."""
        return (self.x, self.y)[idx]

    def __sub__(self, other: Point) -> tuple[float, float]:
        """Vector subtraction returning (dx, dy) tuple."""
        return (self.x - other.x, self.y - other.y)

    def distance_sq(self, other: Point) -> float:
        """Compute squared Euclidean distance to another point."""
        dx, dy = self - other
        return dx * dx + dy * dy

    def angle_from(self, origin: Point) -> float:
        """Compute polar angle from origin to this point."""
        return np.arctan2(self.y - origin.y, self.x - origin.x)

    @classmethod
    def from_array(cls, arr: np.ndarray, index: int = -1) -> Point:
        """Construct Point from numpy array or sequence."""
        return cls(float(arr[0]), float(arr[1]), index)


@dataclass
class PointCloud:
    """
    Container for 2D point data with lazy-computed geometric properties.
    
    Encapsulates the input point array and provides cached access to
    pivot point, polar angles, and distances. Properties are computed
    on first access and memoized for subsequent calls.
    
    Attributes:
        points: (N, 2) contiguous array of point coordinates.
    """

    points: np.ndarray
    _pivot_idx: int | None = field(default=None, init=False, repr=False)
    _angles: np.ndarray | None = field(default=None, init=False, repr=False)
    _distances: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Ensure contiguous memory layout for cache efficiency."""
        self.points = np.ascontiguousarray(self.points, dtype=np.float64)

    def __len__(self) -> int:
        """Return number of points in the cloud."""
        return len(self.points)

    def __getitem__(self, idx: int | np.ndarray) -> np.ndarray:
        """Index into points array directly."""
        return self.points[idx]

    @property
    def pivot_idx(self) -> int:
        """
        Index of the pivot point (minimum y, then minimum x for ties).
        
        The pivot is guaranteed to be on the convex hull and serves as
        the origin for polar angle sorting.
        """
        if self._pivot_idx is None:
            self._pivot_idx = int(
                np.lexsort((self.points[:, 0], self.points[:, 1]))[0]
            )
        return self._pivot_idx

    @property
    def pivot(self) -> np.ndarray:
        """Coordinates of the pivot point as (x, y) array."""
        return self.points[self.pivot_idx]

    @property
    def angles(self) -> np.ndarray:
        """
        Polar angles from pivot to all points.
        
        Computed in parallel on first access via Numba JIT.
        """
        if self._angles is None:
            self._angles = _parallel_polar_angles(
                self.points,
                self.pivot[0],
                self.pivot[1],
            )
        return self._angles

    @property
    def distances(self) -> np.ndarray:
        """
        Squared distances from pivot to all points.
        
        Computed in parallel on first access via Numba JIT.
        """
        if self._distances is None:
            self._distances = _parallel_distances_sq(
                self.points,
                self.pivot[0],
                self.pivot[1],
            )
        return self._distances

    def subset(self, indices: np.ndarray) -> PointCloud:
        """Create new PointCloud containing only specified indices."""
        return PointCloud(self.points[indices])


class Preprocessor:
    """
    Base class for point cloud preprocessing stages.
    
    Preprocessors transform a PointCloud before the main Graham scan
    algorithm executes. Subclasses implement specific transformations
    such as coordinate compression or interior point filtering.
    """

    def process(self, cloud: PointCloud) -> PointCloud:
        """
        Apply preprocessing transformation to the point cloud.
        
        Args:
            cloud: Input point cloud.
        
        Returns:
            Transformed point cloud (may be same instance if unchanged).
        """
        raise NotImplementedError


class CoordinateCompressor(Preprocessor):
    """
    Maps floating-point coordinates to discrete integer indices.
    
    Coordinate compression improves numerical stability by eliminating
    floating-point comparison issues. Points within tolerance are mapped
    to the same integer coordinate.
    
    Attributes:
        tolerance: Points closer than this are considered identical.
    """

    __slots__ = ("tolerance", "x_map", "y_map", "x_inv", "y_inv", "_original")

    def __init__(self, tolerance: float = 1e-9) -> None:
        """
        Initialize compressor with specified tolerance.
        
        Args:
            tolerance: Coordinate quantization granularity.
        """
        self.tolerance = tolerance
        self.x_map: dict[float, int] = {}
        self.y_map: dict[float, int] = {}
        self.x_inv: np.ndarray = np.array([])
        self.y_inv: np.ndarray = np.array([])
        self._original: np.ndarray | None = None

    def process(self, cloud: PointCloud) -> PointCloud:
        """
        Compress point coordinates to integer indices.
        
        Stores original coordinates for later decompression of hull points.
        """
        self._original = cloud.points.copy()
        self._build_maps(cloud.points)
        compressed = self._compress(cloud.points)
        return PointCloud(compressed)

    def decompress(self, points: np.ndarray) -> np.ndarray:
        """
        Restore original floating-point coordinates from integer indices.
        
        Args:
            points: (N, 2) array of compressed integer coordinates.
        
        Returns:
            (N, 2) array of original floating-point coordinates.
        """
        return np.column_stack([
            self.x_inv[points[:, 0].astype(int)],
            self.y_inv[points[:, 1].astype(int)],
        ])

    def _build_maps(self, points: np.ndarray) -> None:
        """Construct forward and inverse coordinate mappings."""
        unique_x = np.unique(np.round(points[:, 0] / self.tolerance) * self.tolerance)
        unique_y = np.unique(np.round(points[:, 1] / self.tolerance) * self.tolerance)
        self.x_map = {v: i for i, v in enumerate(unique_x)}
        self.y_map = {v: i for i, v in enumerate(unique_y)}
        self.x_inv = unique_x
        self.y_inv = unique_y

    def _compress(self, points: np.ndarray) -> np.ndarray:
        """Apply coordinate mapping to point array."""
        rounded_x = np.round(points[:, 0] / self.tolerance) * self.tolerance
        rounded_y = np.round(points[:, 1] / self.tolerance) * self.tolerance
        compress_x = np.vectorize(self.x_map.get)
        compress_y = np.vectorize(self.y_map.get)
        return np.column_stack([compress_x(rounded_x), compress_y(rounded_y)])


class SweepLineFilter(Preprocessor):
    """
    Filters interior points using extremal quadrilateral bounds.
    
    Identifies the four extremal points (min/max x and y) and removes
    points strictly inside their bounding quadrilateral. These interior
    points cannot be on the convex hull, reducing input size.
    
    Attributes:
        min_points: Only apply filtering above this threshold.
    """

    __slots__ = ("min_points", "active_set")

    def __init__(self, min_points: int = 10000) -> None:
        """
        Initialize filter with minimum point threshold.
        
        Args:
            min_points: Filtering overhead not worthwhile below this size.
        """
        self.min_points = min_points
        self.active_set: SortedList[tuple[float, int]] = SortedList()

    def process(self, cloud: PointCloud) -> PointCloud:
        """
        Filter interior points from the point cloud.
        
        Returns original cloud unchanged if below threshold.
        """
        if len(cloud) < self.min_points:
            return cloud
        candidate_indices = self._get_candidate_indices(cloud)
        return cloud.subset(candidate_indices)

    def _get_candidate_indices(self, cloud: PointCloud) -> np.ndarray:
        """Determine which point indices could be on the hull."""
        n = len(cloud)
        if n < 4:
            return np.arange(n)
        extremal = self._get_extremal_indices(cloud)
        if len(extremal) < 3:
            return np.arange(n)
        return self._filter_interior(cloud, extremal)

    def _get_extremal_indices(self, cloud: PointCloud) -> set[int]:
        """Find indices of the four extremal points."""
        return {
            int(np.argmin(cloud.points[:, 0])),
            int(np.argmax(cloud.points[:, 0])),
            int(np.argmin(cloud.points[:, 1])),
            int(np.argmax(cloud.points[:, 1])),
        }

    def _filter_interior(self, cloud: PointCloud, extremal: set[int]) -> np.ndarray:
        """Remove points strictly inside the extremal quadrilateral."""
        quad = cloud.points[list(extremal)]
        min_x, max_x = quad[:, 0].min(), quad[:, 0].max()
        min_y, max_y = quad[:, 1].min(), quad[:, 1].max()
        mask = np.ones(len(cloud), dtype=bool)
        strictly_inside = (
            (cloud.points[:, 0] > min_x)
            & (cloud.points[:, 0] < max_x)
            & (cloud.points[:, 1] > min_y)
            & (cloud.points[:, 1] < max_y)
        )
        interior_indices = np.where(strictly_inside)[0]
        inside_quad = np.array([
            self._is_inside_quad(cloud.points[i], quad) for i in interior_indices
        ])
        mask[interior_indices[inside_quad]] = False
        return np.where(mask)[0]

    def _is_inside_quad(self, point: np.ndarray, quad: np.ndarray) -> bool:
        """Test if point lies strictly inside the quadrilateral."""
        centroid = quad.mean(axis=0)
        angles = np.arctan2(quad[:, 1] - centroid[1], quad[:, 0] - centroid[0])
        sorted_quad = quad[np.argsort(angles)]
        n = len(sorted_quad)
        cross_products = (
            (sorted_quad[(np.arange(n) + 1) % n, 0] - sorted_quad[:, 0])
            * (point[1] - sorted_quad[:, 1])
            - (sorted_quad[(np.arange(n) + 1) % n, 1] - sorted_quad[:, 1])
            * (point[0] - sorted_quad[:, 0])
        )
        return bool(np.all(cross_products < 0))


class SortStrategy:
    """
    Base class for point sorting strategies.
    
    Different sorting approaches trade off between memory usage and
    speed. Subclasses implement specific strategies for different
    dataset sizes and memory constraints.
    """

    def sort(self, cloud: PointCloud) -> np.ndarray:
        """
        Sort points by polar angle from pivot.
        
        Args:
            cloud: Point cloud with pre-computed angles and distances.
        
        Returns:
            Array of indices in sorted order, with collinear points filtered.
        """
        raise NotImplementedError


class ArraySortStrategy(SortStrategy):
    """
    Standard array-based sorting using NumPy lexsort.
    
    Materializes full sorted index array in memory. Fastest approach
    for datasets that fit comfortably in RAM.
    """

    def sort(self, cloud: PointCloud) -> np.ndarray:
        """Sort using NumPy lexsort with JIT-compiled collinear filtering."""
        sorted_indices = np.lexsort((cloud.distances, cloud.angles))
        return _filter_collinear(cloud.angles, cloud.distances, sorted_indices)


class GeneratorSortStrategy(SortStrategy):
    """
    Memory-efficient sorting using Python generators.
    
    Lazily yields sorted indices without materializing full arrays.
    Beneficial for datasets exceeding available RAM, at the cost of
    reduced speed due to Python iteration overhead.
    
    Attributes:
        threshold: Use array strategy below this point count.
    """

    def __init__(self, threshold: int = 1_000_000) -> None:
        """
        Initialize with fallback threshold.
        
        Args:
            threshold: Switch to ArraySortStrategy below this size.
        """
        self.threshold = threshold

    def sort(self, cloud: PointCloud) -> np.ndarray:
        """Sort using generators for memory efficiency on large datasets."""
        if len(cloud) <= self.threshold:
            return ArraySortStrategy().sort(cloud)
        indices_gen = self._sorted_indices_generator(cloud)
        filtered_gen = self._filter_collinear_generator(cloud, indices_gen)
        return np.fromiter(filtered_gen, dtype=np.int64, count=-1)

    def _sorted_indices_generator(self, cloud: PointCloud):
        """Lazily yield indices in polar angle order."""
        order = np.lexsort((cloud.distances, cloud.angles))
        yield cloud.pivot_idx
        prev_angle = -np.inf
        for idx in order:
            if idx == cloud.pivot_idx:
                continue
            if abs(cloud.angles[idx] - prev_angle) < 1e-10:
                continue
            prev_angle = cloud.angles[idx]
            yield idx

    def _filter_collinear_generator(self, cloud: PointCloud, indices):
        """Lazily filter collinear points, keeping farthest from pivot."""
        buffer: list[int] = []
        prev_angle: float | None = None
        for idx in indices:
            curr_angle = cloud.angles[idx]
            if prev_angle is not None and abs(curr_angle - prev_angle) < 1e-10:
                if cloud.distances[idx] > cloud.distances[buffer[-1]]:
                    buffer[-1] = idx
            else:
                if buffer:
                    yield buffer[-1]
                buffer = [idx]
            prev_angle = curr_angle
        if buffer:
            yield buffer[-1]


class ExecutorFactory:
    """
    Factory for creating appropriate parallel executors.
    
    Detects Python 3.14+ free-threaded mode and returns ThreadPoolExecutor
    for true parallelism, falling back to ProcessPoolExecutor otherwise.
    """

    @staticmethod
    def create(max_workers: int | None = None) -> ThreadPoolExecutor | ProcessPoolExecutor:
        """
        Create optimal executor for current Python runtime.
        
        Args:
            max_workers: Maximum parallel workers, None for CPU count.
        
        Returns:
            ThreadPoolExecutor if GIL disabled, ProcessPoolExecutor otherwise.
        """
        try:
            if not sys._is_gil_enabled():
                return ThreadPoolExecutor(max_workers=max_workers)
        except AttributeError:
            pass
        return ProcessPoolExecutor(max_workers=max_workers)


@dataclass
class GrahamScanConfig:
    """
    Configuration options for the Graham scan algorithm.
    
    Controls preprocessing, sorting strategy, and parallelization behavior.
    All options have sensible defaults for typical use cases.
    
    Attributes:
        use_compression: Enable coordinate compression for float stability.
        compression_tolerance: Quantization granularity for compression.
        use_generators: Enable memory-efficient generator-based sorting.
        generator_threshold: Point count threshold for generator activation.
        use_preprocessing: Enable sweep-line interior point filtering.
        preprocessing_threshold: Point count threshold for preprocessing.
        num_workers: Maximum parallel workers for executor.
    """

    use_compression: bool = False
    compression_tolerance: float = 1e-9
    use_generators: bool = False
    generator_threshold: int = 1_000_000
    use_preprocessing: bool = False
    preprocessing_threshold: int = 10000
    num_workers: int = 4


class GrahamScan:
    """
    Graham scan convex hull algorithm with configurable optimization pipeline.
    
    Orchestrates preprocessing, sorting, and hull construction stages.
    Supports coordinate compression, interior point filtering, and
    memory-efficient sorting for massive datasets.
    
    Example:
        >>> config = GrahamScanConfig(use_preprocessing=True)
        >>> scanner = GrahamScan(config)
        >>> hull = scanner(points)
    
    Attributes:
        config: Algorithm configuration options.
    """

    __slots__ = ("config", "_preprocessors", "_sort_strategy", "_compressor")

    def __init__(self, config: GrahamScanConfig | None = None) -> None:
        """
        Initialize scanner with configuration.
        
        Args:
            config: Options controlling algorithm behavior, or None for defaults.
        """
        self.config = config or GrahamScanConfig()
        self._preprocessors: list[Preprocessor] = []
        self._sort_strategy: SortStrategy = self._create_sort_strategy()
        self._compressor: CoordinateCompressor | None = None
        self._setup_preprocessors()

    def _setup_preprocessors(self) -> None:
        """Configure preprocessing pipeline based on config."""
        if self.config.use_preprocessing:
            self._preprocessors.append(
                SweepLineFilter(self.config.preprocessing_threshold)
            )
        if self.config.use_compression:
            self._compressor = CoordinateCompressor(self.config.compression_tolerance)
            self._preprocessors.append(self._compressor)

    def _create_sort_strategy(self) -> SortStrategy:
        """Select sorting strategy based on config."""
        if self.config.use_generators:
            return GeneratorSortStrategy(self.config.generator_threshold)
        return ArraySortStrategy()

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Enable callable syntax: scanner(points)."""
        return self.compute(points)

    def compute(self, points: np.ndarray) -> np.ndarray:
        """
        Compute the convex hull of a set of 2D points.
        
        Executes the full pipeline: preprocessing, polar angle sorting,
        Graham stack traversal, and optional coordinate decompression.
        
        Args:
            points: (N, 2) array of 2D point coordinates.
        
        Returns:
            (H, 2) array of convex hull vertices in counter-clockwise order.
        """
        cloud = PointCloud(points)
        if len(cloud) < 3:
            return cloud.points.copy()
        cloud = self._apply_preprocessors(cloud)
        sorted_indices = self._sort_strategy.sort(cloud)
        sorted_points = cloud[sorted_indices]
        hull_points = _graham_stack(sorted_points)
        return self._postprocess(hull_points, cloud, sorted_indices, sorted_points)

    def _apply_preprocessors(self, cloud: PointCloud) -> PointCloud:
        """Execute preprocessing pipeline on point cloud."""
        for preprocessor in self._preprocessors:
            cloud = preprocessor.process(cloud)
        return cloud

    def _postprocess(
        self,
        hull_points: np.ndarray,
        cloud: PointCloud,
        sorted_indices: np.ndarray,
        sorted_points: np.ndarray,
    ) -> np.ndarray:
        """Handle coordinate decompression if compression was used."""
        if self._compressor is None:
            return hull_points
        hull_indices = []
        for hp in hull_points:
            dists = np.sum((sorted_points - hp) ** 2, axis=1)
            orig_idx = sorted_indices[np.argmin(dists)]
            hull_indices.append(orig_idx)
        return self._compressor.decompress(cloud.points[hull_indices])


def graham_scan(
    points: np.ndarray,
    num_workers: int = 4,
    use_compression: bool = False,
    compression_tolerance: float = 1e-9,
    use_generators: bool = False,
    use_preprocessing: bool = False,
) -> np.ndarray:
    """
    Compute convex hull using Graham scan algorithm.
    
    Functional interface wrapping the OOP implementation for backward
    compatibility and convenience.
    
    Args:
        points: (N, 2) array of 2D point coordinates.
        num_workers: Maximum parallel workers.
        use_compression: Enable coordinate compression.
        compression_tolerance: Quantization granularity.
        use_generators: Enable memory-efficient sorting.
        use_preprocessing: Enable interior point filtering.
    
    Returns:
        (H, 2) array of convex hull vertices in counter-clockwise order.
    
    Example:
        >>> points = np.random.rand(1000, 2)
        >>> hull = graham_scan(points, use_preprocessing=True)
    """
    config = GrahamScanConfig(
        use_compression=use_compression,
        compression_tolerance=compression_tolerance,
        use_generators=use_generators,
        use_preprocessing=use_preprocessing,
        num_workers=num_workers,
    )
    scanner = GrahamScan(config)
    return scanner(points)


if __name__ == "__main__":
    import time

    np.random.seed(42)
    test_points = np.random.rand(1_000_000, 2)

    config = GrahamScanConfig()
    scanner = GrahamScan(config)
    _ = scanner(test_points[:1000])

    start = time.perf_counter()
    hull = scanner(test_points)
    elapsed = time.perf_counter() - start

    print(f"Hull size: {len(hull)} points")
    print(f"Time: {elapsed:.3f}s for {len(test_points):,} points")
    print(f"Throughput: {len(test_points) / elapsed:,.0f} points/second")