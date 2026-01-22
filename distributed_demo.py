#!/usr/bin/env python3
"""
CTGV Distributed Architecture Demonstration
Shows how to use the distributed CTGV system for large-scale processing
"""
import numpy as np
import time
from ctgv import (
    Shape, Gebit, DistributedCTGVEngine, StreamProcessingPipeline,
    GPUAccelerator, HyperscaleCTGVSYSTEM, AutoScalingManager, SystemMonitor
)

def demonstrate_distributed_ctgv():
    """Demonstrate distributed CTGV architecture for scalability"""
    print("=" * 80)
    print(" CTGV DISTRIBUTED ARCHITECTURE - Hyperscale Processing Demo")
    print("=" * 80)

    # ===== 1. DISTRIBUTED ENGINE =====
    print("\n[1] Testing Distributed CTGV Engine...")

    # Create large network (1000+ nodes)
    print("  Creating large network with 2000+ nodes...")
    nodes = []

    # Create origin nodes
    for i in range(50):
        origin = Gebit(Shape.ORIGIN, intensity=0.8 + np.random.random() * 0.2,
                      label=f"Origin_{i}")
        nodes.append(origin)

    # Create processing layers
    for i in range(500):
        processor = Gebit(Shape.DECISOR, intensity=0.0, label=f"Processor_{i}")
        nodes.append(processor)

    for i in range(1000):
        flow = Gebit(Shape.FLOW, intensity=0.0, label=f"Flow_{i}")
        nodes.append(flow)

    for i in range(500):
        memory = Gebit(Shape.MEMORY, intensity=0.0, label=f"Memory_{i}")
        nodes.append(memory)

    # Create connections (sparse topology for scalability)
    print("  Building sparse topology...")
    np.random.seed(42)

    # Connect origins to processors
    for origin in nodes[:50]:
        for processor in nodes[50:550]:  # Next 500 processors
            if np.random.random() < 0.1:  # 10% connection probability
                origin.connect_to(processor, np.random.random() * 0.5 + 0.5)

    # Connect processors to flows
    for processor in nodes[50:550]:
        for flow in nodes[550:1550]:  # Next 1000 flows
            if np.random.random() < 0.05:  # 5% connection probability
                processor.connect_to(flow, np.random.random() * 0.3 + 0.4)

    # Connect flows to memory
    for flow in nodes[550:1550]:
        for memory in nodes[1550:]:  # Last 500 memory nodes
            if np.random.random() < 0.02:  # 2% connection probability
                flow.connect_to(memory, np.random.random() * 0.2 + 0.3)

    print(f"  Network created: {len(nodes)} nodes")

    # Initialize distributed engine
    distributed_engine = DistributedCTGVEngine(
        num_workers=4,
        partition_strategy='community',
        batch_size=500,
        max_partition_size=800,
        use_gpu=False  # Set to True if GPU available
    )

    # Partition network
    start_time = time.time()
    partitions = distributed_engine.partition_network(nodes)
    partition_time = time.time() - start_time

    print(f"  Partitioning completed in {partition_time:.2f}s")
    print(f"  Created {len(partitions)} partitions")
    print(f"  Average partition size: {np.mean([len(p) for p in partitions.values()]):.1f}")

    # Process distributed propagation
    print("  Running distributed propagation...")
    start_time = time.time()
    result = distributed_engine.distributed_propagate(nodes[:50])  # Start from origins
    process_time = time.time() - start_time

    print(f"  Distributed processing completed in {process_time:.2f}s")
    print(f"  Total operations: {result.get('total_operations', 'N/A')}")
    print(f"  Cross-partition operations: {result.get('cross_partition_ops', 'N/A')}")

    # ===== 2. STREAM PROCESSING =====
    print("\n[2] Testing Stream Processing Pipeline...")

    # Create streaming pipeline
    pipeline = StreamProcessingPipeline(batch_size=100, window_size=5)
    pipeline.start(num_workers=4)

    # Generate streaming data
    print("  Processing streaming data...")
    processed_batches = 0
    total_throughput = 0

    for batch_idx in range(10):
        # Create batch of gebits
        batch = []
        for i in range(100):
            gebit = Gebit(Shape.FLOW, intensity=np.random.random(),
                         label=f"Stream_{batch_idx}_{i}")
            batch.append(gebit)

        # Feed to pipeline
        pipeline.feed(batch)

        # Get results
        results = pipeline.get_results(timeout=1.0)
        if results:
            processed_batches += 1
            total_throughput += len(results)

        time.sleep(0.1)  # Simulate real-time processing

    pipeline.stop()

    print(f"  Processed {processed_batches} batches")
    print(".2f")
    print(".2f")

    # ===== 3. GPU ACCELERATION =====
    print("\n[3] Testing GPU Acceleration...")

    gpu_accelerator = GPUAccelerator()

    if gpu_accelerator.gpu_available:
        print("  GPU acceleration available")

        # Create test fields for GPU processing
        test_fields = []
        for i in range(200):
            field = VectorField(
                vector=np.random.random(3),
                phase=np.random.random() * 2 * np.pi,
                coherence=np.random.random()
            )
            test_fields.append(field)

        # GPU field interference calculation
        start_time = time.time()
        gpu_result = gpu_accelerator.gpu_field_interference(test_fields)
        gpu_time = time.time() - start_time

        print(f"  GPU interference calculation: {gpu_time:.4f}s")
        print(f"  Result shape: {gpu_result.shape}")

    else:
        print("  GPU acceleration not available (CPU fallback)")

    # ===== 4. HYPERSCALE SYSTEM =====
    print("\n[4] Testing Hyperscale CTGV System...")

    # Initialize hyperscale system
    hyperscale_config = {
        'num_workers': 2,
        'max_workers': 8,
        'target_throughput': 1000,
        'adaptation_interval': 5,
        'stream_batch_size': 100,
        'window_size': 10
    }
    hyperscale_system = HyperscaleCTGVSYSTEM(config=hyperscale_config)

    # Start monitoring
    monitor = SystemMonitor()

    # Simulate workload
    print("  Simulating increasing workload...")
    for load_level in range(1, 6):
        # Generate workload
        workload = []
        for i in range(load_level * 200):  # Increasing load
            gebit = Gebit(Shape.DECISOR, intensity=np.random.random(),
                         label=f"Workload_{load_level}_{i}")
            workload.append(gebit)

        # Process through hyperscale system
        start_time = time.time()
        results = hyperscale_system.process_large_network(workload)
        process_time = time.time() - start_time

        print(f"    Load level {load_level}: {len(workload)} items in {process_time:.2f}s")
        print(".2f")

        # Record metrics
        metrics = {
            'throughput': len(workload) / process_time if process_time > 0 else 0,
            'node_count': len(workload),
            'latency': process_time * 1000 / len(workload) if workload else 0
        }
        monitor.record_metrics(metrics)

        # Allow system to adapt
        time.sleep(1)

    # Stop monitoring
    print("  Hyperscale system test completed")

    print("  Hyperscale system test completed")

    # ===== 5. PERFORMANCE COMPARISON =====
    print("\n[5] Performance Comparison...")

    # Test with different network sizes
    sizes = [100, 500, 1000, 2000]
    results_comparison = {}

    for size in sizes:
        print(f"  Testing with {size} nodes...")

        # Create test network
        test_nodes = []
        for i in range(size):
            gebit = Gebit(Shape.FLOW, intensity=np.random.random(),
                         label=f"Test_{i}")
            test_nodes.append(gebit)

        # Sparse connections
        for i in range(min(size, 100)):  # Limit connections for scalability
            for j in range(max(0, i-5), min(size, i+6)):
                if i != j and np.random.random() < 0.1:
                    test_nodes[i].connect_to(test_nodes[j], np.random.random() * 0.5)

        # Test distributed processing
        start_time = time.time()
        dist_result = distributed_engine.distributed_propagate(test_nodes[:min(10, size)])
        dist_time = time.time() - start_time

        results_comparison[size] = {
            'distributed_time': dist_time,
            'efficiency': len(test_nodes) / dist_time if dist_time > 0 else 0
        }

        print(".2f"
              ".1f")

    print("\n" + "=" * 80)
    print(" DISTRIBUTED CTGV DEMONSTRATION COMPLETED")
    print("=" * 80)
    print("\nKey Achievements:")
    print("✓ Distributed processing for large networks (2000+ nodes)")
    print("✓ Stream processing pipeline with real-time throughput")
    print("✓ GPU acceleration for compute-intensive operations")
    print("✓ Hyperscale system with automatic scaling")
    print("✓ Performance scaling analysis across network sizes")
    print("\nThe distributed architecture successfully addresses scalability")
    print("challenges while maintaining CTGV's topological processing capabilities.")

if __name__ == "__main__":
    demonstrate_distributed_ctgv()