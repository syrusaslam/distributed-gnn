"""
Script to download and prepare datasets for training.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.loader import load_cora_dataset, save_graph, get_graph_statistics


def download_cora():
    """Download and save Cora dataset"""
    print("=" * 60)
    print("Downloading Cora Dataset")
    print("=" * 60)

    try:
        graph, features, labels, train_mask, val_mask, test_mask = load_cora_dataset()

        # Print statistics
        stats = get_graph_statistics(graph)
        print("\nCora Dataset Statistics:")
        print(f"  Nodes: {stats['num_nodes']}")
        print(f"  Edges: {stats['num_edges']}")
        print(f"  Avg Degree: {stats['avg_degree']:.2f}")
        print(f"  Max Degree: {stats['max_degree']}")
        print(f"  Density: {stats['density']:.6f}")

        # Print split info
        num_train = sum(train_mask.values())
        num_val = sum(val_mask.values())
        num_test = sum(test_mask.values())

        print(f"\nDataset Splits:")
        print(f"  Train: {num_train} nodes ({num_train/stats['num_nodes']*100:.1f}%)")
        print(f"  Val:   {num_val} nodes ({num_val/stats['num_nodes']*100:.1f}%)")
        print(f"  Test:  {num_test} nodes ({num_test/stats['num_nodes']*100:.1f}%)")

        # Feature info
        feature_dim = list(features.values())[0].shape[0]
        num_classes = len(set(labels.values()))
        print(f"\nFeature Info:")
        print(f"  Feature Dimension: {feature_dim}")
        print(f"  Number of Classes: {num_classes}")

        print("\n✓ Cora dataset downloaded successfully!")
        print(f"  Data saved to: data/cora/")

        return True

    except Exception as e:
        print(f"\n✗ Failed to download Cora dataset: {e}")
        print("\nMake sure torch_geometric is installed:")
        print("  pip install torch-geometric")
        return False


def main():
    """Main download script"""
    print("Distributed GNN Dataset Downloader")
    print()

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Download datasets
    success = download_cora()

    if success:
        print("\n" + "=" * 60)
        print("All datasets downloaded successfully!")
        print("=" * 60)
        print("\nYou can now run:")
        print("  python examples/single_machine_train.py")
    else:
        print("\n" + "=" * 60)
        print("Some datasets failed to download")
        print("=" * 60)
        sys.exit(1)


if __name__ == '__main__':
    main()
