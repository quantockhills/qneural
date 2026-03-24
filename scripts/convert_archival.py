#!/usr/bin/env python3
"""
Convert all archival models to new qneural format.

This script converts the 5 publication-quality model files from
the archival format to the new qneural checkpoint format.

Usage:
    python scripts/convert_archival.py
"""

import sys
from pathlib import Path

# Add qneural to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qneural.utils import convert_archival_to_new_format


def main():
    """Convert all archival models."""
    
    # Define conversions
    conversions = [
        {
            'old_path': 'archival/data/final_models/5e-5pi_to_0.05pi',
            'new_path': 'qneural/data/publication_models/5e-5pi_to_0.05pi.pt',
            'metadata': {
                'source': 'archival_publication',
                'original_file': '5e-5pi_to_0.05pi',
                'note': 'Publication-quality results for small angles (5e-5π to 0.05π)',
                'missing_data': ['training_history', 'epoch_count', 'optimizer_states']
            }
        },
        {
            'old_path': 'archival/data/final_models/0.05pi_to_0.1pi',
            'new_path': 'qneural/data/publication_models/0.05pi_to_0.1pi.pt',
            'metadata': {
                'source': 'archival_publication',
                'original_file': '0.05pi_to_0.1pi',
                'note': 'Publication-quality results for small angles (0.05π to 0.1π)',
                'missing_data': ['training_history', 'epoch_count', 'optimizer_states']
            }
        },
        {
            'old_path': 'archival/data/final_models/pt1pi_to_pt3pi',
            'new_path': 'qneural/data/publication_models/pt1pi_to_pt3pi.pt',
            'metadata': {
                'source': 'archival_publication',
                'original_file': 'pt1pi_to_pt3pi',
                'note': 'Publication-quality results for mid-range angles (0.1π to 0.3π)',
                'missing_data': ['training_history', 'epoch_count', 'optimizer_states']
            }
        },
        {
            'old_path': 'archival/data/final_models/pt3pi_to_pt5pi',
            'new_path': 'qneural/data/publication_models/pt3pi_to_pt5pi.pt',
            'metadata': {
                'source': 'archival_publication',
                'original_file': 'pt3pi_to_pt5pi',
                'note': 'Publication-quality results for mid-range angles (0.3π to 0.5π)',
                'missing_data': ['training_history', 'epoch_count', 'optimizer_states']
            }
        },
        {
            'old_path': 'archival/data/final_models/pt5pi_to_pi',
            'new_path': 'qneural/data/publication_models/pt5pi_to_pi.pt',
            'metadata': {
                'source': 'archival_publication',
                'original_file': 'pt5pi_to_pi',
                'note': 'Publication-quality results for CZ gate (0.5π = π for CZ)',
                'missing_data': ['training_history', 'epoch_count', 'optimizer_states']
            }
        }
    ]
    
    print("Converting archival models to new format...")
    print("=" * 60)
    
    for i, conv in enumerate(conversions, 1):
        print(f"\n[{i}/5] Converting {conv['old_path']}...")
        
        result = convert_archival_to_new_format(
            conv['old_path'],
            conv['new_path'],
            conv['metadata']
        )
        
        if result['success']:
            print(f"  ✓ Success!")
            print(f"  Architecture: {result['architecture_detected']['time_hidden_layers']}x{result['architecture_detected']['time_hidden_units']} (time), "
                  f"{result['architecture_detected']['control_hidden_layers']}x{result['architecture_detected']['control_hidden_units']} (control)")
            print(f"  Angle range: [{result['angle_range'][0]/3.14159:.4f}π, {result['angle_range'][1]/3.14159:.4f}π]")
            print(f"  Saved to: {result['output_path']}")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("\nConverted files are in: qneural/data/publication_models/")
    print("\nThese models can be loaded with:")
    print("  checkpoint = torch.load('path/to/model.pt')")
    print("  controller = TimeOptimalController(...)")
    print("  controller.time_predictor.load_state_dict(checkpoint['time_network_state_dict'])")
    print("  controller.control_generator.load_state_dict(checkpoint['control_network_state_dict'])")


if __name__ == '__main__':
    main()
