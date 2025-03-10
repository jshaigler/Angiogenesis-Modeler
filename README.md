# 3D Vascular Network Simulation

A Python-based computational model for simulating three-dimensional vascular networks with realistic tissue-like cell clustering and oxygen diffusion dynamics.



## Overview

This project provides a detailed simulation of blood vessel formation (angiogenesis) in 3D space, with emphasis on creating biologically plausible vascular architectures. The model includes:

- Recursive generation of arterial and venous trees
- Capillary formation between vessel terminals
- Tissue-like cell clustering with different cell types
- Oxygen diffusion modeling
- Dynamic capillary sprouting to meet cellular oxygen demands
- Interactive parameter tuning via sliders

## Requirements

```
numpy
scipy
plotly
ipywidgets
```

## Installation

```bash
pip install numpy scipy plotly ipywidgets
```

## Usage

Run the simulation in a Jupyter notebook or similar interactive Python environment:

```python
# Import the module
import vascular_simulation

# For a more direct approach, run the file directly
# Adjust the renderer as needed for your environment:
# - 'notebook' for Jupyter
# - 'colab' for Google Colab
# - 'browser' for standalone HTML
```

## Interactive Parameters

The simulation provides interactive sliders to control various aspects of vascular development:

| Parameter | Description | Range |
|-----------|-------------|-------|
| Max Depth | Controls branching generations | 2-8 |
| Angle Range | Controls vessel branching divergence (degrees) | 10-60 |
| Initial Branch Length | Sets scale of first generation branches | 0.05-0.2 |
| Capillary Threshold | Maximum distance for auto-connection | 0.05-0.5 |
| Num Cells | Number of cells in the simulation | 50-500 |
| Oxygen Scale | Controls oxygen diffusion rate | 0.01-0.2 |
| Num Arteries | Number of arterial entry points | 1-5 |
| Num Veins | Number of venous exit points | 1-5 |

## Key Functions

- `generate_tree_3d`: Recursively creates branching vessel structures
- `random_direction_in_cone`: Generates biologically plausible branching angles
- `generate_curve_points`: Creates smooth vessel paths using Bézier curves
- `compute_oxygen_levels`: Calculates oxygen concentration for each cell
- `generate_cells`: Creates tissue-like cell clusters of different types
- `simulate_vasculature_3d`: Main function that brings everything together

## Visualization

The simulation uses Plotly to create interactive 3D visualizations with:
- Red lines for arteries
- Blue lines for veins
- Green lines for capillaries
- Colored markers for different cell types (purple, orange, teal)

## Applications

This simulation has potential applications in:
- Tissue engineering and regenerative medicine
- Cancer research (tumor angiogenesis)
- Drug delivery optimization
- Vascular disease modeling
- Bioprinting guidance

## Example

```python
from vascular_simulation import simulate_vasculature_3d

# Run with default parameters
simulate_vasculature_3d(
    max_depth=5,
    angle_range=30,
    initial_branch_length=0.1,
    capillary_threshold=0.15,
    num_cells=200,
    oxygen_scale=0.05,
    num_arteries=3,
    num_veins=3
)
```

## Extending the Model

Potential extensions include:
- Adding flow dynamics
- Incorporating vessel remodeling
- Adding more realistic cell metabolism
- Simulating pathological conditions
- Integrating with experimental data

## License

None


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
