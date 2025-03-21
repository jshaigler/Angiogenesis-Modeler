# Angiogenesis-Modeler

## Overview
This project simulates 3D angiogenesis using a web-based visualization. The simulation models the growth of arterial and venous trees, the formation of capillaries, and the distribution of cells with varying oxygen requirements.

## Features
- Interactive sliders to adjust simulation parameters
- Real-time 3D visualization using Plotly.js
- Simulation of arterial, venous, and capillary networks
- Visualization of different cell types and their oxygen levels

## Usage
1. Open `index.html` in a web browser.
2. Use the sliders on the right to adjust the simulation parameters:
   - Max Depth
   - Angle Range (°)
   - Branch Length
   - Capillary Threshold
   - Number of Cells
   - Oxygen Scale
   - Number of Arteries
   - Number of Veins
3. Click the "Recalculate" button to update the simulation with the new parameters.

## Simulation Parameters
- **Max Depth**: Maximum depth of the vascular tree.
- **Angle Range (°)**: Range of angles for branching.
- **Branch Length**: Initial length of each branch.
- **Capillary Threshold**: Distance threshold for forming capillaries.
- **Number of Cells**: Total number of cells in the simulation.
- **Oxygen Scale**: Scale factor for oxygen diffusion.
- **Number of Arteries**: Number of arterial trees.
- **Number of Veins**: Number of venous trees.

## Dependencies
- [Plotly.js](https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.24.2/plotly.min.js)
- [Numeric.js](https://cdnjs.cloudflare.com/ajax/libs/numeric/1.2.6/numeric.min.js)

## License
This project is licensed under the MIT License.