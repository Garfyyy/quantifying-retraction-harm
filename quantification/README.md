# Quantification of Harm

This directory contains the core implementation for calculating the harm of papers. For detailed formulas and methods, see "Quantification of Harm" in supplementary materials.

## Code Structure

- `quantize_v3.py`: Main script for calculating the harm  of individual papers.
- `mult_v3.sh`: Shell script for parallel processing across multiple machines and cores

## Configuration Parameters

The `mult_v3.sh` script contains several key configuration parameters:

```sh
n_cite          # Citation layer level (0 for first-layer citations, etc.)
num_chunks      # Total number of chunks to split the dataset
st              # Starting chunk index
ed              # Ending chunk index
```

## Usage

```sh
# Install dependencies
pip install -r requirements.txt

# Configure the script parameters
vim mult_v3.sh

# Run the parallel processing script
bash ./mult_v3.sh
```

**Note:**

- The script uses trap to handle termination signals and clean up child processes
- Each chunk starts with a 10-second delay to prevent resource overload
- Ensure adequate system resources for parallel processing
- Processing time and memory usage increase with higher `n_cite` values as more papers need to be computed.

