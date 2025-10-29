# Plotting

This directory contains the plotting scripts. The statistical methods for quantifying harm are defined in the "Statistics" section under "Materials and Methods" in the paper.

The same plotting scripts are used to generate results for both experimental group (using `data` or `median_data`) and control group (`d_data or` or `d_median_data`). 

## Usage

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Move data to this directory

```bash
mv ../data_experiment/* ./src/
```

3. Change directory to `src`

```bash
cd src
```

4. Run Plotting Scripts

```bash
# Overall plotting of harm for paper_c1
python fig2_final.py

# Overall plotting of harm for paper_c[2-6]
python fig3.py

# Overall plotting of harm for paper_c[2-6] (drop duplicates)
python fig3_s1.py

# Different JIF intervals plotting of harm for paper_c[1-6]
python fig4_v3.py

# Citation time plotting of harm for paper_c[1-6]
python fig_5.py
```
