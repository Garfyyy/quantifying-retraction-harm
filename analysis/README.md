# Analysis

This directory contains the plotting scripts. The statistical methods for quantifying harm are defined in the "Statistics" section under "Materials and Methods" in the paper.

## Usage



1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Move data to this directory

```bash
mv ../data/* ./src/
```

3. Change directory to `src`

```bash
cd src
```

4. Run Plotting Scripts

```bash
# Overall analysis of harm for paper_c1
python fig2_final.py

# Overall analysis of harm for paper_c[2-6]
python fig3.py

# Overall analysis of harm for paper_c[2-6] (drop duplicates)
python fig3_s1.py

# Different JIF intervals analysis of harm for paper_c[1-6]
python fig4_v3.py

# Citation time analysis of harm for paper_c[1-6]
python fig_5.py
```
