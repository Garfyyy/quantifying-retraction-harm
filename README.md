# Quantifying the Dynamics of Harm Caused by Retracted Research

A framework to quantify and analyze the harm of retracted papers through citation networks. Key findings:

- Retracted papers cause harm through "attention escape" mechanism

- Indirect citations suffer more harm than direct citations

- Lower JIF journals experience greater harm from retractions
  
## Contents

- [Quantifying the Dynamics of Harm Caused by Retracted Research](#quantifying-the-dynamics-of-harm-caused-by-retracted-research)
  - [Contents](#contents)
  - [System Requirements](#system-requirements)
  - [Data Prerequisites](#data-prerequisites)
  - [Repo Contents](#repo-contents)
  - [Installation Guide](#installation-guide)
  - [Demo](#demo)
    - [1. Quantification of Harm](#1-quantification-of-harm)
    - [2. Calculate Quartile Statistics for Harm](#2-calculate-quartile-statistics-for-harm)
    - [3. Analysis](#3-analysis)
  
## System Requirements

**Hardware Requirements**

The package processes a large dataset containing millions of research papers and citation recods. The package has been tested on the following systems:

**(1) Quantification Environment**

- **RAM:** 251 GB
- **CPU:** Intel(R) Xeon(R) Silver 4208 CPU @ 2.10GHz 32 cores
- **Operating System:** CentOS Linux release 7.9.2009

**（2）Analysis Environment**

      ① Linux Environment
      - **RAM:** 251 GB
      - **CPU:** Intel(R) Xeon(R) Silver 4208 CPU @ 2.10GHz 32 cores
      - **Operating System:** CentOS Linux release 7.9.2009

      AND
      
      ② Windows Environment
      - **RAM:** 16.0 GB
      - **CPU:** 12th Gen Intel(R) Core(TM) i7-12700 @ 2.10GHz 12 cores
      - **Operating System:** Windows 11 Home Chinese Version 22H2
  
**Software Requirements**

- **Python:** 3.10
- **IDE:**
  - PyCharm 2020
  - Visual Studio Code 1.85.2


##  Data Prerequisites

Our work involves publicly available data provided by the **Semantic Scholar Database**, **Retraction Watch Database** and **SciSciNet Database**. Therefore, users must first obtain the appropriate access and usage permissions from the relevant parties of the databases.

**1. Semantic Scholar API Key** [Link](https://www.semanticscholar.org/product/api)

First, navigating to the official Semantic Scholar website and completing the requisite application form to request an API key.

**2. Semantic Scholar Dataset Acquisition** [Link](https://api.semanticscholar.org/api-docs/datasets)

According to the official documentation, request and download the following datasets:

- `papers`: Contains metadata for all papers
- `citations`: Contains citation relationships between papers
  
**Notice:** This research utilizes the dataset version released on 2024-01-04 and the API Key obtained in step 1 is required for dataset requests.

**3. Retraction Watch Database** [Link](https://gitlab.com/crossref/retraction-watch-data)

The Retraction Watch Database is a collection of retracted papers and their associated metadata. Download the dataset from the official repository.

**4. SciSciNet Database** [Link](https://springernature.figshare.com/collections/SciSciNet_A_large-scale_open_data_lake_for_the_science_of_science_research/6076908/1)

SciSciNet Databas offering comprehensive author, journal metadata and linkage information among these. Accessible via the provided link and download the following datasets:

- `SciSciNet_PaperDetails`: Contains metadata for all papers
- `SciSciNet_Journals`: Contains metadata for all journals
- `SciSciNet_Authors`: Contains metadata for all authors
- `SciSciNet_Affiliations`: Contains metadata for all affiliations

## Repo Contents

**quantification.** This directory contains the code for quantifying the harm caused by retracted research.

**analysis.** This directory contains the analysis and visualization code for processing the research data in this paper.

**data.** Final statistical results and processed datasets used in the paper. For methodology details, see "Statistics" in supplementary materials.

## Installation Guide

Change the directory, install the dependencies for each part, and refer to the README of each part for the process:

1. **Quantification of Harm**

```bash
cd quantification

pip install -r requirements.txt
```

The installation process takes about 10 seconds.

This section refers to the README, which can be accessed via the link. [Quantification README](https://github.com/Garfyyy/quantifying-retraction-harm/tree/master/quantification)

2. **Analysis**

```bash
cd analysis

pip install -r requirements.txt
```

The installation process takes about 10 seconds.

This section refers to the README, which can be accessed via the link. [Analysis README](https://github.com/Garfyyy/quantifying-retraction-harm/tree/master/analysis)

## Demo

### 1. Quantification of Harm

```bash
cd analysis
bash mult_v3.sh

# Output: Quantized results (e.g., ./res/paper_c1/quantize_0_0.parquet),  
# containing both experimental group and control group results.
```
This gives us the harm received by 100 demo papers within 10 years of publication.

### 2. Calculate Quartile Statistics for Harm

```bash
python calc_harm.py

# Output the 'Medicine' field quartile statistics
# q1: [-0.72870662 -0.59714599 -0.35194113 -0.57356077 -0.70212766  -0.62033037 -0.5308642  -0.62432411 -0.59534771 -0.57835616]
# q2: [-0.06854839  0.07968844  0.1659919   0.21257367  0.20075312  0.26370023 0.28019454  0.47482014  0.44240077  0.50371471]
# q3: [0.46927966   0.57701758   0.6498847  0.53979239  0.64233888  0.69115027 0.85652007 1. 0.78139134 1.]
```

In this demo, we obtain the first quartile (q1), median (q2), and third quartile (q3) values of harm for the 'Medicine' field over a 10-year period.
The datasets used for quantification and quartile calculations are very large. The calculation results in the demo are not accurate and are for reference only. 

### 3. Analysis

The complete calculation results are available in the **data** folder. Here use the complete results for analysis.

(1) Move data to `analysis/src/`.

```bash
mv data/* analysis/src/
```

(2) Changes the current directory to `analysis/src/`.

```bash
cd analysis/src/
```

(3) Runs the Python script.

```bash
# The definition of paper_cn is detailed in the "Citation Distance" section under 
# "Materials and Methods" in the paper.

# The statistical methods for quantifying harm are defined in the "Statistics" 
# section under "Materials and Methods" in the paper.

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

**Note on Runtime:**

- Each figure generation takes approximately ***2*** seconds
- Total runtime for all analyses may vary depending on the data size
- Runtime measurements are based on Windows 11 with i7-12700 CPU and 16GB RAM

**Demo fig2a output example**

![fig2a_example](https://github.com/Garfyyy/quantifying-retraction-harm/blob/master/fig2a_example.png)
