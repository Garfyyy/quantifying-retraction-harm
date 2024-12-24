# Quantifying the Harm Caused by Retracted Research

A framework to quantify and analyze the harm of retracted papers through citation networks. Key findings:

- Retracted papers cause harm through "attention escape" mechanism

- Long-term impact is more significant than short-term effects

- Indirect citations suffer more harm than direct citations

- High-impact journals show more resilience to retraction harm
  
![Alt text](https://github.com/Garfyyy/quantifying-retraction-harm/blob/master/image.png)

## Contents

- [Data Prerequisites](#data-prerequisites)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)

##  Data Prerequisites

Our work involves publicly available data provided by the **Semantic Scholar Database**, **Retraction Watch Database** and **SciSciNet Database**. Therefore, users must first obtain the appropriate access and usage permissions from the relevant parties of the databases.

**1. Semantic Scholar API Key** [Link](https://www.semanticscholar.org/product/api)

First, navigating to the official Semantic Scholar website and completing the requisite application form to request an API key.

**2. Semantic Scholar Dataset Acquisition** [Link](https://api.semanticscholar.org/api-docs/datasets)

According to the official documentation, request and download the following datasets:

- `papers`: Contains metadata for all papers
- `citations`: Contains citation relationships between papers
  
**Notice:** This research utilizes the dataset version released on 2024-01-04 and the API Key obtained in step 1 is required for dataset requests.

**3. Retraction Watch Database** [Link](https://gitlab.com/crossref/eretraction-watch-data)

The Retraction Watch Database is a collection of retracted papers and their associated metadata. Download the dataset from the official repository.

**4. SciSciNet Database** [Link](https://springernature.figshare.com/collections/SciSciNet_A_large-scale_open_data_lake_for_the_science_of_science_research/6076908/1)

SciSciNet Databas offering comprehensive author, journal metadata and linkage information among these. Accessible via the provided link and download the following datasets:

- `SciSciNet_PaperDetails`: Contains metadata for all papers
- `SciSciNet_Journals`: Contains metadata for all journals
- `SciSciNet_Authors`: Contains metadata for all authors
- `SciSciNet_Affiliations`: Contains metadata for all affiliations

## Repo Contents

**quantification.** This directory contains the code for quantifying the harm caused by retracted research. [Code Usage](https://github.com/Garfyyy/quantifying-retraction-harm/tree/master/quantification)

**analysis.** This directory contains the analysis and visualization code for processing the research data in this paper. [Code Usage](https://github.com/Garfyyy/quantifying-retraction-harm/tree/master/analysis)

**data.** Final statistical results and processed datasets used in the paper. For methodology details, see "Statistics" in supplementary materials.

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
