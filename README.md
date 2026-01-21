# BioHaP: Bioinformatics, Handy and Practical

Visit [tutorials](https://py-ualg.github.io/biohap/) online. Reach out and let's get you tutorial scheduled and published.

## Spring 2026 edition

### 

## Fall 2025 edition

### Biodata.pt workshop - EMO-BON Metagenomics: From Backend Integration to Frontend Processing

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17805439.svg)](https://doi.org/10.5281/zenodo.17805439)

**David Paleček [contact](mailto:dpalecek@ualg.pt)**

Minimal examples demonstrating how to generate a knowledge graph from a collection of RO-Crates and query it via a SPARQL endpoint, either directly or using Python (`rdflib`). The examples are based on data from the EMO-BON long-term omics observatory for marine biodiversity.

- Tutorial on [Tess.elixir](https://tess.elixir-europe.org/materials/emo-bon-metagenomics-from-backend-integration-to-frontend-processing)
- 🗓 Date: November 17, 10:30 - 12:30
- 📍 Location: Faculty of Pharmacy of the University of Porto.

**Coverage**: fuseki server, SPARQ endpoint and queries, EMO-BON Data Analysis Kit

**Requirements**: Jupyter Notebook, basics of python

## Spring 2025 edition

### 1️⃣ Simple Linux - An Introduction to Genomic Analysis on Linux

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16760061.svg)](https://doi.org/10.5281/zenodo.16760061)

**Andrzej Tkacz [contact](mailto:atkacz@ualg.pt)**

- Tutorial on [Tess.elixir](https://tess.elixir-europe.org/materials/simple-linux-an-introduction-to-genomic-analysis-on-linux)
- 🗓 Date: April 16, 15:00 - 17:00 (help with technical setup at 14:30)
- 📍 Location: CCMAR, Gambelas, building 7, room 1.39.

Large files or datasets—especially those containing genomic data—no problem. This workshop introduces essential Linux commands and simple Bash scripts to streamline data manipulation tasks. We’ll cover key operations such as searching for patterns, globally modifying content, and aligning DNA sequences, such as FASTAQ and FASTA files.

**Coverage**: bash, blast

**Requirements**: bash command line tool such, for windows WSL, or VBox are good. Alternatively, if comfortable, ask for account on HPC (contact David).

## 2️⃣ Microbiome Analysis of Amplicon Sequencing

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16759291.svg)](https://doi.org/10.5281/zenodo.16759291)

**Tânia Aires [contact](mailto:taires@ualg.pt), David Paleček [contact](mailto:dpalecek@ualg.pt)**

- Tutorial on [Tess.elixir](https://tess.elixir-europe.org/materials/microbiome-analysis-of-amplicon-sequencing)
- 🗓 Date: May 7, 15:00 - 17:00 (help with technical setup at 14:30)
- 📍 Location: CCMAR, Gambelas, building 7, room 1.39.

This workshop will provide an overview of microbiome data analysis from Illumina amplicon sequencing raw data to data processing, visualization and statistics. Participants will learn basic concepts and tools to preprocess (QIIME2) and analyse (MicrobiomeAnalyst) microbiome data, in particular bacteria associated with cold-water corals.

**Coverage**: QIIME2, MicrobiomeAnalyst

**Requirements**: HPC account (contact David) for running QIIME2

## 3️⃣ RNA-Seq Data Analysis in R - From Counts to Biological Insights

**Isabel Duarte [contact](mailto:isabel.duarte@gmail.com)**

- Tutorial on [Tess.elixir](https://tess.elixir-europe.org/materials/rna-seq-from-counts-2-biological-insights)
- 🗓 Date: June 4, 15:00 - 17:00 (help with technical setup at 14:30)
- 📍 Location: CCMAR, Gambelas, building 7, room 1.39.

Introduction to RNA-seq data analysis using R, focusing on differential expression analysis and learning how to choose the right Generalized linear model (GLM) for your question and your data. Participants will learn essential concepts and basic workflow, from preprocessing raw count data to identifying differentially expressed genes.

**Coverage**: BioConductor, DESeq2, tidyverse

**Requirements**: R studio

## Note

This project uses Quarto's `_freeze/` caching. The `_freeze/` folder is committed to the repo and marked as binary in `.gitattributes` to avoid merge conflicts. If in doubt, you can safely delete it and re-render with `quarto render biohap/`.
