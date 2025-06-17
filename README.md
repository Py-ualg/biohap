# BioHaP: Bioinformatics, Handy and Practical

Visit deployed tutorials [here](https://py-ualg.github.io/biohap/). Reach out and let's already start setting up *Fall 2025* edition.

## Spring 2025 edition

### 1️⃣ Simple Linux – An Introduction to Genomic Analysis on Linux

**Andrzej Tkacz [contact](mailto:atkacz@ualg.pt)**

- 🗓 Date: April 16, 15:00 - 17:00 (help with technical setup at 14:30)
- 📍 Location: CCMAR, Gambelas, building 7, room 1.39.

Large files or datasets—especially those containing genomic data—no problem. This workshop introduces essential Linux commands and simple Bash scripts to streamline data manipulation tasks. We’ll cover key operations such as searching for patterns, globally modifying content, and aligning DNA sequences, such as FASTAQ and FASTA files.

**Coverage**: bash, blast

**Requirements**: bash command line tool such, for windows WSL, or VBox are good. Alternatively, if comfortable, ask for account on HPC (contact David).

## 2️⃣ Microbiome Analysis of Amplicon Sequencing 

**Tânia Aires [contact](mailto:taires@ualg.pt), David Paleček [contact](mailto:dpalecek@ualg.pt)**

- 🗓 Date: May 7, 15:00 - 17:00 (help with technical setup at 14:30)
- 📍 Location: CCMAR, Gambelas, building 7, room 1.39.

This workshop will provide an overview of microbiome data analysis from Illumina amplicon sequencing raw data to data processing, visualization and statistics. Participants will learn basic concepts and tools to preprocess (QIIME2) and analyse (MicrobiomeAnalyst, Phyloseq) microbiome data, in particular bacteria associated with marine organisms.

**Coverage**: QIIME2, MicrobiomeAnalyst

**Requirements**: HPC account (contact David) for running QIIME2, R studio for Phyloseq.

## 3️⃣ RNA-Seq Data Analysis in R - From Counts to Biological Insights

**Isabel Duarte [contact](mailto:isabel.duarte@gmail.com)**

- 🗓 Date: June 4, 15:00 - 17:00 (help with technical setup at 14:30)
- 📍 Location: CCMAR, Gambelas, building 7, room 1.39.

Introduction to RNA-seq data analysis using R, focusing on differential expression analysis and learning how to choose the right Generalized linear model (GLM) for your question and your data. Participants will learn essential concepts and basic workflow, from preprocessing raw count data to identifying differentially expressed genes.

**Coverage**: BioConductor, DESeq2, tidyverse

**Requirements**: R studio

## Note

This project uses Quarto's `_freeze/` caching. The `_freeze/` folder is committed to the repo and marked as binary in `.gitattributes` to avoid merge conflicts. If in doubt, you can safely delete it and re-render with `quarto render biohap/`.
