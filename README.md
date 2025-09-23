# UTR Sequence Machine Learning Optimization Platform

## Project Introduction

This project systematically optimizes the 5' and 3' untranslated regions (UTRs) of mRNAs through machine learning and high-throughput experimental screening, aiming to improve the translation efficiency and stability of mRNA therapeutics. The project integrates a complete workflow of sequence design, predictive modeling, experimental validation, and in-depth analysis.

## Main Functions

### 1. Sequence Design and Optimization
- Generate optimized UTR sequences using genetic algorithms and CNN/RNN models
- Multi-objective optimization strategy (ribosome loading MRL + translation efficiency TE)
- Combined optimization of 5' and 3' UTRs

### 2. Prediction Models
- **HEK Model**: Predict translation efficiency
- **Optimus Model**: Predict ribosome loading
- **K-fold Cross-Validation**: Ensure model generalization

### 3. Experimental Validation
- **In Vitro Validation**: Luciferase, hEPO, and RSV protein expression
- **Cell Lines**: C2C12 (muscle cells), HepG2 (hepatocytes)
- **In Vivo Validation**: Rat pharmacokinetic experiments

### 4. Sequence Analysis
- **k-mer enrichment analysis**: Identify functional sequence patterns between 4 and 10 bp
- **Secondary structure prediction**: ViennaRNA calculates MFE and stem-loop structure
- **Stability assessment**: Coefficient of variation analysis of expression stability
- **Condition dependency**: Effects of time, dose, and cell line

## Instructions

### 1. Sequence Design
\`\`\`python
# Generate new UTR sequences using a genetic algorithm
python 03_sequence_design/stepwise_evolve_utrs_combo_prefix.py

# Extract sequence features
python 03_sequence_design/UTR_Features.py
\`\`\`

### 2. Sequence Prediction
\`\`\`python
# Predict sequence efficiency using the trained model
python 02_Prediction/scripts/Test_model.py
\`\`\`

### 3. Data Analysis
\`\`\`bash
# Run analysis notebooks
jupyter notebook 04_analysis/

# Batch generate RNA secondary structures
./04_analysis/RNA_secondary_structure_batch_visualization.sh input.fasta output_dir
\`\`\`

### 4. View Results
- K-mer Analysis Results: \`05_results/RESULT_utr5_*/\`
- Expression Analysis Charts: \`05_results/expression_*.png\`
- Secondary Structure Data: \`05_results/RESULT_SS/\`
- Stability Analysis: \`05_results/vMay16_Stability.png\`
## Data File Description

### Raw Data (\`01_data/raw/\`)
- \`Cao_HEK.csv\`: Endogenous gene expression data in HEK cells
- \`GSM3130435_egfp_unmod_1.csv.zip\`: EGFP reporter gene dataset (40,000+ sequences)
- \`RVAC Preclinical RD_mRNA supply_m001-m100 sequence.xlsx\`: First batch of designed sequences
- \`20220516_100UTR_*.xlsx\`: Second batch of designed sequences
- \`RSV_ELISA_P1-P4.xlsx\`: RSV protein ELISA data

### Screening Data (\`01_data/screening/\`)
- **First round of screening**: Luciferase expression of 475 UTRs in two cell lines
- **Second Round Screening**: In-Depth Validation of 81 UTR Combinations (9×9 Pairs)
- **hEPO Screening**: Validation of Human Erythropoietin Expression of the Optimal UTRs

### Sequence Files (\`UTR_sequences/\`)
- \`round_1_screening_475_UTR_sequences.xlsx\`: Initial Screening Sequences
- \`round_2_nine_5UTR_combine_nine_3UTR+10_paired.xlsx\`: Optimized Combinations
- \`vMay14_luciferase_screening_summary_log2.csv\`: Screening Summary Data

## Analysis Script Introduction

| Script | Function |
|------|------|
| \`vMay11_utr_kmer_enrichment_analysis.ipynb\` | K-mer Pattern Recognition and Enrichment Analysis |
| vMay11_utr_secondary_structure_comparison_475vs195.ipynb | RNA Secondary Structure Comparison |
| vMay14_parade_model_data_processor.py.ipynb | Machine Learning Data Preprocessing |
| vMay16_utr_expression_Difference_across_Time_Dosage_CellLine.ipynb | Multidimensional Expression Analysis |
| vJun23_design_vs_background_distribution.ipynb | Validation of Design Sequence Effects |

## Key Findings

- **Expression Improvement**: Optimizing UTRs Achieves 2-8-fold Expression Increases
- **Enrichment Pattern**: Identification of Functional Motifs Such as CUUC, UUGC, and GAGAA
- **Structural Features**: Highly Efficient UTRs Prefer Low GC Content and Simple Secondary Structures
- **Stability**: Sequence Variation Coefficient <15% for More Than 40%
- **Combination Effect**: Synergistic and antagonistic interactions between the 5' and 3' UTRs exist.

## Context Dependency

See `requirements.txt` file.