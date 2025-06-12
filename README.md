# Characterizing Linguistic Shifts in Croatian News via Diachronic Word Embeddings

Code for the paper accepted at the 10th Workshop on Slavic Natural Language Processing 2025 (SlavicNLP 2025).

## TLDR
We conducted an analysis of semantic change in Croatian language over 25 years using word embeddings trained on 9.5 million news articles.

## Overview
We investigate how word meanings evolve by training skip-gram embeddings on Croatian news articles from five-year periods (2000-2024). Our analysis captures linguistic shifts related to major events like COVID-19, EU accession, and technological changes. We also find evidence that embeddings from post-2020 encode increased positivity in sentiment analysis tasks.

## Embeddings
We release the trained embeddings from **five-year periods** in this repository and also model trained on **whole 25-year periods**. 

To obtain the embeddings:

1. Navigate to the `5ysplits_models` folder
2. Run the data retrieval script:
   ```bash
   ./get_data.sh
   ```
This will download and set up all the trained embedding models.

