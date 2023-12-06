# Intro

[eQTLs](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3682727/) are regions within the genome where genetic variation (such as single nucleotide polymorphisms or SNPs) is associated with the expression levels of specific genes. eQTLs are detected through a process that involves correlating genetic variation (usually SNPs) with gene expression levels. 

In this project, you will apply a masked language model (MLM) to predict gene expression levels and identify associated eQTLs. 

Your analysis will be based on data from Phase 3 of 1000 Genomes project. [The 1000 Genomes Project](https://www.internationalgenome.org/1000-genomes-summary/) created a catalogue of common human genetic variation, using openly consented samples from people who declared themselves to be healthy. 

# Workflow

 Measured expression ([Lappalainen et al.,2013](https://www.nature.com/articles/nature12531)) data is [available](https://www.ebi.ac.uk/biostudies/files/E-GEUV-1/E-GEUV-1/analysis_results/GD660.GeneQuantRPKM.txt.gz) for 1042 genes of 503 samples from 1000 Genomes.
You will try to predict these values using sequence embeddings from the MLM.

For this project, a [modified version](model/) of the MLM from [Gankin et al., 2023](https://www.biorxiv.org/content/10.1101/2023.01.26.525670v1) will be used. In contrast to the original MLM trained to reconstruct DNA sequences, the modified version is trained to reconstruct individual's haplotypes within each analysed gene (AI-based reconstruction of haplotypes was also done previously using denoising autoencoders ([Chen&Shi,2019](https://pubmed.ncbi.nlm.nih.gov/31466333/)) and split transformers [Mowlaei et al.,2023](https://www.biorxiv.org/content/10.1101/2023.03.05.531190v1)). 

Each haplotype is represented by a chunk of at most 5000 commonly mutated positions (Minor allele frequency (MAF) > 3%) around the transcription start site (TSS) of each gene. Note that given the mutation density of 1 SNP per 400 DNA basepairs (1000 Genomes Phase 3, 2054 samples, MAF > 3%), each chunk spans on average a region of about 2Mbp around the TSS. Each chunk strictly corresponds to a single gene. During the training, random masking will be performed at each epoch by stratifying haplotype positions according to the corresponding MAF.

To measure the model performance, we use two metrics: accuracy at the masked positions and the imputation quality score ([Lin et al.,2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009697)).

Out of 2054 individuals available in 1000 Genomes Phase 3, 503 samples for which gene expression is reported were designated as the test set and the remaining 1551 samples were used for training and validation. 

The resulting dataset consists of 2 files: 
* `dataset.parquet` is a table of genotypes for all chunks. Each position is encoded as 'R','F','M','B', meaning the reference allele, 'farther' variant, 'mother' variant, and both alleles mutated (homozygous variant). Remember that since 1000 Genomes genotypes are phased, each genotype can be unambigously split into two haplotypes. This splitting occurs at random in the SeqDataset, where 'R','F','M', and 'B' are converted to 'R' and 'A', which stands for the 'reference' and 'alternative' alleles. The chunk label format is 'split:sample_id:gene' where 'split' is 'train' or 'test'.
* `meta.csv.gz` is a list of all positions considered for each gene (same for all individuals). For a given gene, the SNP position in meta.csv.gz matches exactly the position of this SNP in the genotype string in `dataset.parquet`.

See [dataprep/dataprep.ipynb](dataprep/dataprep.ipynb) to get a more comprehensive understanding on how the data is preprocessed.

Two datasets have been prepared: based on top10 mutated genes (validation masked accuracy 99.2%, IQS=98%) and based on top100 mutated genes. These datasets alongside with the pre-trained model weights will be provided to you with a separate link.

# Research questions

1. The current model is trained to reconstruct haplotypes. Can you modify the model code s.t. it reconstructs genotypes? (Hint: modify SeqDataset s.t. it consecutively yields two haplotypes from the same person and design a custom collate_fn for data loaders. Don't forget to modify the metrics correspondingly.You don't have to retrain the model.)
2. Generate embeddings on the test set and use them as an input of an auxiliary linear model to predict measured gene expression. You may try Linear/Ridge/Lasso regression or SVR. Find a way to combine embeddings from two haplotypes of the same individual.
3. Try to predict gene expression using fine-tuning, i.e. by freezing some model layers and adding an extra prediction head.
4. Compare your results with expression predictions based on single SNPs (`rvalue` in [this table](https://www.ebi.ac.uk/arrayexpress/files/E-GEUV-1/EUR373.gene.cis.FDR5.all.rs137.txt.gz)).
5. Let's make a step towards explainable AI. Figure out how to identify SNPs relevant for expression prediction. Can you investigate biological reasons for this? (Hint: use [different ENCODE tracks for GM12878](https://www.encodeproject.org/search/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=Histone+ChIP-seq&assay_title=Mint-ChIP-seq&status=released&biosample_ontology.term_name=GM12878))

# Running the model

1. Create new [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment:

```
conda create -n lm-eqtl python=3.10
conda activate lm-eqtl
```
2. Install [Pytorch v.1.13.1](https://pytorch.org/get-started/previous-versions/)

3. Install the other requirements using pip:

```
pip install -r requirements.txt
```

If you're working on GoogleColab, you can skip steps 1 and 2.

The following command can be used to get embeddings:

```
python model/main.py --test --get_embeddings --dataset datasets/phase3_top10/dataset.parquet --output_dir ./test --model_weight checkpoints/phase3_top10/aware_large_splitmsk/weights/epoch_100_weights_model.pt
```

Embeddings will then be saved as a `.pickle` object under the `./test` directory.

With the `--mask_at_test` option embeddings will be generated faster, but it is not certain if this would improve or degrade your results.

When you need to fine-tune the model, drop the `--test` and `--get_embeddings` options and add `--Nfolds 5 --fold 0`. This way, the teain data will be automatically split into 5 folds and the 0th fold will be used for validation. 

The hyperparameters are well-tuned, so you normally don't need to modify them. If you run out of memory, try to reduce `--batch_size`. You may also want to change the `--learning_rate` for fine-tuning.

Running inference and fine-tuning on the top100 dataset will take some time, so it makes sense to do everything first based on the top10 dataset, then switch to top100.

Note that the running time improves significantly when running on GPU.




