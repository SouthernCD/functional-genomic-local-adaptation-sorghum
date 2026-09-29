#!/usr/bin/env Rscript

# This backend only combines a caller-specified G and beta with random effects.
suppressPackageStartupMessages(library(PhenotypeSimulator))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 8L) {
    stop("Usage: backend.R genotype.tsv effects.tsv kinship.tsv output_prefix ",
         "causal_var kinship_var residual_var seed")
}

read_tsv <- function(path) {
    read.delim(path, sep = "\t", header = TRUE, check.names = FALSE,
               stringsAsFactors = FALSE)
}
scale_effect <- function(x, target, sample_ids) {
    if (target == 0) return(list(value = rep(0, length(sample_ids)), factor = 0))
    x <- matrix(as.numeric(x), ncol = 1L,
                dimnames = list(sample_ids, "Trait_1"))
    if (!all(is.finite(x)) || !is.finite(var(x[, 1L])) || var(x[, 1L]) <= 0) {
        stop("Cannot rescale a nonfinite or zero-variance component")
    }
    scaled <- PhenotypeSimulator::rescaleVariance(x, propvar = target)
    list(value = as.numeric(scaled$component[, 1L]),
         factor = as.numeric(scaled$scale_factor))
}
write_tsv <- function(x, path) {
    write.table(x, path, sep = "\t", quote = FALSE, row.names = FALSE)
}

geno <- read_tsv(args[1L])
effects <- read_tsv(args[2L])
kin <- read_tsv(args[3L])
prefix <- args[4L]
targets <- as.numeric(args[5:7])
seed <- as.integer(args[8L])
if (anyNA(targets) || any(targets < 0) || abs(sum(targets) - 1) > 1e-8 || is.na(seed)) {
    stop("Invalid variance targets or seed")
}
if (ncol(geno) < 2L || names(geno)[1L] != "sample_id" ||
    names(kin)[1L] != "sample_id" ||
    !all(c("SNP_ID", "beta_raw") %in% names(effects))) {
    stop("Invalid genotype, effect, or kinship headers")
}
ids <- as.character(geno$sample_id)
G <- as.matrix(geno[, -1L, drop = FALSE])
K <- as.matrix(kin[, -1L, drop = FALSE])
storage.mode(G) <- "double"
storage.mode(K) <- "double"
beta <- as.numeric(effects$beta_raw)
if (anyDuplicated(ids) || anyDuplicated(names(geno)[-1L]) ||
    !identical(ids, as.character(kin$sample_id)) ||
    !identical(ids, colnames(K)) ||
    !identical(colnames(G), as.character(effects$SNP_ID)) ||
    !identical(dim(K), c(length(ids), length(ids))) ||
    length(beta) != ncol(G) || anyNA(beta) ||
    !all(is.finite(G)) || !all(is.finite(K)) || !all(is.finite(beta))) {
    stop("Input alignment or numeric validation failed")
}
rownames(G) <- rownames(K) <- colnames(K) <- ids
n <- length(ids)

causal <- scale_effect(drop(G %*% beta), targets[1L], ids)
if (targets[2L] > 0) {
    set.seed(seed + 1001L)
    bg <- PhenotypeSimulator::geneticBgEffects(
        P = 1L, N = n, kinship = K, id_samples = ids,
        id_phenos = "Trait_1", shared = TRUE, independent = FALSE)
    kinship <- scale_effect(bg$shared, targets[2L], ids)
} else {
    kinship <- scale_effect(rep(0, n), 0, ids)
}
if (targets[3L] > 0) {
    set.seed(seed + 2001L)
    noise <- PhenotypeSimulator::noiseBgEffects(
        N = n, P = 1L, mean = 0, sd = 1, id_samples = ids,
        id_phenos = "Trait_1", shared = FALSE, independent = TRUE)
    residual <- scale_effect(noise$independent, targets[3L], ids)
} else {
    residual <- scale_effect(rep(0, n), 0, ids)
}
y <- causal$value + kinship$value + residual$value
components <- data.frame(sample_id = ids, causal_effect = causal$value,
                         kinship_effect = kinship$value,
                         residual = residual$value, phenotype = y,
                         check.names = FALSE)
write_tsv(components[, c("sample_id", "phenotype")],
          paste0(prefix, ".phenotype.tsv"))
write_tsv(components, paste0(prefix, ".components.tsv"))
write_tsv(data.frame(SNP_ID = colnames(G), beta_raw = beta,
                     beta_scaled = beta * causal$factor),
          paste0(prefix, ".causal.tsv"))
write.table(y, paste0(prefix, ".gemma.pheno"), quote = FALSE,
            row.names = FALSE, col.names = FALSE)
write_tsv(data.frame(
    key = c("PhenotypeSimulator_version", "R_version", "causal_scale_factor",
            "kinship_scale_factor", "residual_scale_factor"),
    value = c(as.character(packageVersion("PhenotypeSimulator")), R.version.string,
              causal$factor, kinship$factor, residual$factor)),
    paste0(prefix, ".backend.tsv"))
