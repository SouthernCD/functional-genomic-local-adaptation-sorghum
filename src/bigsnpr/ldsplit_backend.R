#!/usr/bin/env Rscript
# Backend for ldsplit.py; intentionally uses only bigsnpr and base R APIs.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 13L) stop("expected 13 backend arguments")

suppressPackageStartupMessages(library(bigsnpr))
# bigsnpr parallel workers and multi-threaded BLAS cannot be nested.
bigparallelr::set_blas_ncores(1L)
options(default.nproc.blas = 1L)
prefix <- args[[1]]
output <- args[[2]]
size_kb <- as.numeric(args[[3]])
thr_r2 <- as.numeric(args[[4]])
min_size <- as.integer(args[[5]])
max_size <- as.integer(args[[6]])
max_r2 <- as.numeric(args[[7]])
max_K <- as.integer(args[[8]])
ncores <- as.integer(args[[9]])
save_corr <- as.logical(args[[10]])
max_cost <- if (args[[11]] == "DEFAULT") NULL else as.numeric(args[[11]])
corr_source <- if (args[[12]] == "NONE") NULL else args[[12]]
# The final argument is reserved for a future format version; Python sends 1.
if (args[[13]] != "1") stop("unsupported backend format version")

expected <- read.delim(file.path(output, "expected.tsv"),
                       colClasses = c("integer", "character", "character",
                                      "integer", "numeric"), check.names = FALSE)
if (nrow(expected) == 0L) stop("no selected BIM variants")

if (is.null(corr_source)) {
  # Convert only selected columns; backing files remain in the output directory.
  rds <- snp_readBed2(paste0(prefix, ".bed"),
                      backingfile = file.path(output, "genotypes"),
                      ind.col = expected$Index, ncores = ncores)
  obj <- snp_attach(rds)
  map <- obj$map
  if (nrow(map) != nrow(expected) ||
      !identical(as.character(map$chromosome), expected$Chrom) ||
      !identical(as.character(map$marker.ID), expected$SNP) ||
      !identical(as.integer(map$physical.pos), expected$Position)) {
    stop("bigsnpr map order, IDs, chromosomes, or positions differ from PLINK BIM")
  }
} else {
  # Python has verified the source run's variant order against this BIM.
  map <- data.frame(chromosome = expected$Chrom,
                    marker.ID = expected$SNP,
                    physical.pos = expected$Position)
}
write.table(data.frame(Chrom = as.character(map$chromosome),
                       SNP = as.character(map$marker.ID),
                       Position = as.integer(map$physical.pos)),
            file = file.path(output, "variant_order.tsv"), sep = "\t",
            row.names = FALSE, quote = FALSE)

for (chrom in unique(expected$Chrom)) {
  indices <- which(expected$Chrom == chrom)
  directory <- file.path(output, paste0("chr_", chrom))
  dir.create(directory)
  positions <- expected$Position[indices]
  if (is.unsorted(positions, strictly = FALSE))
    stop("selected positions are not sorted on chromosome ", chrom)
  cat("Chromosome ", chrom, ": ", length(indices), " variants\n", sep = "")
  corr <- if (is.null(corr_source)) {
    snp_cor(obj$genotypes, ind.col = indices, size = size_kb,
            thr_r2 = thr_r2, infos.pos = positions, ncores = ncores)
  } else readRDS(corr_source)
  if (ncol(corr) != length(indices)) stop("cached correlation dimensions differ from BIM")
  if (save_corr && is.null(corr_source)) saveRDS(corr, file.path(directory, "corr.rds"))
  feasible_max <- min(max_size, length(indices))
  if (min_size > feasible_max)
    stop("min_size exceeds variant count on chromosome ", chrom)
  if (max_K < ceiling(length(indices) / feasible_max))
    stop("max_K is below the minimum number of blocks required by max_size on chromosome ", chrom)
  # Any pair with r² > max_r2 forbids every boundary between its variants.
  # Check whether that hard rule already forces a block larger than max_size.
  if (inherits(corr, "dsCMatrix") && corr@uplo == "U" && max_r2 < 1) {
    n <- ncol(corr)
    delta <- integer(n + 1L)
    for (j in seq_len(n)) {
      first <- corr@p[j] + 1L
      last <- corr@p[j + 1L]
      if (first > last) next
      strong <- corr@i[first:last][corr@x[first:last]^2 > max_r2]
      if (length(strong)) {
        start <- min(strong) + 1L
        if (start < j) {
          delta[start] <- delta[start] + 1L
          delta[j] <- delta[j] - 1L
        }
      }
    }
    cuts <- which(cumsum(delta)[seq_len(n - 1L)] == 0L)
    forced <- diff(c(0L, cuts, n))
    largest <- max(forced)
    write.table(data.frame(n_snp = n, allowed_cuts = length(cuts),
                           largest_forced_block = largest,
                           max_size = feasible_max, max_r2 = max_r2),
                file.path(directory, "feasibility.tsv"), sep = "\t",
                row.names = FALSE, quote = FALSE)
    if (largest > feasible_max)
      stop("max_r2=", max_r2, " forces at least ", largest,
           " SNPs into one block on chromosome ", chrom,
           ", exceeding max_size=", feasible_max,
           ". Increase max_r2 or max_size; see feasibility.tsv")
  }
  split_args <- list(corr = corr, thr_r2 = thr_r2, min_size = min_size,
                     max_size = feasible_max, max_K = max_K,
                     max_r2 = max_r2,
                     pos_scaled = expected$PosScaled[indices])
  if (!is.null(max_cost)) split_args$max_cost <- max_cost
  result <- do.call(snp_ldsplit, split_args)
  if (is.null(result) || nrow(result) == 0L)
    stop("no feasible LD split on chromosome ", chrom,
         "; inspect min_size, max_size, max_r2, max_K, max_cost, and pos_scaled")
  saveRDS(result, file.path(directory, "solutions.rds"))
  flat <- data.frame(max_size = result$max_size, n_block = result$n_block,
                     cost = result$cost, cost2 = result$cost2,
                     perc_kept = result$perc_kept,
                     all_last = vapply(result$all_last, paste, character(1),
                                       collapse = ","),
                     all_size = vapply(result$all_size, paste, character(1),
                                       collapse = ","))
  write.table(flat, file.path(directory, "solutions.tsv"), sep = "\t",
              row.names = FALSE, quote = FALSE)
  score <- result$cost2 * sqrt(5 + result$cost)
  if (!any(is.finite(score))) stop("no finite LD split score on chromosome ", chrom)
  selected <- which.min(replace(score, !is.finite(score), Inf))
  sizes <- as.integer(result$all_size[[selected]])
  if (length(sizes) == 0L || any(sizes <= 0L) || sum(sizes) != length(indices))
    stop("selected LD split does not cover chromosome ", chrom)
  write.table(sizes, file.path(directory, "selected_sizes.tsv"),
              row.names = FALSE, col.names = FALSE, quote = FALSE)
  write.table(flat[selected, , drop = FALSE],
              file.path(directory, "selected_solution.tsv"), sep = "\t",
              row.names = FALSE, quote = FALSE)
  cat("Selected ", length(sizes), " blocks; cost=", result$cost[selected],
      "; score=", score[selected], "\n", sep = "")
  rm(corr, result)
  gc(verbose = FALSE)
}
