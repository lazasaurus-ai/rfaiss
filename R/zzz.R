# R/zzz.R
.onLoad <- function(libname, pkgname) {
  Sys.setenv(OPENBLAS_NUM_THREADS = Sys.getenv("OPENBLAS_NUM_THREADS", "1"))
  Sys.setenv(OMP_NUM_THREADS = Sys.getenv("OMP_NUM_THREADS", "1"))
  Sys.setenv(TOKENIZERS_PARALLELISM = Sys.getenv("TOKENIZERS_PARALLELISM", "false"))
  invisible()
}

#' Set OpenBLAS and OMP thread counts
#' @export
set_openblas_threads <- function(threads = 1L) {
  Sys.setenv(OPENBLAS_NUM_THREADS = as.character(threads))
  Sys.setenv(OMP_NUM_THREADS = as.character(threads))
  invisible(TRUE)
}
