# Read files and chunk text
read_text_safe <- function(path) {
  ext <- tolower(tools::file_ext(path))
  if (ext %in% c("txt","md","r","py","csv","tsv","json","yaml","yml","qmd","rmd")) {
    return(paste(readLines(path, warn = FALSE, encoding = "UTF-8"), collapse = "\n"))
  }
  if (ext == "pdf" && requireNamespace("pdftools", quietly = TRUE)) {
    return(paste(pdftools::pdf_text(path), collapse = "\n"))
  }
  # Fallback: try reading as text
  tryCatch(paste(readLines(path, warn = FALSE, encoding = "UTF-8"), collapse = "\n"),
           error = function(e) "")
}

# Simple char-based chunker with overlap
chunk_text <- function(txt, chunk_chars = 1200L, overlap = 200L) {
  if (nchar(txt, type = "chars") == 0) return(character(0))
  n <- nchar(txt, type = "chars")
  starts <- seq(1L, n, by = (chunk_chars - overlap))
  vapply(starts, function(s) {
    e <- min(n, s + chunk_chars - 1L)
    substr(txt, s, e)
  }, character(1))
}

list_files <- function(dir, pattern = NULL, recursive = TRUE) {
  files <- list.files(dir, pattern = pattern, recursive = recursive, full.names = TRUE)
  # filter out directories and hidden files
  files[file.info(files)$isdir == FALSE]
}
