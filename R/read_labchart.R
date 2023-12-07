require(data.table)
require(stringr)
require(lubridate)
require(dplyr)

# (latest version on Github at https://github.com/tytell/tytell-lab.git)
# This file from commit $Id$

#' Parse a text file exported from ADI LabChart
#' 
#' `read_labchart` parses a data from from LabChart, potentially with multiple blocks, and returns a tibble.
#' 
#' This function parses a text file exported from ADI LabChart. It processes the tags at the beginning of the file,
#' which include the names and units for the channels, the sampling rate, and the start date for the file.
#' Files with multiple blocks are parsed correctly.
#' 
#' The column names in the output are the channel titles from LabChart (with spaces replaced by underscores) and
#' the units attached with a dot (e.g., a channel called "X Force" with units of mN in LabChart would be parsed as
#' a column name `X_Force.mN`).
#' 
#' Several columns are added:
#' * `t.sec`: Time in seconds, based on the sampling rate
#' * `Comment`: Text of comments in the file.
#' * `Start_Time`: The time recorded when the block actually started
#' * `Block_Number`: Number of the block in the file, sequentially from the beginning.
#' 
#' @param filename Name of the file to import
#' @param include_units Include the units in the column names (default = TRUE)
#' 
#' *Note*: Probably will not correctly handle files where channels or blocks have different sampling rates.
read_labchart <- function(filename, 
                          include_units = TRUE,
                          .progress = TRUE)
{
  fio <- file(description = filename,
              open = "rb",
              blocking = TRUE)
  on.exit(close(fio), add = TRUE)
  
  filesize <- file.size(filename)
  readsize <- 0
  
  done <- FALSE
  header <- list()
  while (!done) {
    ln <- readLines(fio, n = 1)
    readsize <- readsize + length(ln) + 1
    
    hd1 <- parse_header_line(ln)
    done <- is.null(hd1)
    
    if (!done) {
      header[[length(header)+1]] <- hd1
    }
  }
  
  header <- dplyr::bind_rows(header) |>   
    tibble::column_to_rownames("name")
  
  col_names <- header["ChannelTitle", "value"][[1]]
  
  if (include_units & "UnitName" %in% row.names(header)) {
    col_units <- header["UnitName", "value"][[1]]
    col_names <- purrr::map2(col_names, col_units,
                            ~ case_when(
                              .y == "*"  ~  .x,
                              .default = str_c(.x, '.', .y)))
  }

  col_names <- col_names |>
    str_replace("\\s+", "_")
  
  dt <- header["Interval", "value"][[1]] |> 
                 stringr::str_extract("[+-]*[\\d.]+") |> 
                 as.numeric()
  
  start_time <- header["ExcelDateTime", "value"][[1]][[2]] |> 
    lubridate::parse_date_time(orders = "m/d/Y H:M:S", tz = Sys.timezone())
  
  data <- strsplit(ln, "\t", fixed = TRUE)
  data <- data[[1]]
  
  if (length(data) == length(col_names) + 1) {
    nstartcol <- 1
    col_names <- c("t.sec", col_names)
  } else if (length(data) == length(col_names) + 2) {
    nstartcol <- 2
    col_names <- c("t.sec", "Date", col_names)
  } else {
    stop(stringr::str_glue("Unrecognized number of columns (expected {length(col_names)+1} or {length(col_names)+2}, but found {length(data)})"))
  }
  numcols <- length(col_names)
  
  data <- parse_data_line(data, nstartcol, col_names)

  blocknum <- 1
  
  data[1, "Start_Time"] <- start_time
  data[1, "Block_Number"] <- blocknum
  data[1, "Comment"] <- NA_character_

  if (is.logical(.progress) && .progress) {
    cli::cli_progress_bar(stringr::str_glue("Reading LabChart file {filename}"),
                          total = filesize)
  } else if (is.character(.progress)) {
    cli::cli_progress_bar(.progress, total = filesize)
    .progress <- TRUE    
  } else if (is.list(.progress)) {
    do.call(cli::cli_progress_bar, c(.progress, total=filesize))
    .progress <- TRUE    
  }
  
  header <- list()
  isblockheader <- FALSE

  i <- 2
  while (TRUE) {
    ln <- readLines(fio, n = 1)
    if (.progress) {
      cli::cli_progress_update(inc = nchar(ln)+1)
    }
    
    if (is.null(ln) || (length(ln) == 0)) {
      break
    }
    
    r <- strsplit(ln, "\t", fixed = TRUE) 
    r <- r[[1]]
    
    if ((length(r) > 0) && check_numeric(r[1])) {
      hd1 <- parse_header_line(ln)
      
      if (is.null(hd1)) {
        stop(stringr::str_glue("Could not parse line {i}: {ln}"))
      } else {
        header[[length(header)+1]] <- hd1 
        isblockheader <- TRUE
      }
      next
    } else if (length(r) < numcols) {
      stop(stringr::str_glue("Error on line {i}: Expected {numcols} columns, found {length(r)}"))
    }
    
    data1 <- parse_data_line(r[1:numcols], nstartcol, col_names)
    data[i, 1:numcols] <- data1
    
    if ((length(r) == numcols + 1) && stringr::str_starts(r[numcols+1], '#')) {
      data[i, "Comment"] <- r[numcols+1]
    } 
    
    if (isblockheader) {
      # we just got the first row of a new block
      blocknum <- blocknum + 1
      
      header <- dplyr::bind_rows(header) |>   
        tibble::column_to_rownames("name")
      
      data[i, "Block_Number"] <- blocknum
      data[i, "Start_Time"] <- header["ExcelDateTime", "value"][[1]][[2]] |> 
        lubridate::parse_date_time(orders = "m/d/Y H:M:S", tz = Sys.timezone())
      
      isblockheader <- FALSE
      header <- list()
    }

    i <- i + 1
  }
  cli::cli_progress_done()
  
  data <- tidyr::fill(data, Block_Number, Start_Time, .direction = "down")
  
  data
}

check_numeric <- function(x)
{
  suppressWarnings(is.na(as.numeric(x)))
}

parse_data_line <- function(r, nstartcol, col_names)
{
  r <- as.list(r)
  
  r[[1]] <- as.numeric(r[[1]])
  if (nstartcol == 2) {
    r[[2]] <- lubridate::parse_date_time(r[[2]], orders = "m/d/Y", tz = Sys.timezone())
  }
  r[(nstartcol+1):length(r)] <- lapply(r[(nstartcol+1):length(r)], 
                                       \(x) suppressWarnings(as.numeric(x)))
  
  r <- tibble::as_tibble(r, .name_repair = "minimal")
  colnames(r) <- col_names
  
  r
}

parse_header_line <- function(ln)
{
  hd1 <- stringr::str_match(ln, "(\\w+)=\\s*(.+)")
  if (is.na(hd1[[1]])) {
    header1 <- NULL
  } else {
    header1 <- hd1 |> 
      tibble::as_tibble(.name_repair = "minimal") |> 
      dplyr::rename(match = 1,
             name = 2,
             value = 3) |> 
      dplyr::select(-match) |> 
      dplyr::mutate(value = str_split(value, '\\t'))
  }
  header1  
}