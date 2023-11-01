require(readr)
require(lubridate)

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
read_labchart <- function(filename, include_units = TRUE)
{
  fio <- file(description = filename,
              open = "rb",
              blocking = TRUE)
  on.exit(close(fio), add = TRUE)
  
  # we hard coded 9 lines for the header, but it's possible ADI may change this at some point in the
  # future.
  # TODO: detect the header size automatically
  headlines <- readLines(fio, n = 9)
  
  header <- str_match(headlines, "(\\w+)=\\s*(.+)") |> 
    as_data_frame() |> 
    rename(match = 1,
           name = 2,
           value = 3) |> 
    select(-match) |> 
    mutate(value = purrr::map_vec(value, ~str_split(.x, '\\t'))) |> 
    column_to_rownames("name")
  
  col_names <- header["ChannelTitle", "value"][[1]]
  col_units <- header["UnitName", "value"][[1]]
  
  if (include_units) {
    col_names <- purrr::map2(col_names, col_units,
                            ~ case_when(
                              .y == "*"  ~  .x,
                              .default = str_c(.x, '.', .y)))
  }

  col_names <- col_names |>
    str_replace("\\s+", "_")
  
  col_names <- c("t.sec", col_names)
  
  dt <- header["Interval", "value"][[1]] |> 
                 str_extract("[+-]*[\\d.]+") |> 
                 as.numeric()
  
  start_time <- header["ExcelDateTime", "value"][[1]][[2]] |> 
    lubridate::parse_date_time(orders = "m/d/Y H:M:S", tz = Sys.timezone())
  
  # TODO: make vroom not display information
  data <- vroom::vroom(fio, col_names = col_names)
  
  probs <- problems(data)
  
  # comment lines
  # TODO: I'm not sure whether these are for comments that span all channels
  # or just for single channel comments. Should be tested
  comments <- probs |> 
    filter(str_detect(actual, '#')) |> 
    mutate(v = str_split(actual, "\\t#"),
           last_val = purrr::map_vec(v, ~ as.numeric(.x[[1]])),
           comment = purrr::map_vec(v, ~ .x[[2]]))
  
  data[comments$row, ncol(data)] <- comments$last_val
  data[comments$row, "Comment"] <- comments$comment

  data[1, "Start_Time"] <- start_time
  
  block_times <- probs |> 
    filter(lag(actual,2) == "ExcelDateTime=") |> 
    mutate(Time = parse_date_time(actual, orders = "m/d/Y H:M:S", tz = Sys.timezone()))

  # TODO: If the sampling rate changes between blocks, we won't detect that change. That
  # might cause a problem

  # the problems matrix is designed to be read by people, not parsed, which
  # makes it annoying here. It lists rows multiple times if it finds problems
  # on them, so we need to look for blocks of sequential rows with problems, which
  # are the block markers
  block_start <- probs |>
    # this builds us "groups" of rows that are repeating or sequential
    mutate(group = row - lag(row) <= 1,
           group = as.numeric(is.na(group) | !group), 
           group = cumsum(group)) |> 
    group_by(group) |> 
    # get rid of a block of rows that are about comments
    filter(!any(str_detect(actual, '#'))) |> 
    # and the beginning of the block is the max row number in the group
    summarize(block_start = max(row)+1) |> 
    pull(block_start)
  
  data[block_start, "Start_Time"] <- block_times$Time
  data[1, "Block_Number"] <- 1
  data[block_start, "Block_Number"] <- seq(2,nrow(block_times)+1)
  
  remove_rows <- probs |> 
    filter(!str_detect(actual, '#')) |> 
    pull(row)
    
  keep_rows <- setdiff(seq(1, nrow(data)), remove_rows)
  
  data <- data[keep_rows, ] |> 
    fill(Block_Number, Start_Time, .direction = "down")

  data
}