required_packages <- c("readr", "dplyr", "lavaan")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    "Install required packages first: install.packages(c(",
    paste(sprintf('"%s"', missing_packages), collapse = ", "),
    "))",
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(lavaan)
})

parse_int_env <- function(name, default) {
  value <- Sys.getenv(name, unset = as.character(default))
  parsed <- suppressWarnings(as.integer(value))
  if (is.na(parsed) || parsed < 0) default else parsed
}

input_file <- Sys.getenv("SEM_INPUT_FILE", unset = "Results.csv")
n_boot <- parse_int_env("SEM_BOOTSTRAPS", 5000)
seed <- parse_int_env("SEM_SEED", 42)
write_outputs <- tolower(Sys.getenv("SEM_WRITE_OUTPUTS", unset = "1")) %in%
  c("1", "true", "yes", "y")
output_dir <- Sys.getenv("SEM_OUTPUT_DIR", unset = "sem_outputs")

latent_indicator_map <- list(
  Emotional_Trust = c("ET_1", "ET_2", "ET_3"),
  Integrity_Trust = c("CTI_1", "CTI_2", "CTI_3")
)
trust_item_vars <- unlist(latent_indicator_map, use.names = FALSE)
structural_path_labels <- c(
  "a1_w", "a1_p", "a2_w", "a2_p", "c_w", "c_p", "c_wp", "b1", "b2"
)

to_numeric <- function(x) suppressWarnings(as.numeric(x))

z_score <- function(x) {
  x <- as.numeric(x)
  x_sd <- sd(x, na.rm = TRUE)
  if (is.na(x_sd) || x_sd == 0) {
    return(rep(NA_real_, length(x)))
  }
  (x - mean(x, na.rm = TRUE)) / x_sd
}

standardized_betas <- function(model) {
  x <- model.matrix(model)
  y <- model.response(model.frame(model))
  beta <- coef(model) * (apply(x, 2, sd) / sd(y))

  out <- data.frame(
    term = names(coef(model)),
    unstd_b = unname(coef(model)),
    std_beta = unname(beta),
    se = unname(summary(model)$coefficients[, "Std. Error"]),
    t = unname(summary(model)$coefficients[, "t value"]),
    p = unname(summary(model)$coefficients[, "Pr(>|t|)"]),
    row.names = NULL
  )

  out[out$term != "(Intercept)", , drop = FALSE]
}

select_existing <- function(data, cols) {
  data[, intersect(cols, names(data)), drop = FALSE]
}

set_value_where <- function(x, condition, value) {
  x[which(condition %in% TRUE)] <- value
  x
}

clean_survey_data <- function(path) {
  df <- readr::read_csv(
    path,
    col_types = readr::cols(.default = readr::col_character()),
    show_col_types = FALSE,
    progress = FALSE
  )

  # Python: df.drop([0, 1]).reset_index(drop=True)
  if (nrow(df) >= 2) {
    df <- df[-c(1, 2), , drop = FALSE]
  }
  df$.python_index <- seq_len(nrow(df)) - 1L

  discussion_filter <- (df$DiscNoPers == "1") | (df$DiscPers == "1")
  attention_filter <- df$AC3 == "1"
  df <- df[which((discussion_filter %in% TRUE) & (attention_filter %in% TRUE)), , drop = FALSE]

  drop_cols <- c(
    "StartDate", "EndDate", "Status", "IPAddress", "Progress",
    "Finished", "RecordedDate", "DiscNoPers", "DiscPers",
    "RecipientLastName", "RecipientFirstName", "RecipientEmail",
    "ExternalReference", "LocationLatitude", "LocationLongitude",
    "DistributionChannel", "UserLanguage", "Q_RecaptchaScore",
    "__js_NextPage_MS", "ResponseId",
    "__js_Prompt1_Clicked", "__js_Prompt2_Clicked", "__js_Prompt3_Clicked",
    "__js_Prompt4_Clicked", "__js_AllPromptsClicked", "Gradio",
    "PROLIFIC_PID", "Q_RecaptchaStatus", "Q_RecaptchaError"
  )
  df <- df[, setdiff(names(df), drop_cols), drop = FALSE]

  warmth_columns <- c("W_Q1_1", "W_Q1_2", "W_Q1_3", "W_Q1_4")
  personalization_columns <- c("MC_Personal_1", "MC_Personal_2")
  emotional_trust_columns <- c("ET_1", "ET_2", "ET_3")
  integrity_trust_columns <- c("CTI_1", "CTI_2", "CTI_3")
  digital_literacy_columns <- c("DL_1", "DL_2", "DL_3", "DL_4", "DL_5", "DL_6", "DL_7")
  personality_columns <- paste0("Personality_", 1:10)

  numeric_cols <- c(
    warmth_columns,
    personalization_columns,
    emotional_trust_columns,
    integrity_trust_columns,
    digital_literacy_columns,
    personality_columns,
    "PostDV_Binary", "PreDV_Binary",
    "PreDV_Slider_1", "PostDV_Slider_1",
    "UF_Q1", "Personalization", "Warmth",
    "Duration (in seconds)"
  )
  numeric_cols <- intersect(numeric_cols, names(df))
  df[numeric_cols] <- lapply(df[numeric_cols], to_numeric)

  # --- Personality item missingness: mean imputation ---------------------
  # Changed from zero-imputation to item-mean imputation. Missing TIPI item
  # responses are replaced with that item's observed mean (computed before
  # reverse-coding, on the raw 1-7 responses).
  personality_columns <- intersect(personality_columns, names(df))
  personality_na_counts <- vapply(
    df[personality_columns],
    function(x) sum(is.na(x)),
    integer(1)
  )
  cat("\n=== Personality (TIPI) item missingness before imputation ===\n")
  print(data.frame(
    item = names(personality_na_counts),
    n_missing = unname(personality_na_counts),
    row.names = NULL
  ))
  cat(sprintf(
    "Total missing TIPI item responses: %d (mean-imputed)\n",
    sum(personality_na_counts)
  ))
  df[personality_columns] <- lapply(df[personality_columns], function(x) {
    x[is.na(x)] <- mean(x, na.rm = TRUE)
    x
  })

  df$Slider_Difference <- df$PreDV_Slider_1 - df$PostDV_Slider_1

  df$SD_ChangeDirection <- "No Change"
  df$SD_ChangeDirection <- set_value_where(
    df$SD_ChangeDirection,
    df$Slider_Difference > 0,
    "Increase"
  )
  df$SD_ChangeDirection <- set_value_where(
    df$SD_ChangeDirection,
    df$Slider_Difference < 0,
    "Decrease"
  )

  df$DV_Binary_Diff <- df$PostDV_Binary - df$PreDV_Binary
  df$DV_Change_Diff <- "ChangeTrust"
  df$DV_Change_Diff <- set_value_where(
    df$DV_Change_Diff,
    (df$PostDV_Binary == 2) & (df$PreDV_Binary == 2),
    "Distrust"
  )
  df$DV_Change_Diff <- set_value_where(
    df$DV_Change_Diff,
    (df$PostDV_Binary == 1) & (df$PreDV_Binary == 1),
    "Trust"
  )
  df$DV_Change_Diff <- set_value_where(
    df$DV_Change_Diff,
    (df$PostDV_Binary == 2) & (df$PreDV_Binary == 1),
    "ChangeDistrust"
  )

  df$Warmth_Check <- rowMeans(select_existing(df, warmth_columns), na.rm = TRUE)
  df$Personalization_Check <- rowMeans(select_existing(df, personalization_columns), na.rm = TRUE)
  df$Emotional_Trust <- rowMeans(select_existing(df, emotional_trust_columns), na.rm = TRUE)
  df$Integrity_Trust <- rowMeans(select_existing(df, integrity_trust_columns), na.rm = TRUE)
  df$Digital_Literacy <- rowMeans(select_existing(df, digital_literacy_columns), na.rm = TRUE)
  df$UF_Q1 <- as.integer(df$UF_Q1)
  df$UF_Q1 <- 8 - df$UF_Q1

  df$IV_Congruence <- 1L
  df$IV_Congruence <- set_value_where(
    df$IV_Congruence,
    (df$Warmth + df$Personalization) == 1,
    0L
  )

  # TIPI reverse coding matches the Python notebook.
  df$P_Extraversion <- (6 - df$Personality_1) + df$Personality_6
  df$P_Agreeableness <- df$Personality_2 + (6 - df$Personality_7)
  df$P_Conscientiousness <- (6 - df$Personality_3) + df$Personality_8
  df$P_Neuroticism <- (6 - df$Personality_4) + df$Personality_9
  df$P_Openness <- (6 - df$Personality_5) + df$Personality_10

  duration <- df$`Duration (in seconds)`
  df <- df[which(!is.na(duration) & duration > 300 & duration < 2500), , drop = FALSE]

  attr(df, "columns") <- list(
    warmth = warmth_columns,
    personalization = personalization_columns,
    emotional_trust = emotional_trust_columns,
    integrity_trust = integrity_trust_columns,
    digital_literacy = digital_literacy_columns,
    personality = paste0("Personality_", 1:10)
  )

  df
}

remove_iqr_outliers_by_cell <- function(data,
                                        outcome,
                                        group_vars,
                                        cells,
                                        iqr_mult = 1.5,
                                        verbose = TRUE) {
  flagged <- rep(FALSE, nrow(data))
  summaries <- list()

  for (i in seq_len(nrow(cells))) {
    cell_mask <- rep(TRUE, nrow(data))
    cell_label_parts <- character(length(group_vars))

    for (j in seq_along(group_vars)) {
      group_var <- group_vars[j]
      group_value <- cells[[group_var]][i]
      cell_mask <- cell_mask & !is.na(data[[group_var]]) & data[[group_var]] == group_value
      cell_label_parts[j] <- paste0(group_var, "=", group_value)
    }

    idx <- which(cell_mask & !is.na(data[[outcome]]))
    if (length(idx) == 0) next

    x <- data[[outcome]][idx]
    q <- quantile(x, c(0.25, 0.75), na.rm = TRUE, type = 7)
    iqr <- q[[2]] - q[[1]]
    lower <- q[[1]] - iqr_mult * iqr
    upper <- q[[2]] + iqr_mult * iqr
    cell_flag <- x < lower | x > upper
    flagged[idx[cell_flag]] <- TRUE

    summaries[[length(summaries) + 1L]] <- data.frame(
      cell = paste(cell_label_parts, collapse = ", "),
      n = length(x),
      q1 = unname(q[[1]]),
      q3 = unname(q[[2]]),
      iqr = unname(iqr),
      lower = unname(lower),
      upper = unname(upper),
      flagged = sum(cell_flag),
      stringsAsFactors = FALSE
    )
  }

  summary_table <- dplyr::bind_rows(summaries)

  if (verbose) {
    cat("\n=== IQR outlier removal by Warmth x Personalization ===\n")
    print(summary_table, row.names = FALSE)
    cat(sprintf(
      "\nIQR step: flagged %d; N: %d -> %d\n",
      sum(flagged), nrow(data), nrow(data) - sum(flagged)
    ))
  }

  result <- data[!flagged, , drop = FALSE]
  attr(result, "iqr_summary") <- summary_table
  attr(result, "iqr_outliers") <- data[flagged, , drop = FALSE]
  result
}

remove_outliers_by_cooks_d <- function(data,
                                       formula,
                                       threshold_mult = 4,
                                       verbose = TRUE) {
  required <- all.vars(formula)
  d <- data[complete.cases(data[, required, drop = FALSE]), , drop = FALSE]
  fit <- lm(formula, data = d)
  cooks <- cooks.distance(fit)
  threshold <- threshold_mult / nrow(d)
  flagged <- cooks > threshold

  diagnostics <- data.frame(
    row_in_model = seq_len(nrow(d)),
    python_index = if (".python_index" %in% names(d)) d$.python_index else NA_integer_,
    cooks_d = unname(cooks),
    flagged = unname(flagged),
    stringsAsFactors = FALSE
  )

  if (verbose) {
    cat("\n=== Cook's D outlier removal from OLS influence ===\n")
    cat(sprintf(
      "Cook's D step: threshold = %d/n = %.4f; flagged %d; N: %d -> %d\n",
      threshold_mult,
      threshold,
      sum(flagged, na.rm = TRUE),
      nrow(d),
      sum(!flagged, na.rm = TRUE)
    ))
    if (any(flagged, na.rm = TRUE)) {
      print(diagnostics[flagged, , drop = FALSE], row.names = FALSE)
    }
  }

  result <- d[!flagged, , drop = FALSE]
  attr(result, "cooks_diagnostics") <- diagnostics
  attr(result, "initial_lm") <- fit
  result
}

# Tidy a fitted lm into a labelled coefficient table for cross-model comparison.
tidy_ols <- function(model, model_label) {
  coefs <- summary(model)$coefficients
  data.frame(
    model = model_label,
    term = rownames(coefs),
    estimate = unname(coefs[, "Estimate"]),
    se = unname(coefs[, "Std. Error"]),
    t = unname(coefs[, "t value"]),
    p = unname(coefs[, "Pr(>|t|)"]),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
}

# Side-by-side comparison of the IQR-only and IQR + Cook's D OLS models.
compare_ols_models <- function(iqr_only_model, cooks_model) {
  iqr_tidy <- tidy_ols(iqr_only_model, "IQR_only")
  cooks_tidy <- tidy_ols(cooks_model, "IQR_plus_CooksD")

  names(iqr_tidy)[names(iqr_tidy) %in% c("estimate", "se", "t", "p")] <-
    paste0("iqr_", c("estimate", "se", "t", "p"))
  names(cooks_tidy)[names(cooks_tidy) %in% c("estimate", "se", "t", "p")] <-
    paste0("cooks_", c("estimate", "se", "t", "p"))

  iqr_tidy$model <- NULL
  cooks_tidy$model <- NULL

  comparison <- merge(iqr_tidy, cooks_tidy, by = "term", all = TRUE, sort = FALSE)
  comparison$delta_estimate <- comparison$cooks_estimate - comparison$iqr_estimate
  comparison$sign_flip <- sign(comparison$iqr_estimate) != sign(comparison$cooks_estimate)
  comparison$sig_change <- (comparison$iqr_p < 0.05) != (comparison$cooks_p < 0.05)

  comparison[, c(
    "term",
    "iqr_estimate", "iqr_se", "iqr_p",
    "cooks_estimate", "cooks_se", "cooks_p",
    "delta_estimate", "sign_flip", "sig_change"
  ), drop = FALSE]
}

# Zero-order (Pearson) correlation matrix plus a long-format table with
# pairwise n and p-values, for the key model variables.
zero_order_correlations <- function(data, vars, verbose = TRUE) {
  vars <- intersect(vars, names(data))
  d <- data[, vars, drop = FALSE]
  d[] <- lapply(d, as.numeric)

  cor_mat <- cor(d, use = "pairwise.complete.obs", method = "pearson")

  pairs <- t(combn(vars, 2))
  long <- lapply(seq_len(nrow(pairs)), function(i) {
    v1 <- pairs[i, 1]
    v2 <- pairs[i, 2]
    ok <- stats::complete.cases(d[, c(v1, v2)])
    n_pair <- sum(ok)
    if (n_pair > 3) {
      ct <- suppressWarnings(cor.test(d[[v1]][ok], d[[v2]][ok], method = "pearson"))
      r <- unname(ct$estimate)
      p <- ct$p.value
    } else {
      r <- NA_real_
      p <- NA_real_
    }
    data.frame(
      var1 = v1, var2 = v2, n = n_pair, r = r, p = p,
      stringsAsFactors = FALSE
    )
  })
  long_table <- dplyr::bind_rows(long)

  if (verbose) {
    cat("\n=== Zero-order correlation matrix (Pearson, pairwise) ===\n")
    print(round(cor_mat, 3))
    cat("\n=== Zero-order correlations: pairwise n and p-values ===\n")
    long_print <- long_table
    long_print$r <- round(long_print$r, 3)
    long_print$p <- round(long_print$p, 4)
    print(long_print, row.names = FALSE)
  }

  list(matrix = cor_mat, long = long_table)
}

prepare_sem_data <- function(ancova_df) {
  df_sem <- ancova_df
  df_sem$Warmth_x_Personalization <- df_sem$Warmth * df_sem$Personalization
  df_sem$IV_Congruence <- as.integer(df_sem$Warmth == df_sem$Personalization)

  continuous_vars <- c(
    "Slider_Difference",
    "Digital_Literacy",
    "P_Extraversion",
    "P_Agreeableness",
    "P_Openness",
    "P_Conscientiousness",
    "P_Neuroticism",
    "UF_Q1"
  )

  df_model <- df_sem
  df_model[continuous_vars] <- lapply(df_model[continuous_vars], z_score)

  model_vars <- c(
    "Slider_Difference",
    trust_item_vars,
    "Warmth",
    "Personalization",
    "Warmth_x_Personalization",
    "Digital_Literacy",
    "P_Extraversion",
    "P_Agreeableness",
    "P_Openness",
    "P_Conscientiousness",
    "P_Neuroticism",
    "UF_Q1"
  )

  df_model <- df_model[complete.cases(df_model[, model_vars, drop = FALSE]), model_vars, drop = FALSE]
  df_model[] <- lapply(df_model, as.numeric)
  df_model
}

safe_fit_measures <- function(fit, fit_measure_names) {
  values <- lapply(fit_measure_names, function(measure) {
    tryCatch(
      unname(fitMeasures(fit, measure)),
      error = function(e) NA_real_
    )
  })

  data.frame(
    measure = fit_measure_names,
    value = unlist(values, use.names = FALSE),
    row.names = NULL
  )
}

fit_measures_wide <- function(fit, model_name, fit_measure_names) {
  fit_df <- safe_fit_measures(fit, fit_measure_names)
  values <- as.list(fit_df$value)
  names(values) <- fit_df$measure
  data.frame(
    model = model_name,
    values,
    check.names = FALSE,
    row.names = NULL
  )
}

calculate_ave_cr <- function(parameter_estimates, indicator_map) {
  rows <- lapply(names(indicator_map), function(construct) {
    indicators <- indicator_map[[construct]]
    loadings <- parameter_estimates[
      parameter_estimates$op == "=~" &
        parameter_estimates$lhs == construct &
        parameter_estimates$rhs %in% indicators,
      ,
      drop = FALSE
    ]
    loadings <- loadings[match(indicators, loadings$rhs), , drop = FALSE]
    lambda <- loadings$std.all

    residuals <- parameter_estimates[
      parameter_estimates$op == "~~" &
        parameter_estimates$lhs == parameter_estimates$rhs &
        parameter_estimates$lhs %in% indicators,
      ,
      drop = FALSE
    ]
    residuals <- residuals[match(indicators, residuals$lhs), , drop = FALSE]
    theta <- residuals$std.all
    missing_theta <- is.na(theta)
    theta[missing_theta] <- 1 - lambda[missing_theta]^2

    lambda_sq_sum <- sum(lambda^2, na.rm = TRUE)
    theta_sum <- sum(theta, na.rm = TRUE)
    lambda_sum <- sum(lambda, na.rm = TRUE)

    data.frame(
      construct = construct,
      indicators = paste(indicators, collapse = ", "),
      n_indicators = length(indicators),
      min_std_loading = min(lambda, na.rm = TRUE),
      max_std_loading = max(lambda, na.rm = TRUE),
      ave = lambda_sq_sum / (lambda_sq_sum + theta_sum),
      cr = lambda_sum^2 / (lambda_sum^2 + theta_sum),
      ave_ge_0_50 = lambda_sq_sum / (lambda_sq_sum + theta_sum) >= 0.50,
      cr_ge_0_70 = lambda_sum^2 / (lambda_sum^2 + theta_sum) >= 0.70,
      stringsAsFactors = FALSE
    )
  })

  dplyr::bind_rows(rows)
}

calculate_discriminant_validity <- function(parameter_estimates, reliability_table) {
  constructs <- reliability_table$construct
  latent_covs <- parameter_estimates[
    parameter_estimates$op == "~~" &
      parameter_estimates$lhs != parameter_estimates$rhs &
      parameter_estimates$lhs %in% constructs &
      parameter_estimates$rhs %in% constructs,
    ,
    drop = FALSE
  ]

  if (nrow(latent_covs) == 0) {
    return(data.frame())
  }

  latent_covs$sqrt_ave_lhs <- sqrt(reliability_table$ave[match(latent_covs$lhs, reliability_table$construct)])
  latent_covs$sqrt_ave_rhs <- sqrt(reliability_table$ave[match(latent_covs$rhs, reliability_table$construct)])
  latent_covs$latent_correlation <- latent_covs$std.all
  latent_covs$squared_correlation <- latent_covs$std.all^2
  latent_covs$fornell_larcker_pass <- (
    latent_covs$sqrt_ave_lhs > abs(latent_covs$latent_correlation) &
      latent_covs$sqrt_ave_rhs > abs(latent_covs$latent_correlation)
  )

  latent_covs[, c(
    "lhs", "rhs", "latent_correlation", "squared_correlation",
    "sqrt_ave_lhs", "sqrt_ave_rhs", "fornell_larcker_pass"
  ), drop = FALSE]
}

extract_admissibility_checks <- function(fit, parameter_estimates = NULL) {
  if (is.null(parameter_estimates)) {
    parameter_estimates <- parameterEstimates(fit, standardized = TRUE)
  }

  latent_names <- tryCatch(lavNames(fit, type = "lv"), error = function(e) character())
  observed_names <- tryCatch(lavNames(fit, type = "ov"), error = function(e) character())

  variances <- parameter_estimates[
    parameter_estimates$op == "~~" &
      parameter_estimates$lhs == parameter_estimates$rhs,
    ,
    drop = FALSE
  ]

  if (nrow(variances) == 0) {
    return(data.frame())
  }

  variances$variable_type <- ifelse(
    variances$lhs %in% latent_names,
    "latent",
    ifelse(variances$lhs %in% observed_names, "observed", "unknown")
  )
  variances$negative_variance <- variances$est < 0
  variances$near_zero_variance <- variances$est >= 0 & variances$est < 1e-6
  variances$negative_std_variance <- variances$std.all < 0

  variances[, c(
    "lhs", "variable_type", "est", "se", "z", "pvalue", "std.all",
    "negative_variance", "near_zero_variance", "negative_std_variance"
  ), drop = FALSE]
}

# Harman's single-factor check removed at user request. The CFA-based
# common-method factor comparison below is retained as the common-method
# bias diagnostic.
run_measurement_diagnostics <- function(df_model,
                                        indicator_map,
                                        two_factor_model,
                                        one_factor_model,
                                        common_method_model) {
  fit_measure_names <- c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr", "aic", "bic")

  fit_two_factor <- lavaan::cfa(two_factor_model, data = df_model, estimator = "ML")
  fit_one_factor <- lavaan::cfa(one_factor_model, data = df_model, estimator = "ML")
  fit_common_method <- lavaan::cfa(common_method_model, data = df_model, estimator = "ML")

  fit_comparison <- dplyr::bind_rows(
    fit_measures_wide(fit_two_factor, "Two-factor trust CFA", fit_measure_names),
    fit_measures_wide(fit_one_factor, "One-factor trust CFA", fit_measure_names),
    fit_measures_wide(fit_common_method, "Two-factor CFA + common method factor", fit_measure_names)
  )

  two_factor_cfi <- fit_comparison$cfi[fit_comparison$model == "Two-factor trust CFA"]
  two_factor_rmsea <- fit_comparison$rmsea[fit_comparison$model == "Two-factor trust CFA"]
  fit_comparison$delta_cfi_vs_two_factor <- fit_comparison$cfi - two_factor_cfi
  fit_comparison$delta_rmsea_vs_two_factor <- fit_comparison$rmsea - two_factor_rmsea

  method_pe <- parameterEstimates(fit_common_method, standardized = TRUE)
  method_loadings <- method_pe[
    method_pe$op == "=~" & method_pe$lhs == "Common_Method",
    c("lhs", "op", "rhs", "est", "se", "z", "pvalue", "std.all"),
    drop = FALSE
  ]
  method_variance_param <- method_pe[
    method_pe$op == "~~" &
      method_pe$lhs == "Common_Method" &
      method_pe$rhs == "Common_Method",
    ,
    drop = FALSE
  ]
  method_variance <- data.frame(
    method_variance_est = method_variance_param$est,
    method_variance_pvalue = method_variance_param$pvalue,
    mean_method_variance_pct = mean(method_loadings$std.all^2, na.rm = TRUE) * 100,
    max_method_variance_pct = max(method_loadings$std.all^2, na.rm = TRUE) * 100,
    stringsAsFactors = FALSE
  )

  list(
    fit_comparison = fit_comparison,
    common_method_loadings = method_loadings,
    common_method_variance = method_variance
  )
}

extract_labelled_structural_paths <- function(parameter_estimates, path_labels) {
  paths <- parameter_estimates[
    parameter_estimates$op == "~" & parameter_estimates$label %in% path_labels,
    ,
    drop = FALSE
  ]

  paths[, c("label", "lhs", "rhs", "est", "se", "z", "pvalue", "std.all"), drop = FALSE]
}

compare_common_method_paths <- function(main_parameter_estimates,
                                        common_method_fit,
                                        path_labels) {
  method_parameter_estimates <- parameterEstimates(
    common_method_fit,
    standardized = TRUE,
    ci = TRUE,
    level = 0.95
  )

  main_paths <- extract_labelled_structural_paths(main_parameter_estimates, path_labels)
  method_paths <- extract_labelled_structural_paths(method_parameter_estimates, path_labels)

  names(main_paths)[names(main_paths) %in% c("est", "se", "z", "pvalue", "std.all")] <-
    paste0("main_", names(main_paths)[names(main_paths) %in% c("est", "se", "z", "pvalue", "std.all")])
  names(method_paths)[names(method_paths) %in% c("est", "se", "z", "pvalue", "std.all")] <-
    paste0("cmb_", names(method_paths)[names(method_paths) %in% c("est", "se", "z", "pvalue", "std.all")])

  comparison <- merge(
    main_paths,
    method_paths,
    by = c("label", "lhs", "rhs"),
    all = TRUE,
    sort = FALSE
  )
  comparison$delta_std_all <- comparison$cmb_std.all - comparison$main_std.all
  comparison
}

# Latent-variable PARALLEL mediation SEM: the two trust constructs are measured
# by their item-level indicators and modeled side by side (correlated residual,
# no path between them). The Warmth x Personalization term is a direct
# predictor of the outcome and is not modeled as a mediated effect.
sem_model <- '
  Emotional_Trust =~ ET_1 + ET_2 + ET_3
  Integrity_Trust =~ CTI_1 + CTI_2 + CTI_3

  Emotional_Trust ~ a1_w*Warmth + a1_p*Personalization +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  Integrity_Trust ~ a2_w*Warmth + a2_p*Personalization +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  Slider_Difference ~ c_w*Warmth + c_p*Personalization + c_wp*Warmth_x_Personalization +
    b1*Emotional_Trust + b2*Integrity_Trust +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  Emotional_Trust ~~ Integrity_Trust

  ind_w_emotional := a1_w*b1
  ind_w_integrity := a2_w*b2
  ind_w_total := a1_w*b1 + a2_w*b2
  direct_w := c_w
  total_w := c_w + ind_w_total

  ind_p_emotional := a1_p*b1
  ind_p_integrity := a2_p*b2
  ind_p_total := a1_p*b1 + a2_p*b2
  direct_p := c_p
  total_p := c_p + ind_p_total

  direct_wp := c_wp
'

trust_measurement_model <- '
  Emotional_Trust =~ ET_1 + ET_2 + ET_3
  Integrity_Trust =~ CTI_1 + CTI_2 + CTI_3

  Emotional_Trust ~~ Integrity_Trust
'

trust_one_factor_model <- '
  Trust_Common =~ ET_1 + ET_2 + ET_3 + CTI_1 + CTI_2 + CTI_3
'

trust_common_method_model <- '
  Emotional_Trust =~ ET_1 + ET_2 + ET_3
  Integrity_Trust =~ CTI_1 + CTI_2 + CTI_3
  Common_Method =~ 1*ET_1 + 1*ET_2 + 1*ET_3 + 1*CTI_1 + 1*CTI_2 + 1*CTI_3

  Emotional_Trust ~~ Integrity_Trust
  Common_Method ~~ 0*Emotional_Trust
  Common_Method ~~ 0*Integrity_Trust
'

sem_common_method_model <- '
  Emotional_Trust =~ ET_1 + ET_2 + ET_3
  Integrity_Trust =~ CTI_1 + CTI_2 + CTI_3
  Common_Method =~ 1*ET_1 + 1*ET_2 + 1*ET_3 + 1*CTI_1 + 1*CTI_2 + 1*CTI_3

  Emotional_Trust ~ a1_w*Warmth + a1_p*Personalization +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  Integrity_Trust ~ a2_w*Warmth + a2_p*Personalization +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  Slider_Difference ~ c_w*Warmth + c_p*Personalization + c_wp*Warmth_x_Personalization +
    b1*Emotional_Trust + b2*Integrity_Trust +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  Emotional_Trust ~~ Integrity_Trust
  Common_Method ~~ 0*Emotional_Trust
  Common_Method ~~ 0*Integrity_Trust

  ind_w_emotional := a1_w*b1
  ind_w_integrity := a2_w*b2
  ind_w_total := a1_w*b1 + a2_w*b2
  direct_w := c_w
  total_w := c_w + ind_w_total

  ind_p_emotional := a1_p*b1
  ind_p_integrity := a2_p*b2
  ind_p_total := a1_p*b1 + a2_p*b2
  direct_p := c_p
  total_p := c_p + ind_p_total

  direct_wp := c_wp
'

cat(sprintf("Reading data from %s\n", input_file))
df <- clean_survey_data(input_file)
cat(sprintf("After Python-equivalent cleaning: N = %d\n", nrow(df)))
cat("\nCell counts after initial cleaning:\n")
print(table(df$Warmth, df$Personalization, useNA = "ifany"))

condition_cells <- data.frame(
  Warmth = c(0, 0, 1, 1),
  Personalization = c(0, 1, 0, 1)
)

df_clean <- remove_iqr_outliers_by_cell(
  data = df,
  outcome = "Slider_Difference",
  group_vars = c("Warmth", "Personalization"),
  cells = condition_cells,
  iqr_mult = 1.5
)

ancova_required <- c(
  "Slider_Difference", "Warmth", "Personalization",
  "ET_1", "ET_2", "ET_3", "CTI_1", "CTI_2", "CTI_3",
  "Emotional_Trust", "Integrity_Trust", "Digital_Literacy", "UF_Q1",
  "P_Extraversion", "P_Agreeableness", "P_Openness",
  "P_Conscientiousness", "P_Neuroticism",
  "SD_ChangeDirection", "DV_Change_Diff"
)

ols_formula <- as.formula(
  paste(
    "Slider_Difference ~ factor(Warmth) * factor(Personalization)",
    "+ Emotional_Trust + Integrity_Trust + Digital_Literacy + UF_Q1",
    "+ P_Extraversion + P_Agreeableness + P_Openness",
    "+ P_Conscientiousness + P_Neuroticism"
  )
)

# --- IQR-only OLS model -----------------------------------------------------
# Fit the OLS model on the IQR-cleaned data BEFORE Cook's D removal, so the
# Cook's D step can be evaluated as a sensitivity analysis rather than a
# silent precondition for the headline interaction.
ancova_df_iqr_only <- df_clean[
  complete.cases(df_clean[, ancova_required, drop = FALSE]),
  ,
  drop = FALSE
]
ancova_df_iqr_only$Warmth <- as.integer(ancova_df_iqr_only$Warmth)
ancova_df_iqr_only$Personalization <- as.integer(ancova_df_iqr_only$Personalization)

model_iqr_only <- lm(ols_formula, data = ancova_df_iqr_only)
cat("\n=== IQR-only OLS model (before Cook's D removal) ===\n")
cat(sprintf("N = %d\n", nrow(ancova_df_iqr_only)))
print(summary(model_iqr_only))

cat("\nCell counts after IQR-only removal:\n")
print(table(ancova_df_iqr_only$Warmth, ancova_df_iqr_only$Personalization, useNA = "ifany"))

# --- IQR + Cook's D OLS model ----------------------------------------------
ancova_df <- remove_outliers_by_cooks_d(
  data = ancova_df_iqr_only,
  formula = ols_formula,
  threshold_mult = 4
)

model_robust <- lm(ols_formula, data = ancova_df)
cat("\n=== Robust OLS model after IQR + Cook's D removal ===\n")
print(summary(model_robust))

cat("\n=== Standardized betas for robust OLS model ===\n")
std_beta_table <- standardized_betas(model_robust)
numeric_std_cols <- vapply(std_beta_table, is.numeric, logical(1))
std_beta_table[numeric_std_cols] <- lapply(std_beta_table[numeric_std_cols], round, 4)
print(std_beta_table, row.names = FALSE)

cat("\nCell counts after both outlier removal steps:\n")
print(table(ancova_df$Warmth, ancova_df$Personalization, useNA = "ifany"))

# --- IQR-only vs IQR + Cook's D comparison ---------------------------------
ols_model_comparison <- compare_ols_models(model_iqr_only, model_robust)
ols_comparison_print <- ols_model_comparison
numeric_cmp_cols <- vapply(ols_comparison_print, is.numeric, logical(1))
ols_comparison_print[numeric_cmp_cols] <- lapply(
  ols_comparison_print[numeric_cmp_cols], round, 4
)
cat("\n=== OLS model comparison: IQR-only vs IQR + Cook's D ===\n")
cat("(sign_flip = coefficient changes sign; sig_change = crosses p = .05)\n")
print(ols_comparison_print, row.names = FALSE)

interaction_term <- "factor(Warmth)1:factor(Personalization)1"
interaction_row <- ols_model_comparison[ols_model_comparison$term == interaction_term, , drop = FALSE]
if (nrow(interaction_row) == 1) {
  cat("\n--- Warmth x Personalization interaction across models ---\n")
  cat(sprintf(
    "IQR-only:        b = %.3f, SE = %.3f, p = %.4f\n",
    interaction_row$iqr_estimate, interaction_row$iqr_se, interaction_row$iqr_p
  ))
  cat(sprintf(
    "IQR + Cook's D:  b = %.3f, SE = %.3f, p = %.4f\n",
    interaction_row$cooks_estimate, interaction_row$cooks_se, interaction_row$cooks_p
  ))
  if (isTRUE(interaction_row$sig_change)) {
    cat("NOTE: the interaction crosses the p = .05 threshold between models.\n")
  } else {
    cat("The interaction's significance status is stable across both models.\n")
  }
}

df_model <- prepare_sem_data(ancova_df)

# --- Zero-order correlations -----------------------------------------------
# Computed on the analysis sample (post IQR + Cook's D, pre-standardization
# values from ancova_df) so the bivariate signs can be read alongside the
# partial coefficients in the SEM. Includes both trust composites and the DV.
zero_order_vars <- c(
  "Slider_Difference",
  "Emotional_Trust",
  "Integrity_Trust",
  "Warmth",
  "Personalization",
  "Digital_Literacy",
  "UF_Q1",
  "P_Extraversion",
  "P_Agreeableness",
  "P_Openness",
  "P_Conscientiousness",
  "P_Neuroticism"
)
zero_order <- zero_order_correlations(ancova_df, zero_order_vars)

measurement_diagnostics <- run_measurement_diagnostics(
  df_model = df_model,
  indicator_map = latent_indicator_map,
  two_factor_model = trust_measurement_model,
  one_factor_model = trust_one_factor_model,
  common_method_model = trust_common_method_model
)

cat("\n=== Trust measurement and common-method CFA fit comparison ===\n")
print(measurement_diagnostics$fit_comparison, row.names = FALSE)

cat("\n=== Equal-loading common method factor variance ===\n")
print(measurement_diagnostics$common_method_variance, row.names = FALSE)

cat("\n=== SEM data ===\n")
cat(sprintf("N = %d\n", nrow(df_model)))
cat(sprintf(
  "Bootstrap draws = %d%s\n",
  n_boot,
  if (n_boot == 0) " (standard SEs)" else ""
))

set.seed(seed)
sem_args <- list(
  model = sem_model,
  data = df_model,
  estimator = "ML",
  fixed.x = TRUE
)

if (n_boot > 0) {
  sem_args$se <- "bootstrap"
  sem_args$bootstrap <- n_boot
} else {
  sem_args$se <- "standard"
}

fit_sem <- do.call(lavaan::sem, sem_args)

cat("\n=== SEM summary ===\n")
print(summary(
  fit_sem,
  standardized = TRUE,
  ci = TRUE,
  rsquare = TRUE,
  fit.measures = TRUE
))

boot_ci_type <- if (n_boot > 0) "perc" else "norm"
parameter_estimates <- parameterEstimates(
  fit_sem,
  standardized = TRUE,
  ci = TRUE,
  boot.ci.type = boot_ci_type,
  level = 0.95
)

latent_reliability <- calculate_ave_cr(parameter_estimates, latent_indicator_map)
discriminant_validity <- calculate_discriminant_validity(parameter_estimates, latent_reliability)
sem_admissibility_checks <- extract_admissibility_checks(fit_sem, parameter_estimates)

cat("\n=== Latent construct reliability and convergent validity ===\n")
print(latent_reliability, row.names = FALSE)

cat("\n=== Discriminant validity check ===\n")
print(discriminant_validity, row.names = FALSE)

cat("\n=== SEM variance admissibility checks ===\n")
print(sem_admissibility_checks, row.names = FALSE)

defined_effect_cols <- c(
  "lhs", "op", "rhs", "label", "est", "se", "z", "pvalue",
  "ci.lower", "ci.upper", "std.all", "std.nox"
)
defined_effects <- parameter_estimates[parameter_estimates$op == ":=", , drop = FALSE]
defined_effects <- defined_effects[, intersect(defined_effect_cols, names(defined_effects)), drop = FALSE]

cat("\n=== Defined direct, indirect, and total effects ===\n")
print(defined_effects, row.names = FALSE)

fit_measure_names <- c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr", "aic", "bic")
fit_measures <- fitMeasures(fit_sem, fit_measure_names)
fit_measures_df <- data.frame(
  measure = names(fit_measures),
  value = unname(fit_measures),
  row.names = NULL
)

cat("\n=== Fit measures ===\n")
print(fit_measures_df, row.names = FALSE)

fit_sem_common_method <- lavaan::sem(
  sem_common_method_model,
  data = df_model,
  estimator = "ML",
  fixed.x = TRUE,
  se = "standard"
)
common_method_fit_measures <- safe_fit_measures(
  fit_sem_common_method,
  c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr", "aic", "bic")
)
common_method_admissibility_checks <- extract_admissibility_checks(fit_sem_common_method)
common_method_path_comparison <- compare_common_method_paths(
  main_parameter_estimates = parameter_estimates,
  common_method_fit = fit_sem_common_method,
  path_labels = structural_path_labels
)

cat("\n=== Common-method-adjusted SEM fit measures ===\n")
print(common_method_fit_measures, row.names = FALSE)

cat("\n=== Common-method-adjusted variance admissibility checks ===\n")
print(common_method_admissibility_checks, row.names = FALSE)

cat("\n=== Common-method-adjusted structural path comparison ===\n")
print(common_method_path_comparison, row.names = FALSE)

if (write_outputs) {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  readr::write_csv(df_clean, file.path(output_dir, "df_after_iqr.csv"))
  readr::write_csv(ancova_df_iqr_only, file.path(output_dir, "ancova_after_iqr_only.csv"))
  readr::write_csv(ancova_df, file.path(output_dir, "ancova_after_iqr_and_cooksd.csv"))
  readr::write_csv(attr(df_clean, "iqr_summary"), file.path(output_dir, "iqr_summary.csv"))
  readr::write_csv(attr(ancova_df, "cooks_diagnostics"), file.path(output_dir, "cooks_diagnostics.csv"))
  readr::write_csv(tidy_ols(model_iqr_only, "IQR_only"), file.path(output_dir, "ols_iqr_only.csv"))
  readr::write_csv(tidy_ols(model_robust, "IQR_plus_CooksD"), file.path(output_dir, "ols_iqr_plus_cooksd.csv"))
  readr::write_csv(ols_model_comparison, file.path(output_dir, "ols_model_comparison.csv"))
  zero_order_matrix_df <- as.data.frame(round(zero_order$matrix, 4))
  zero_order_matrix_df <- cbind(variable = rownames(zero_order_matrix_df), zero_order_matrix_df)
  readr::write_csv(zero_order_matrix_df, file.path(output_dir, "zero_order_correlation_matrix.csv"))
  readr::write_csv(zero_order$long, file.path(output_dir, "zero_order_correlations_long.csv"))
  readr::write_csv(parameter_estimates, file.path(output_dir, "sem_parameter_estimates.csv"))
  readr::write_csv(defined_effects, file.path(output_dir, "sem_defined_effects.csv"))
  readr::write_csv(fit_measures_df, file.path(output_dir, "sem_fit_measures.csv"))
  readr::write_csv(latent_reliability, file.path(output_dir, "sem_latent_reliability_ave_cr.csv"))
  readr::write_csv(discriminant_validity, file.path(output_dir, "sem_discriminant_validity.csv"))
  readr::write_csv(sem_admissibility_checks, file.path(output_dir, "sem_admissibility_checks.csv"))
  readr::write_csv(
    measurement_diagnostics$fit_comparison,
    file.path(output_dir, "sem_measurement_fit_comparison.csv")
  )
  readr::write_csv(
    measurement_diagnostics$common_method_loadings,
    file.path(output_dir, "sem_common_method_loadings.csv")
  )
  readr::write_csv(
    measurement_diagnostics$common_method_variance,
    file.path(output_dir, "sem_common_method_variance.csv")
  )
  readr::write_csv(common_method_fit_measures, file.path(output_dir, "sem_common_method_fit_measures.csv"))
  readr::write_csv(
    common_method_admissibility_checks,
    file.path(output_dir, "sem_common_method_admissibility_checks.csv")
  )
  readr::write_csv(
    common_method_path_comparison,
    file.path(output_dir, "sem_common_method_path_comparison.csv")
  )

  cat(sprintf("\nWrote output tables to %s\n", output_dir))
}