library(dplyr)
library(readr)
library(lavaan)

# ---- Data preparation ----
df <- read_csv("Results.csv")

# Drop first two rows (Qualtrics metadata rows)
df <- df[-c(1, 2), ]

# Filter: assigned to one of the two discussion conditions AND passed AC3
df <- df %>%
  filter((DiscNoPers == "1" | DiscPers == "1")) %>%
  filter(AC3 == "1")

# Duration filter (must happen BEFORE we drop the Duration column)
df$`Duration (in seconds)` <- suppressWarnings(as.numeric(df$`Duration (in seconds)`))
df <- df %>%
  filter(`Duration (in seconds)` > 300 & `Duration (in seconds)` < 2500)

# Drop columns we don't need
drop_cols <- c("StartDate","EndDate","Status","IPAddress","Progress",
               "Duration (in seconds)","Finished","RecordedDate",
               "DiscNoPers","DiscPers",
               "RecipientLastName","RecipientFirstName","RecipientEmail",
               "ExternalReference","LocationLatitude","LocationLongitude",
               "DistributionChannel","UserLanguage","Q_RecaptchaScore",
               "__js_NextPage_MS","__js_Prompt1_Clicked","__js_Prompt2_Clicked",
               "__js_Prompt3_Clicked","__js_Prompt4_Clicked","__js_AllPromptsClicked",
               "Gradio","PROLIFIC_PID","ResponseId",
               "Q_RecaptchaStatus","Q_RecaptchaError")
df <- df %>% select(-any_of(drop_cols))

warmth_cols      <- c("W_Q1_1","W_Q1_2","W_Q1_3","W_Q1_4")
pers_cols        <- c("MC_Personal_1","MC_Personal_2")
et_cols          <- c("ET_1","ET_2","ET_3")
cti_cols         <- c("CTI_1","CTI_2","CTI_3")
ctc_cols         <- c("CTC_1","CTC_2")  # kept for CFA validation only
dl_cols          <- c("DL_1","DL_2","DL_3","DL_4","DL_5","DL_6","DL_7")
pers_items       <- paste0("Personality_", 1:10)

num_cols <- c(warmth_cols, pers_cols, et_cols, cti_cols, ctc_cols,
              dl_cols, pers_items,
              "PreDV_Slider_1","PostDV_Slider_1","PreDV_Binary","PostDV_Binary",
              "Personalization","Warmth","UF_Q1")
df[num_cols] <- lapply(df[num_cols], function(x) suppressWarnings(as.numeric(x)))

# Personality items: replace NA with 0
df[pers_items] <- lapply(df[pers_items], function(x) ifelse(is.na(x), 0, x))

# UF_Q1 as integer (matches Python .astype(int))
df$UF_Q1 <- as.integer(df$UF_Q1)

# Composites
df$Slider_Difference  <- df$PreDV_Slider_1 - df$PostDV_Slider_1
df$DV_Binary_Diff     <- df$PostDV_Binary - df$PreDV_Binary
df$DV_Binary_Diff_Cat <- ifelse(df$DV_Binary_Diff == -1, 0, df$DV_Binary_Diff)

df$Warmth_Check          <- rowMeans(df[warmth_cols], na.rm = TRUE)
df$Personalization_Check <- rowMeans(df[pers_cols],   na.rm = TRUE)
df$Emotional_Trust       <- rowMeans(df[et_cols],     na.rm = TRUE)
df$Integrity_Trust       <- rowMeans(df[cti_cols],    na.rm = TRUE)
df$Digital_Literacy      <- rowMeans(df[dl_cols],     na.rm = TRUE)

df$WarmthxPersonalization <- ifelse((df$Warmth + df$Personalization) == 1, 1, 0)
df$W_x_P <- df$Warmth * df$Personalization

# TIPI personality
df$P_Extraversion      <- (6 - df$Personality_1)  + df$Personality_6
df$P_Agreeableness     <-       df$Personality_2  + (6 - df$Personality_7)
df$P_Conscientiousness <- (6 - df$Personality_3)  + df$Personality_8
df$P_Neuroticism       <- (6 - df$Personality_4)  + df$Personality_9
df$P_Openness          <- (6 - df$Personality_5)  + df$Personality_10

# ---- Outlier removal helpers ----
remove_outliers_by_cell <- function(data, vars, group_vars,
                                    iqr_mult = 1.5,
                                    verbose = TRUE) {
  d <- data[complete.cases(data[, c(vars, group_vars)]), ]
  group_key <- interaction(d[, group_vars, drop = FALSE], drop = TRUE)
  flag <- rep(FALSE, nrow(d))
  
  for (g in levels(group_key)) {
    cell_idx <- which(group_key == g)
    for (v in vars) {
      x <- d[[v]][cell_idx]
      q <- quantile(x, c(.25, .75), na.rm = TRUE, type = 7)
      iqr <- q[2] - q[1]
      lo  <- q[1] - iqr_mult * iqr
      hi  <- q[2] + iqr_mult * iqr
      cell_flag <- x < lo | x > hi
      flag[cell_idx[cell_flag]] <- TRUE
      if (verbose && any(cell_flag)) {
        cat(sprintf("  Cell %s, var %s: Q1=%.2f Q3=%.2f IQR=%.2f bounds=[%.2f, %.2f] -> %d flagged\n",
                    g, v, q[1], q[2], iqr, lo, hi, sum(cell_flag)))
      }
    }
  }
  if (verbose) cat(sprintf("\nIQR step: flagged %d; N: %d -> %d\n",
                           sum(flag), nrow(d), sum(!flag)))
  d[!flag, , drop = FALSE]
}

remove_outliers_by_cooksD <- function(data, formula, threshold_mult = 4,
                                      verbose = TRUE) {
  d <- data[complete.cases(data[, all.vars(formula)]), ]
  fit <- lm(formula, data = d)
  cooks <- cooks.distance(fit)
  threshold <- threshold_mult / nrow(d)
  flag <- cooks > threshold
  if (verbose) cat(sprintf("\nCook's D step: threshold = %d/n = %.4f; flagged %d; N: %d -> %d\n",
                           threshold_mult, threshold, sum(flag, na.rm = TRUE),
                           nrow(d), sum(!flag, na.rm = TRUE)))
  d[!flag, , drop = FALSE]
}

# ---- Reliability helpers ----
compute_reliability <- function(fit, factor_name) {
  pe <- parameterEstimates(fit, standardized = TRUE)
  loadings <- pe %>%
    dplyr::filter(op == "=~", lhs == factor_name) %>%
    dplyr::pull(std.all)
  cr  <- (sum(loadings))^2 / ((sum(loadings))^2 + sum(1 - loadings^2))
  ave <- mean(loadings^2)
  list(loadings = loadings, CR = cr, AVE = ave)
}

cronbach_alpha <- function(data, items) {
  d <- data[, items]
  d <- d[complete.cases(d), ]
  k <- ncol(d)
  var_total <- var(rowSums(d))
  var_items <- sum(apply(d, 2, var))
  (k / (k - 1)) * (1 - var_items / var_total)
}

# ---- Outlier removal pipeline ----
df_clean <- remove_outliers_by_cell(
  data = df,
  vars = "Slider_Difference",
  group_vars = c("Warmth", "Personalization"),
  iqr_mult = 1.5
)

ols_formula <- as.formula(
  "Slider_Difference ~ factor(Warmth) * factor(Personalization) +
   Emotional_Trust + Integrity_Trust + Digital_Literacy + UF_Q1 +
   P_Extraversion + P_Agreeableness + P_Openness + P_Conscientiousness + P_Neuroticism"
)

df_clean <- remove_outliers_by_cooksD(
  data = df_clean,
  formula = ols_formula,
  threshold_mult = 4
)

# ---- CFA validation: justify Option A by comparing 1, 2, 3-factor models ----
cfa_one_factor <- '
  Trust =~ ET_1 + ET_2 + ET_3 + CTC_1 + CTC_2 + CTI_1 + CTI_2 + CTI_3
'
cfa_two_factor <- '
  ET =~ ET_1 + ET_2 + ET_3
  CT =~ CTC_1 + CTC_2 + CTI_1 + CTI_2 + CTI_3
'
cfa_three_factor <- '
  ET  =~ ET_1 + ET_2 + ET_3
  CTC =~ CTC_1 + CTC_2
  CTI =~ CTI_1 + CTI_2 + CTI_3
'

fit_cfa1 <- cfa(cfa_one_factor,   data = df_clean, estimator = "ML", std.lv = TRUE)
fit_cfa2 <- cfa(cfa_two_factor,   data = df_clean, estimator = "ML", std.lv = TRUE)
fit_cfa3 <- cfa(cfa_three_factor, data = df_clean, estimator = "ML", std.lv = TRUE)

cat("\n=== CFA model comparison ===\n")
fit_compare <- rbind(
  `One-factor`   = fitMeasures(fit_cfa1, c("chisq","df","cfi","tli","rmsea","srmr","aic","bic")),
  `Two-factor`   = fitMeasures(fit_cfa2, c("chisq","df","cfi","tli","rmsea","srmr","aic","bic")),
  `Three-factor` = fitMeasures(fit_cfa3, c("chisq","df","cfi","tli","rmsea","srmr","aic","bic"))
)
print(round(fit_compare, 3))

cat("\n=== Chi-square difference test (1 vs 2 vs 3 factors) ===\n")
print(anova(fit_cfa1, fit_cfa2, fit_cfa3))

# Reliability and discriminant validity for the three-factor model
et_rel  <- compute_reliability(fit_cfa3, "ET")
ctc_rel <- compute_reliability(fit_cfa3, "CTC")
cti_rel <- compute_reliability(fit_cfa3, "CTI")

cat("\n=== Reliability (three-factor model) ===\n")
cat(sprintf("Emotional Trust (ET):  CR = %.3f  AVE = %.3f  alpha = %.3f\n",
            et_rel$CR, et_rel$AVE,
            cronbach_alpha(df_clean, et_cols)))
cat(sprintf("Competence (CTC):      CR = %.3f  AVE = %.3f  alpha = %.3f\n",
            ctc_rel$CR, ctc_rel$AVE,
            cronbach_alpha(df_clean, ctc_cols)))
cat(sprintf("Integrity (CTI):       CR = %.3f  AVE = %.3f  alpha = %.3f\n",
            cti_rel$CR, cti_rel$AVE,
            cronbach_alpha(df_clean, cti_cols)))

psi <- lavInspect(fit_cfa3, "std")$psi
cat("\n=== Factor correlations (three-factor model) ===\n")
print(round(psi, 3))

cat("\n=== Fornell-Larcker (sqrt(AVE) on diagonal) ===\n")
fl <- psi
diag(fl) <- c(sqrt(et_rel$AVE), sqrt(ctc_rel$AVE), sqrt(cti_rel$AVE))
print(round(fl, 3))
cat("Discriminant validity passes if diagonal > corresponding off-diagonals.\n")

# ---- PROCESS Model 4 with two latent mediators (ET + CTI) ----
# CTC dropped from the structural model due to two-indicator under-identification
# and high collinearity with ET and CTI in preliminary runs.

run_process_model4_2med_latent <- function(data, ivs, dv, covariates,
                                           n_boot = 5000, seed = 42) {
  needed <- c(ivs, dv, covariates,
              "ET_1","ET_2","ET_3",
              "CTI_1","CTI_2","CTI_3")
  d <- data[complete.cases(data[, needed]), needed]
  d[] <- lapply(d, as.numeric)
  
  cov_str <- if (length(covariates)) paste("+", paste(covariates, collapse = " + ")) else ""
  
  # a paths for ET and CTI
  a_paths <- paste(
    sapply(c("ET","CTI"), function(m) {
      mi <- match(m, c("ET","CTI"))
      iv_terms <- paste(
        sapply(seq_along(ivs), function(ii) sprintf("a%d_%d*%s", mi, ii, ivs[ii])),
        collapse = " + "
      )
      sprintf("%s ~ %s %s", m, iv_terms, cov_str)
    }),
    collapse = "\n    "
  )
  
  # DV equation
  cp_terms <- paste(
    sapply(seq_along(ivs), function(ii) sprintf("cp%d*%s", ii, ivs[ii])),
    collapse = " + "
  )
  dv_eq <- sprintf("%s ~ %s + b1*ET + b2*CTI %s", dv, cp_terms, cov_str)
  
  # Indirect / total effects
  ind_defs <- c()
  for (ii in seq_along(ivs)) {
    per_med <- c(
      sprintf("ind_%s_via_ET  := a1_%d*b1", ivs[ii], ii),
      sprintf("ind_%s_via_CTI := a2_%d*b2", ivs[ii], ii)
    )
    total_via_iv <- sprintf("total_ind_%s := a1_%d*b1 + a2_%d*b2",
                            ivs[ii], ii, ii)
    total_iv <- sprintf("total_%s := cp%d + total_ind_%s", ivs[ii], ii, ivs[ii])
    ind_defs <- c(ind_defs, per_med, total_via_iv, total_iv)
  }
  ind_block <- paste(ind_defs, collapse = "\n    ")
  
  model <- sprintf("
    # Measurement
    ET  =~ ET_1 + ET_2 + ET_3
    CTI =~ CTI_1 + CTI_2 + CTI_3

    # a paths
    %s

    # b paths + direct
    %s

    # Defined effects
    %s
  ", a_paths, dv_eq, ind_block)
  
  set.seed(seed)
  fit <- sem(model, data = d, se = "bootstrap", bootstrap = n_boot,
             estimator = "ML", std.lv = TRUE, fixed.x = FALSE)
  
  cat(sprintf("\n%s\nPROCESS Model 4 (parallel, 2 latent mediators: ET + CTI)\nIVs: %s | DV: %s\nCovariates: %s | N = %d | Bootstraps = %d\n%s\n",
              strrep("=", 70),
              paste(ivs, collapse = ", "),
              dv,
              paste(covariates, collapse = ", "),
              nrow(d), n_boot, strrep("=", 70)))
  print(summary(fit, standardized = TRUE, ci = TRUE, rsquare = TRUE,
                fit.measures = TRUE))
  
  cat("\n--- Defined effects: bootstrap percentile CIs ---\n")
  pe <- parameterEstimates(fit, boot.ci.type = "perc", level = 0.95,
                           standardized = TRUE) %>%
    dplyr::filter(op == ":=") %>%
    dplyr::select(label, est, se, z, pvalue, ci.lower, ci.upper,
                  std.all)
  print(pe, row.names = FALSE)
  
  cat("\n--- Standardized regression paths ---\n")
  reg <- parameterEstimates(fit, standardized = TRUE) %>%
    dplyr::filter(op == "~", label != "") %>%
    dplyr::select(lhs, op, rhs, label, est, se, pvalue, std.all)
  print(reg, row.names = FALSE)
  
  invisible(fit)
}

# ---- Run final mediation model ----
covariates <- c("Digital_Literacy","P_Extraversion","P_Agreeableness",
                "P_Openness","P_Conscientiousness","P_Neuroticism","UF_Q1")

fit_mediation <- run_process_model4_2med_latent(
  data = df_clean,
  ivs  = c("Warmth", "Personalization", "W_x_P"),
  dv   = "Slider_Difference",
  covariates = covariates,
  n_boot = 5000, seed = 42
)
