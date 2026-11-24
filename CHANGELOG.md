## Unreleased

Baseline: [ankane/gslr 0.4.0](https://github.com/ankane/gslr/commit/925fad50be43409da5aae97f166d1c64b3dc037c) (`925fad5`, April 9, 2026). Entries below are that upstream history.

- Added OLS summary readers `r2`, `adjusted_r2`, `var_p_values`, and `exp_p_value`, and `to_formatted_s`
- Added `GSLR::OLS.lm` for a Polars frame and a `y ~ x1 + x2` formula
- Added `GSLR::Model.train_test_split` and the `fit_intercept` reader
- Added GSL CDF bindings `gsl_cdf_tdist_P` and `gsl_cdf_fdist_P`

## 0.4.0 (2026-04-07)

- Dropped support for Ruby < 3.3

## 0.3.2 (2025-05-04)

- Fixed memory leaks

## 0.3.1 (2024-12-29)

- Fixed GSL detection on Mac x86-64

## 0.3.0 (2024-10-22)

- Dropped support for Ruby < 3.1

## 0.2.1 (2023-06-07)

- Improved support for Mac ARM

## 0.2.0 (2022-09-02)

- Fixed error on Mac
- Dropped support for Ruby < 2.7

## 0.1.3 (2020-07-26)

- Improved performance
- Improved error message when GSL not found on Mac

## 0.1.2 (2019-12-08)

- Added `covariance` and `chi2` methods

## 0.1.1 (2019-12-07)

- Fixed `undefined symbol` error on Linux

## 0.1.0 (2019-12-07)

- First release
