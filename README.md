# GSLR

:fire: High performance linear regression for Ruby, powered by [GSL](https://www.gnu.org/software/gsl/)

[![Build Status](https://github.com/bmck/gslr/actions/workflows/build.yml/badge.svg)](https://github.com/bmck/gslr/actions)

## Baseline

This tree is [ankane/gslr](https://github.com/ankane/gslr) **0.4.0**, commit [`925fad5`](https://github.com/ankane/gslr/commit/925fad50be43409da5aae97f166d1c64b3dc037c) (April 9, 2026). That commit is the baseline for the local changes. Changelog entries through 0.4.0 are upstream releases, including the 0.3.2 memory-free pattern and the Ruby `>= 3.3` requirement.

`GSLR::OLS`, `GSLR::Ridge`, `fit`, `predict`, `coefficients`, `intercept`, `covariance`, and `chi2` match that baseline.

Local additions on top of 0.4.0:

- OLS summary readers `r2`, `adjusted_r2`, `var_p_values`, and `exp_p_value`, plus `to_formatted_s`
- `GSLR::OLS.lm` for a Polars frame and a `y ~ x1 + x2` formula
- `GSLR::Model.train_test_split`
- `fit_intercept` reader
- GSL bindings `gsl_cdf_tdist_P` and `gsl_cdf_fdist_P`

## Installation

[Install GSL](#gsl-installation). For Homebrew, use:

```sh
brew install gsl
```

Add this line to your application’s Gemfile:

```ruby
gem "gslr"
```

## Getting Started

### Ordinary Least Squares

Prep your data

```ruby
x = [[1, 3], [2, 3], [3, 5], [4, 5]]
y = [10, 11, 16, 17]
```

Train a model

```ruby
model = GSLR::OLS.new
model.fit(x, y)
```

Make predictions

```ruby
model.predict(x)
```

Get the coefficients and intercept

```ruby
model.coefficients
model.intercept
```

Get the covariance matrix and chi-squared

```ruby
model.covariance
model.chi2
```

Pass weights

```ruby
weight = [1, 2, 3, 4]
model.fit(x, y, weight: weight)
```

Disable the intercept

```ruby
GSLR::OLS.new(intercept: false)
```

Read the summary written by `fit`

```ruby
model.r2
model.adjusted_r2
model.var_p_values
model.exp_p_value
model.fit_intercept
model.to_formatted_s
```

`r2` is `1 - chi2 / SST`, where SST is the sum of squared deviations of `y` from its mean. `adjusted_r2` is `1 - (n - 1) / (n - 2) * (1 - r2)`. `var_p_values` holds two-sided t p-values with `n - 2` degrees of freedom: the intercept first when it was fit, then one entry per coefficient. `exp_p_value` is the upper-tail p-value of `r2 * (n - 2) / (1 - r2)`, tested as F on 1 and `n - 2` degrees of freedom. `n` is `x.length` (the row count for an Array, the element count for a Numo matrix). `to_formatted_s` prints the coefficient table and returns `nil`.

Pass column names for that table with `indep_vars:`. Omitted names are labeled `x0`, `x1`, and so on. `dep_var:` is accepted and unused.

```ruby
model.fit(x, y, indep_vars: ["x1", "x2"])
```

Fit a Polars frame from a formula. Terms are split on `~` and `+`, and rows with nulls in those columns are dropped. Load the `polars` gem yourself; this gem does not depend on it.

```ruby
model = GSLR::OLS.lm(df, "y ~ x1 + x2")
model = GSLR::OLS.lm(df, "y ~ x1 + x2", intercept: false)
```

Split a Polars frame. The method sorts the frame in place, writes a boolean `train_data` column, and returns that frame. `training_frac` defaults to `0.7`. The cutoff uses `ceil` when the fraction is at least 0.5, and `floor` otherwise. `seed` must be an integer. The default `seed: nil` passes a Float to `Random.new` and raises `TypeError`.

```ruby
GSLR::Model.train_test_split(df, training_frac: 0.7, seed: 1)
```

### Ridge

Train a model

```ruby
model = GSLR::Ridge.new
model.fit(x, y)
```

Set the regularization strength

```ruby
GSLR::Ridge.new(alpha: 1.0)
```

**Note:** Weights aren’t supported yet

## GSL Installation

### Mac

```sh
brew install gsl
```

### Windows

Check out [the options](https://www.gnu.org/software/gsl/extras/native_win_builds.html).

### Ubuntu

```sh
sudo apt install libgsl-dev
```

### Heroku

Use the [Apt buildpack](https://github.com/heroku/heroku-buildpack-apt) and create an `Aptfile` with:

```text
libgsl-dev
```

## History

View the [changelog](CHANGELOG.md). Entries through 0.4.0 are the ankane/gslr baseline.

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/gslr/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/gslr/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/gslr.git
cd gslr
bundle install
bundle exec rake test
```
