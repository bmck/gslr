module GSLR
  class OLS < Model
    attr_reader :covariance, :chi2

    def self.lm(df, str, intercept: true)
      dep_var = str.split('~').first.strip
      indep_vars = str.split('~').second.split('+').map(&:strip)

      df2 = df.drop_nulls(subset: indep_vars + [dep_var])

      y = df2[dep_var].to_a
      x = indep_vars.map{|c| df2[c].to_a}.transpose

      model = GSLR::OLS.new(intercept: intercept)
      model.fit(x, y)
      model
    end

    def fit(x, y, weight: nil, dep_var: nil, indep_vars: nil)
      # set data
      xc, s1, s2 = set_matrix(x, intercept: @fit_intercept)
      yc = set_vector(y)

      # allocate solution
      c = FFI.gsl_vector_alloc(s2)
      cov = FFI.gsl_matrix_alloc(s2, s2)
      chisq = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE)
      work = FFI.gsl_multifit_linear_alloc(s1, s2)

      # fit
      if weight
        wc = set_vector(weight)
        check_status FFI.gsl_multifit_wlinear(xc, wc, yc, c, cov, chisq, work)
      else
        check_status FFI.gsl_multifit_linear(xc, yc, c, cov, chisq, work)
      end

      # read solution
      c_ptr = FFI.gsl_vector_ptr(c, 0)
      @coefficients = c_ptr[0, s2 * Fiddle::SIZEOF_DOUBLE].unpack("d*")
      @intercept = @fit_intercept ? @coefficients.shift : 0.0
      @covariance = read_matrix(cov, s2)
      @chi2 = chisq[0, Fiddle::SIZEOF_DOUBLE].unpack1("d")


      # generate formatted output
      # Taken from https://stackoverflow.com/questions/5503733/getting-p-value-for-linear-regression-in-c-gsl-fit-linear-function-from-gsl-li
      n = x.length
      @formatted_output = "Coefficients\tEstimate\tStd. Error\tt value\tPr(>|t|)\n"
      sd = Math.sqrt(covariance[0][0])
      t = c[0].to_f / sd.to_f
      # The following is the p-value of the constant term
      pv = t<0 ? 2.0*(1.0-FFI.gsl_cdf_tdist_P(-t,n-2)) : 2.0*(1.0-FFI.gsl_cdf_tdist_P(t,n-2))
      @formatted_output += "Intercept\t#{c[0]}\t#{sd}\t#{t}\t#{pv}\n";

      (1..covariance.length-1).each do |i|
        sd = Math.sqrt(covariance[i][i])
        t = c[i].to_f / sd.to_f
        # ;//This is the p-value of the linear term
        pv = t<0 ? 2.0*(1.0-gsl_cdf_tdist_P(-t,n-2)) : 2.0*(1.0-gsl_cdf_tdist_P(t,n-2))
        @formatted_output += "#{indep_vars.is_a?(Array) ? indep_vars[i] : "x#{i}" }\t" \
            "#{c[i].to_f}\t#{sd}\t#{t}\t#{pv}\n";
      end

      dof = n-2
      y_mean = y.sum.to_f / y.length.to_f
      sct = (0..y.length-1).to_a.map{|i| (y[i] - y_mean)*(y[i] - y_mean) }.sum
      r2 = 1.0-chisq/sct
      @formatted_output += "Multiple R-squared: #{r2},    Adjusted R-squared: #{1-(n-1).to_f/dof.to_f*(1.0-r2)}\n"
      f = r2*dof/(1.0-r2);
      p_value = 1.0 - gsl_cdf_fdist_P(f,1,dof);
      @formatted_output += "F-statistic: #{f} on 1 and #{dof} DoF,  p-value: #{p_value}\n"

      nil
    ensure
      FFI.gsl_matrix_free(xc) if xc
      FFI.gsl_vector_free(yc) if yc
      FFI.gsl_vector_free(wc) if wc
      FFI.gsl_vector_free(c) if c
      FFI.gsl_matrix_free(cov) if cov
      FFI.gsl_multifit_linear_free(work) if work
    end

    def to_formatted_s
      @formatted_output
    end

    private

    def read_matrix(cov, s2)
      ptr = FFI.gsl_matrix_ptr(cov, 0, 0)
      row_size = s2 * Fiddle::SIZEOF_DOUBLE
      s2.times.map do |i|
        ptr[i * row_size, row_size].unpack("d*")
      end
    end
  end
end
