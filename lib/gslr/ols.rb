module GSLR
  class OLS < Model
    attr_reader :covariance, :chi2, :exp_p_value, :var_p_values

    def self.lm(df, str, intercept: true)
      dep_var = str.split('~').first.strip
      indep_vars = str.split('~').second.split('+').map(&:strip)

      df2 = df.drop_nulls(subset: indep_vars + [dep_var])

      y = df2[dep_var].to_a
      x = indep_vars.map{|c| df2[c].to_a}.transpose

      # Rails.logger.info { "#{__FILE__}:#{__LINE__} intercept = #{intercept}" }
      model = GSLR::OLS.new(intercept: intercept)
      # Rails.logger.info { "#{__FILE__}:#{__LINE__} fit_intercept = #{fit_intercept}" }
      model.fit(x, y, dep_var: dep_var, indep_vars: indep_vars)
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
      # Rails.logger.info { "#{__FILE__}:#{__LINE__} fit_intercept = #{fit_intercept}" }
      @intercept = @fit_intercept ? @coefficients.shift : 0.0
      @covariance = read_matrix(cov, s2)
      @chi2 = chisq[0, Fiddle::SIZEOF_DOUBLE].unpack1("d")
      @var_p_values = []


      # generate formatted output
      # Taken from https://stackoverflow.com/questions/5503733/getting-p-value-for-linear-regression-in-c-gsl-fit-linear-function-from-gsl-li
      n = x.length
      @formatted_output = "Coefficients \tEstimate \tStd. Error \tt value \tPr(>|t|)\n"
      if @fit_intercept == true
        sterr = Math.sqrt(@covariance[0][0])
        t = intercept.to_f / sterr.to_f
        # The following is the p-value of the constant term
        p_value = 2.0*(1.0-FFI.gsl_cdf_tdist_P(t.abs, n-2))
        @var_p_values << p_value
        @formatted_output += "Intercept \t#{intercept.round(9).to_s.ljust(10)} \t#{sterr.round(6).to_s.ljust(10)} \t#{t.round(6).to_s.ljust(10)} \t#{p_value.round(6)}\n";
      end

      offset = @fit_intercept == true ? 1 : 0

      (0..@coefficients.length-1).to_a.each do |i|
        sterr = Math.sqrt(@covariance[i+offset][i+offset])
        t = @coefficients[i].to_f / sterr.to_f
        # ;//This is the p-value of the linear term
        pv = 2.0*(1.0-FFI.gsl_cdf_tdist_P(t.abs, n-2))
        @var_p_values << pv
        @formatted_output += "#{(indep_vars.is_a?(Array) ? indep_vars[i].ljust(10) : "x#{i}\t") }\t" \
          "#{@coefficients[i].round(9).to_s.ljust(10)} \t#{sterr.round(6).to_s.ljust(10)} \t#{t.round(6).to_s.ljust(10)} \t#{pv}\n";
      end

      dof = n-2
      y_mean = y.sum.to_f / y.length.to_f
      sct = (0..y.length-1).to_a.map{|i| (y[i] - y_mean)*(y[i] - y_mean) }.sum
      r2 = 1.0- @chi2 / sct
      adj_r2 = 1-(n-1).to_f/dof.to_f*(1.0-r2)
      @formatted_output += "\nMultiple R-squared: #{r2},    Adjusted R-squared: #{adj_r2}\n"
      f = r2 * dof/(1.0 - r2);
      @exp_p_value = 1.0 - FFI.gsl_cdf_fdist_P(f,1,dof);
      @formatted_output += "F-statistic: #{f} on 1 and #{dof} DoF,  p-value: #{@exp_p_value.round(6)}\n"

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
      puts @formatted_output
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
