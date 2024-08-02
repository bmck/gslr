module GSLR
  class Model
    attr_reader :coefficients, :intercept, :fit_intercept

    def self.train_test_split(df, training_frac: 0.7, time_series: false, seed: nil)
      rng = Random.new(seed.nil? ? rand : seed)

      df['orig_order'] = Polars::Series.new((0..(df.length-1)).to_a)
      df['train_data'] = Polars::Series.new(Array.new(df.length) { 
          time_series ? df['orig_order'].to_f/df.length.to_f  : rng.rand() 
      })
      df.sort!('train_data')
      # Rails.logger.info { "#{__FILE__}:#{__LINE__} df = #{df.inspect}"}
      thresh = training_frac >= 0.5 ? 
          (training_frac * df.length).ceil : 
          (training_frac * df.length).floor
      df['train_data'] = Polars::Series.new(([true] * thresh) + ([false] * (df.length - thresh)))
      df.sort!('orig_order')
      # Rails.logger.info { "#{__FILE__}:#{__LINE__} df = #{df.inspect}"}
      df.drop_in_place('orig_order')
      df
    end

    def initialize(intercept: true)
      @fit_intercept = intercept
    end

    def predict(x)
      if numo?(x)
        x.dot(@coefficients) + @intercept
      else
        x.map do |xi|
          xi.zip(@coefficients).map { |xii, c| xii * c }.sum + @intercept
        end
      end
    end

    private

    def set_matrix(x, intercept:)
      s1, s2 = shape(x)
      s2 += 1 if intercept

      xc = FFI.gsl_matrix_alloc(s1, s2)
      x_ptr = FFI.gsl_matrix_ptr(xc, 0, 0)

      if numo?(x)
        if intercept
          ones = Numo::DFloat.ones(s1, 1)
          x = ones.concatenate(x, axis: 1)
        end
        set_data(x_ptr, x)
      else
        # pack efficiently
        str = String.new
        one = [1].pack("d*")
        x.each do |xi|
          str << one if intercept
          xi.pack("d*", buffer: str)
        end
        x_ptr[0, str.bytesize] = str
      end

      [xc, s1, s2]
    end

    def set_vector(x)
      v = FFI.gsl_vector_alloc(x.size)
      ptr = FFI.gsl_vector_ptr(v, 0)
      set_data(ptr, x)
      v
    end

    def set_data(ptr, x)
      if numo?(x)
        x = dfloat(x)
        ptr[0, x.byte_size] = x.to_string
      else
        str = x.pack("d*")
        ptr[0, str.bytesize] = str
      end
    end

    def shape(x)
      numo?(x) ? x.shape : [x.size, x.first.size]
    end

    def numo?(x)
      defined?(Numo::NArray) && x.is_a?(Numo::NArray)
    end

    def dfloat(x)
      x.is_a?(Numo::DFloat) ? x : x.cast_to(Numo::DFloat)
    end

    def check_status(status)
      raise Error, FFI.gsl_strerror(status).to_s if status != 0
    end
  end
end
