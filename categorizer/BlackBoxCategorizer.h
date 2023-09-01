/*
MIT License
Copyright (c) 2023 Philip Arturo Smith
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef CATEGORIZER_BLACKBOXCATEGORIZER_H
#define CATEGORIZER_BLACKBOXCATEGORIZER_H

#include <boost/numeric/ublas/tensor.hpp>
#include <categorizer/TensorTransformer.h>
#include <cstdlib>

namespace categorizer {

template<typename Precision,
		TensorTransformer<Precision> TransformerType>
class BlackBoxCategorizer {
public:
	void extract_features(TransformerType transformer, std::size_t iterations) {
		// extract neural net features
		using namespace boost::numeric::ublas;

		for(auto i = 0; i < iterations; ++i) {
			std::cout << "generating noise..." << std::endl;
			auto noise = generate_noise(transformer);
			std::cout << "generated input noise: " << std::endl << noise << std::endl;
			tensor<Precision> result{transformer.get_output_shape()};
			transformer(noise, result);
			std::cout << "output: " << std::endl << result << std::endl;
		}
	}
private:
	boost::numeric::ublas::tensor<Precision> generate_noise(TransformerType transformer, std::size_t seed = 0) {
		using namespace boost::numeric::ublas;
		std::srand(seed);

		tensor<Precision> noise{transformer.get_input_shape()};
		for(auto i = noise.begin(); i != noise.end(); ++i)
			*i = (Precision)(std::rand());
		return noise;
	}
};

} // namespace categorizer


#endif //CATEGORIZER_BLACKBOXCATEGORIZER_H
