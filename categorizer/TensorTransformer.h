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

#ifndef CATEGORIZER_TENSORTRANSFORMER_H
#define CATEGORIZER_TENSORTRANSFORMER_H

#include <array>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include <concepts>
#include <cstddef>


namespace categorizer {

// Abstract template for any operator that transforms tensors to tensors
// maps input vectors to output vectors but not necessarily linearly hence Tensor "like"
// y = Tx
template<typename T, typename Precision>
concept TensorTransformer = requires(
		T transformer,
		const boost::numeric::ublas::tensor<Precision> &x,
		boost::numeric::ublas::tensor<Precision> &y)
{
	{ transformer(x, y) } -> std::convertible_to<void>;
	// useful for allocating input/output tensors
	{ transformer.get_input_shape() } -> std::convertible_to<boost::numeric::ublas::shape>;
	{ transformer.get_output_shape() } -> std::convertible_to<boost::numeric::ublas::shape>;
};

// In place tensor transformer
template<typename T, typename Precision>
concept InplaceTensorTransformer = requires(
		T transformer,
		boost::numeric::ublas::tensor<Precision> &x)
{
	{ transformer(x) } -> std::convertible_to<void>;
};

} // namespace categorizer

#endif //CATEGORIZER_TENSORTRANSFORMER_H
