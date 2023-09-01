#include <iostream>
#include <categorizer/NeuralCategorizer.h>

class TestNet {
public:
	void operator()(const boost::numeric::ublas::tensor<float>& x, boost::numeric::ublas::tensor<float>& y) {
		y = x*x;
	}

	boost::numeric::ublas::shape get_input_shape() {
		return {2,2};
	}

	boost::numeric::ublas::shape get_output_shape() {
		return {2,2};
	}
};

using namespace categorizer;

int main() {
	TestNet test_net;
	NeuralCategorizer<float, TestNet> categorizer;
	categorizer.extract_features(test_net, 100);

	return 0;
}
