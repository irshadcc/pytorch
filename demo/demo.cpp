#include <iostream>
#include "ATen/core/operator_name.h"

#include <ATen/core/dispatch/OperatorEntry.h>
#include <ATen/ops/embedding.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/ones.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <ATen/Tensor.h>
#include <c10/core/TensorImpl.h>
#include <autograd/function.h>

template <typename T>
void print(const T &value) {
	std::cout << value << "\n"; 
}


/**
 * Create a 2-D Tensor with shape (v,d).
 */
at::Tensor create_weight(int v, int d) {
	long numel = v*d;
	float* data = new float[numel];
	for (long i = 0 ; i < numel; i++) {
		data[i] = static_cast<float>(i); 
	}
	auto options = at::TensorOptions()
		.requires_grad(true)
		.dtype(c10::ScalarType::Float);

	auto tensor = at::from_blob(data, {v,d}, options);
	return tensor;
	// auto tensor = at::ones({v,d}, options);
	// return tensor;
}

at::Tensor create_one_hot(int v, int index) {
	int n = 1;
	int* data = new int[v];
	for (int i = 0 ; i < v; i++) {
		data[i] = 0;
	}
	data[index] = 1;

	auto options = at::TensorOptions()
		.requires_grad(true)
		.dtype(c10::ScalarType::Long);
	return at::from_blob(data, {1, n}, options);
}


at::Tensor create_vector(int d) {
	float* data = new float[d];
	for (int i = 0 ; i < d; i++) {
		data[i] = static_cast<float>(i);
	}

	auto options = at::TensorOptions()
		.requires_grad(true)
		.dtype(c10::ScalarType::Float);
	auto tensor = at::from_blob(data, {1, d}, options);
	tensor.set_requires_grad(true);
	return tensor;
}

int main() {

	auto first = create_vector(10);
	print(first);
	auto second = create_vector(10);
	print(second);
	auto result = first.add(second);
	print(result);

	// if (result.grad_fn()) {
	// 	auto &grad_fn = result.grad_fn();
	// 	print(grad_fn->name());
	// 	print(grad_fn->num_inputs());
	// 	print(grad_fn->num_outputs());

	// } 
	// print(first);
	// print(second);
	// print(result);


	// std::cout << "Embedding shape : " << embedding.sizes();
	// std::cout << "Embedding at 0 : " << embedding.toString() << "\n";
	// auto loss = embedding.sum() ;
	// std::cout << "Loss " << loss << "\n";
	// loss.backward();
	// std::cout << "Weight grad " << weight.grad() << "\n";
	return 0;
}
