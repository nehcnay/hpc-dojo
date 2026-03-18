#include <torch/extension.h>

void square_cuda(torch::Tensor input, torch::Tensor output);

torch::Tensor square(torch::Tensor input) {
    auto output = torch::empty_like(input);

    square_cuda(input, output);

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square", &square, "Element-wise square (CUDA)");
}
