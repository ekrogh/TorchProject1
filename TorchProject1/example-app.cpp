#include <torch/torch.h>
#include <cuda_runtime.h>
#include <iostream>

int main()
{
	torch::Tensor tensor = torch::rand({ 2, 3 });
	if (torch::cuda::is_available())
	{
		std::cout << "CUDA is available! Training on GPU" << std::endl;
		auto tensor_cuda = tensor.cuda();
		std::cout << tensor_cuda << std::endl;

		// Check the number of GPUs available
		int num_gpus = torch::cuda::device_count();
		std::cout << "Number of GPUs available: " << num_gpus << std::endl;

		int deviceCount = 0;
		cudaGetDeviceCount(&deviceCount);

		for (int device = 0; device < deviceCount; ++device)
		{
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, device);

			std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
			std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
			std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
		}

		// Enable automatic mixed precision
		//auto scaler = torch::cuda::GradScaler();
		auto scaler = torch::cuda::amp::GradScaler();

		// Assuming you have a model, optimizer, and loss function defined
		// Example forward pass in mixed precision
		{
			torch::cuda::amp::AutocastMode autocast_enabled;
			torch::cuda::amp::AutocastMode autocast_enabled;
			// auto output = model.forward(tensor_cuda); // Example forward pass
			// auto loss = loss_function(output, labels); // Compute loss

			// scaler.scale(loss).backward(); // Scale loss for mixed precision
			// scaler.step(optimizer); // Optimizer step
			// scaler.update(); // Update the scale for next iteration
		}
	}
	else
	{
		std::cout << "CUDA is not available! Training on CPU" << std::endl;
		std::cout << tensor << std::endl;
	}

	std::cin.get();
}

