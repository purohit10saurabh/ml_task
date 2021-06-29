#include <ATen/ATen.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <stdint.h>
#include <bits/stdc++.h>
//#include <opencv2/opencv.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "im_stb/stb_image.h"
//using namespace std;

// Loads a tensor from an image file
torch::Tensor load_image(const std::string& file_path)
{ 
	int width = 0;
	int height = 0;
	int depth = 0;

	std::unique_ptr<unsigned char, decltype(&stbi_image_free)> image_raw(stbi_load(file_path.c_str(),
			&width, &height, &depth, 0), &stbi_image_free);

	if (!image_raw) {
			throw std::runtime_error("Unable to load image file " + file_path + ".");
	}

	return torch::from_blob(image_raw.get(),
			{width, height, depth}, torch::kUInt8).clone().to(torch::kFloat32).permute({2, 1, 0});
}


int main(int argc, const char* argv[]) { 
	torch::NoGradGuard no_grad;

	torch::jit::script::Module module;
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		module = torch::jit::load("../../traced_model.pt");
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}

	std::cout << "ok now\n";
	
	std::vector<int> indices;

	std::ifstream inp;
	inp.open("../../../data/test_dps.txt");  
	
	if(!inp)
		std::cout << "no dps file\n";
			 
	int value;
	// read the elements in the file into a vector  
	while (inp >> value)
		indices.push_back(value);
	
	inp.close();
	//indices.resize(1000);
	std::cout << "len is "<< indices.size() << std::endl;

	std::ifstream inp2;
	inp2.open("../../../data/test_actions.txt");
 
	int iter = 0;
	int sum = 0;
	for(auto& ind : indices)
	{
		torch::NoGradGuard no_grad;
		std::vector<torch::jit::IValue> inputs;

		std::string file_path = "../../../data/recorded_images/" + std::to_string(ind) + ".png";
		auto res = torch::data::transforms::Normalize<>(32.3, 56.2)(load_image(file_path));
		//std::cout << res << std::endl;
		//torch::data::transforms::Stack<>()  
		auto dp = res.unsqueeze(0);
 		// std::cout << dp << std::endl;

		inputs.push_back(dp);	
		// Execute the model and turn its output into a tensor.
	 	at::Tensor output = module.forward(inputs).toTensor();
		int lab = output.argmax(1).item().toInt();
		// pred_labels.push_back(lab);
		int val;
		inp2 >> val;
		if(lab==val)
			 sum++;
		iter++;
		if(iter%1000==0)
			std::cout << iter << " iter done" << std::endl;
		//std::cout << sum << std::endl;  
	} 
 	inp2.close();

 	std::cout << "Test accruacy is " << 100*(float)sum/indices.size() << std::endl;
}
