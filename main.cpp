#include <SFML/Graphics.hpp>
#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <experimental/filesystem>
#include "ImageProcessor.h"
#include "inf_types.h"
#include "custom_net_imp.h"
#include "Windows.h"
#include <algorithm>
#include <random>


#define IMAGE_SIZE 100


inf::ImageProcessor imp;


// sf::Image to inf::image
inf::image toInfImage(sf::Image image)
{
	inf::image result;
	imp.copyBufferToImage((void*)image.getPixelsPtr(), image.getSize().x, image.getSize().y, inf::RGBA8, &result);
	return result;
}

// inf::Image to sf::image
sf::Image toSfImage(inf::image image)
{
	sf::Texture tex;
	tex.create(image.sizeX,image.sizeY);
	tex.update((sf::Uint8*)image.buffer);
	sf::Image result = tex.copyToImage();
	return result;
}


// used to load names under a certain path
void loadNames(std::string path, std::vector<std::string>& nameStack)
{
	for (const auto& entry : std::experimental::filesystem::directory_iterator(path))
	{
		std::string name = entry.path().string().substr(path.size() + 1, entry.path().string().size() - path.size() - 1);
		nameStack.push_back(name);
	}
}


// load images from the path into the given vector 
void loadImagesFromPath(std::string path, std::vector<sf::Image*>& images)
{ 
	std::vector<std::string> catFileNames;
	loadNames(path, catFileNames);

	// DEBUG
	int i = 0;
	int max = 1500;
	// DEBUG
	for (std::string x : catFileNames)
	{
		sf::Image* img = new sf::Image();
		std::string backString = "/";
		x = backString.append(x);
		img->loadFromFile(path.c_str() + x);
		images.push_back(img);

		// DEBUG
		i++;
		if (i >= max) { break; }
		// DEBUG
	}

}


// image to tensor
torch::Tensor toTensor(inf::image image, torch::DeviceType deviceType)
{

	const sf::Uint8* buff = (sf::Uint8*)image.buffer;
	int sizeX = image.sizeX;
	int sizeY = image.sizeY;
	int sizeTotalImage = sizeX * sizeY;
	int sizeTotalBuff = sizeX * sizeY * 4;
	int sizeTotalTensor = sizeX * sizeY * 3;



	torch::Tensor result = torch::empty({ 1, 3, sizeX, sizeY });

	// OLD AND SLOW
	/*
	torch::Tensor r = torch::empty({sizeX,sizeY});
	torch::Tensor g = torch::empty({ sizeX,sizeY });
	torch::Tensor b = torch::empty({ sizeX,sizeY });


	for (int x = 0; x < sizeX; x++)
	{
		torch::Tensor rx = torch::empty({ sizeY });
		torch::Tensor gx = torch::empty({ sizeY });
		torch::Tensor bx = torch::empty({ sizeY });
		for (int y = 0; y < sizeY; y++)
		{
			int buffIndex = x * sizeY * 4 + y * 4;
			rx[y] = buff[buffIndex];
			gx[y] = buff[buffIndex + 1];
			bx[y] = buff[buffIndex + 2];
		}
		r[x] = rx;
		g[x] = gx;
		b[x] = bx;
	}


	// old and slow method
	int tensorIndex = 0;
	for(int i = 0; i < sizeTotalBuff; i = i + 4)
	{
		int xIndex =  (i % (sizeX*4))/4;
		int yIndex =  i / (sizeX * 4);

		r[xIndex][yIndex] = buff[i];
		g[xIndex][yIndex] = buff[i+1];
		b[xIndex][yIndex] = buff[i+2];

	}

	result[0][0] = r;
	result[0][1] = g;
	result[0][2] = b;
	*/

	// REDUCE THE CHANNEL SIZE TO 3
	int oldBuffSize = image.sizeX * image.sizeY * 4;
	sf::Uint8* oldBuff = (sf::Uint8*)image.buffer;

	int newBuffSize = image.sizeX * image.sizeY * 3;
	int newBuffChannelSize = newBuffSize / 3;
	sf::Uint8* newBuff = (sf::Uint8*)malloc(newBuffSize);

	int a = 0;
	for (int i = 0; i < oldBuffSize; i++)
	{
		if (i % 4 == 3)
		{
			a++;
		}
		else
		{
			newBuff[(i % 4) * newBuffChannelSize + a] = oldBuff[i];
		}
	}
	auto tensor_image = torch::from_blob(newBuff, { 3 ,sizeX, sizeY }, at::kByte);
	
	tensor_image = tensor_image.permute({ 0,2,1 });
	result[0] = tensor_image;
	result = result.toType(torch::kFloat).div(255);
	//result.sub_(0.5).div_(0.5);


	free(newBuff);
	return result.to(deviceType);
}





inf::image toImage(torch::Tensor tensor)
{
	inf::image result;
	
	
	int sizeY = tensor[0][0][0].numel();
	int sizeX = tensor[0][0].numel()/ tensor[0][0][0].numel();

	torch::Tensor r = torch::ones({ sizeX,sizeY });
	torch::Tensor g = torch::ones({ sizeX,sizeY });
	torch::Tensor b = torch::ones({ sizeX,sizeY });

	r = tensor[0][0];
	g = tensor[0][1];
	b = tensor[0][2];

	result.sizeX = sizeX;
	result.sizeY = sizeY;
	result.channelCount = 4;
	result.format = inf::RGBA8;

	sf::Uint8* buff = (sf::Uint8*)malloc(sizeX*sizeY*result.channelCount*sizeof(sf::Uint8));
	result.buffer = (void*)buff;
	for (int y = 0; y < sizeY; y++)
	{
		for (int x = 0; x < sizeX; x++)
		{
			int buffIndex = y * sizeX * result.channelCount + x * result.channelCount;
			buff[buffIndex] = (sf::Uint8)(r[x][y].item<float>() * 255);
			buff[buffIndex + 1] = (sf::Uint8)(g[x][y].item<float>() * 255);
			buff[buffIndex + 2] = (sf::Uint8)(b[x][y].item<float>() * 255);
			buff[buffIndex + 3] = 255;
		}
	}
	return result;

}

// this adds image to the tensor for batchSize > 1
void addToTensor(inf::image image, torch::Tensor& tensor, int batchIndex)
{
	const sf::Uint8* buff = (sf::Uint8*)image.buffer;
	int sizeX = image.sizeX;
	int sizeY = image.sizeY;
	int sizeTotalImage = sizeX * sizeY;
	int sizeTotalBuff = sizeX * sizeY * 4;
	int sizeTotalTensor = sizeX * sizeY * 3;




	int tensorIndex = 0;
	for (int i = 0; i < sizeTotalBuff; i = i + 4)
	{
		int xIndex = tensorIndex % sizeX;
		int yIndex = tensorIndex / sizeY;

		tensor[batchIndex][0][xIndex][yIndex] = buff[i];
		tensor[batchIndex][1][xIndex][yIndex] = buff[i + 1];
		tensor[batchIndex][2][xIndex][yIndex] = buff[i + 2];
		tensorIndex++;
	}

}





/* PYTORCH TIP :
You can find the full list of available built-in modules like torch::nn::Linear, torch::nn::Dropout or torch::nn::Conv2d in the documentation of the torch::nn namespace.
*/

/* PYTORCH TIP :
The documentation for torch::nn::Module contains the full list of methods that operate on the module hierarchy.
*/

std::vector<sf::Image*> cactusImages;
std::vector<sf::Image*> caliImages;
std::vector<sf::Image*> lavantaImages;
std::vector<sf::Image*> minimixImages;
std::vector<sf::Image*> redImages;


void joinImages (
	std::vector<sf::Image*> cactusImages, 
	std::vector<sf::Image*>  caliImages, 
	std::vector<sf::Image*> lavantaImages, 
	std::vector<sf::Image*>  minimixImages, 
	std::vector<sf::Image*>  redImages,
	std::vector<std::pair<sf::Image*, int>>& result)
{

	for (auto x : cactusImages)
	{
		if (x->getSize().x > IMAGE_SIZE / 2  && x->getSize().y > IMAGE_SIZE / 2)
		{
			result.push_back(std::pair<sf::Image*, int>(x, 0));
		}
	}
	for (auto x : caliImages)
	{
		if (x->getSize().x > IMAGE_SIZE / 2 && x->getSize().y > IMAGE_SIZE / 2)
		{
			result.push_back(std::pair<sf::Image*, int>(x, 1));
		}
	}
	for (auto x : lavantaImages)
	{
		if (x->getSize().x > IMAGE_SIZE / 2  && x->getSize().y > IMAGE_SIZE / 2)
		{
			
			result.push_back(std::pair<sf::Image*, int>(x, 2));
		}
	}
	for (auto x : minimixImages)
	{
		if (x->getSize().x > IMAGE_SIZE / 2 && x->getSize().y > IMAGE_SIZE / 2)
		{
			result.push_back(std::pair<sf::Image*, int>(x, 3));
		}
	}
	for (auto x : redImages)
	{
		if (x->getSize().x > IMAGE_SIZE / 2 && x->getSize().y > IMAGE_SIZE / 2)
		{
			result.push_back(std::pair<sf::Image*, int>(x, 4));
		}
	}
	
	auto rng = std::default_random_engine{};
	std::shuffle(std::begin(result), std::end(result), rng);
}

// max used for output class counts
int max(std::vector<float> vec)
{
	int i = 0;
	for (float x : vec)
	{
		bool max = true;
		for (float y : vec)
		{
			if (y > x) { max = false; }
		}
		if (max) { return i;}
		i++;
	}

}

int main()
{
	sf::Font font;
	font.loadFromFile("font.ttf");
	sf::Text text;
	text.setFont(font);
	text.setCharacterSize(20);
	text.setOutlineColor(sf::Color::White);


	sf::RenderWindow* window = new sf::RenderWindow(sf::VideoMode(1920, 1080), "SFML window");

	loadImagesFromPath("B:/nuImagesDataset/data/plantdata/train/cactus",cactusImages);
	loadImagesFromPath("B:/nuImagesDataset/data/plantdata/train/cali", caliImages);
	loadImagesFromPath("B:/nuImagesDataset/data/plantdata/train/lavanta", lavantaImages);
	loadImagesFromPath("B:/nuImagesDataset/data/plantdata/train/minimix", minimixImages);
	loadImagesFromPath("B:/nuImagesDataset/data/plantdata/train/red", redImages);


	std::vector<std::pair<sf::Image*, int>> images;
	joinImages(cactusImages, caliImages, lavantaImages, minimixImages, redImages, images);



	std::cout << cactusImages.size() << std::endl;
	std::cout << caliImages.size() << std::endl;
	std::cout << lavantaImages.size() << std::endl;
	std::cout << minimixImages.size() << std::endl;
	std::cout << redImages.size() << std::endl;


	inf::image imageCactus = toInfImage(*cactusImages.at(15));
	inf::image imageCali = toInfImage(*caliImages.at(15));
	inf::image imageLavanta = toInfImage(*lavantaImages.at(15));
	inf::image imageMinimix = toInfImage(*minimixImages.at(15));
	inf::image imageRed = toInfImage(*redImages.at(15));

	


	int batchSize = 10;
	const double LearningRate = 5e-6;
	const int NumberOfEpochs = 20;

	
	
	torch::DeviceType device_type;
	if (torch::cuda::is_available() && 1) {
		device_type = torch::kCUDA;
		std::cout << "cuda is available" << std::endl;
	}
	else {
		device_type = torch::kCPU;
		std::cout << "cuda is not available" << std::endl;
	}
	torch::Device device(device_type);

	auto Net = std::make_shared<CustomNetImp>();
	//torch::load(Net,"86ac.pt");
	Net->to(device);

	torch::optim::Adam Optimizer(Net->parameters(), torch::optim::AdamOptions(LearningRate));


	//TARGETS
	torch::Tensor cactusTarget = torch::ones(5, device_type);
	cactusTarget[0] = 1.000;
	cactusTarget[1] = 0.000;
	cactusTarget[2] = 0.000;
	cactusTarget[3] = 0.000;
	cactusTarget[4] = 0.000;

	torch::Tensor caliTarget = torch::ones(5, device_type);
	caliTarget[0] = 0.000;
	caliTarget[1] = 1.000;
	caliTarget[2] = 0.000;
	caliTarget[3] = 0.000;
	caliTarget[4] = 0.000;

	torch::Tensor lavantaTarget = torch::ones(5, device_type);
	lavantaTarget[0] = 0.000;
	lavantaTarget[1] = 0.000;
	lavantaTarget[2] = 1.000;
	lavantaTarget[3] = 0.000;
	lavantaTarget[4] = 0.000;

	torch::Tensor minimixTarget = torch::ones(5, device_type);
	minimixTarget[0] = 0.000;
	minimixTarget[1] = 0.000;
	minimixTarget[2] = 0.000;
	minimixTarget[3] = 1.000;
	minimixTarget[4] = 0.000;

	torch::Tensor redTarget = torch::ones(5, device_type);
	redTarget[0] = 0.000;
	redTarget[1] = 0.000;
	redTarget[2] = 0.000;
	redTarget[3] = 0.000;
	redTarget[4] = 1.000;
	//TARGETS

	int cactusCount = 0;
	int caliCount = 0;
	int lavantaCount = 0;
	int minimixCount = 0;
	int redCount = 0;

	sf::Clock batchClock;
	
	inf::image infImage = imp.bicubicResize(toInfImage(*images.at(0).first), IMAGE_SIZE, IMAGE_SIZE);
	torch::Tensor input = toTensor(infImage, device_type);
	std::vector<int> lastPreds;
	float lastPredAccuracy{};
	for (int Epoch = 0; Epoch < NumberOfEpochs; ++Epoch)
	{
		std::vector<torch::Tensor> batchLoss;
		for (size_t i = 0; i < images.size(); i++) {
			
			int batchIndex = i % batchSize;
			
			inf::image infImage = imp.bicubicResize(toInfImage(*images.at(i).first), IMAGE_SIZE, IMAGE_SIZE);

			torch::Tensor input = toTensor(infImage, device_type);
			
			torch::Tensor output = Net->forward(input);
			
			torch::Tensor target;
			if (images.at(i).second == 0) { target = cactusTarget; }
			else if (images.at(i).second == 1) { target = caliTarget; }
			else if (images.at(i).second == 2) { target = lavantaTarget; }
			else if (images.at(i).second == 3) { target = minimixTarget; }
			else if (images.at(i).second == 4) { target = redTarget; }

			target.to(torch::kFloat);
			torch::Tensor loss = torch::l1_loss(output, target);
			
			Optimizer.zero_grad();
			batchLoss.push_back(loss);
			

	
			if (batchIndex == batchSize-1)
			{
				float lossTotal = 0;
				float lossAvg = 0;
				for (auto& x : batchLoss)
				{
					lossTotal = lossTotal + x.item<float>();
					x.backward();
				}
				lossAvg = lossTotal / batchSize;
				torch::Tensor realLoss = torch::ones(1);
				std::cout << "avgLoss " << lossAvg << " : " << std::endl;
				Optimizer.step();
				Optimizer.zero_grad();
			
				batchLoss.clear();

				std::cout << "batch time " << batchClock.getElapsedTime().asSeconds() << std::endl;
				batchClock.restart();

				float total = 0;
				for (int x : lastPreds)
				{
					total = total + x;
				}
				lastPredAccuracy = total;
			}

			if (lastPreds.size() > 100)
			{
				lastPreds.erase(lastPreds.begin());
			}

			
			std::vector<float> outputVector;
			for (int i = 0; i < 5; i++)
			{
				outputVector.push_back(output[i].item<float>());
			}
			int maxIndex = max(outputVector);
			// TODO(yigit) : optimize this for 5 classes
			if (maxIndex == 0) 
			{
				if (images.at(i).second == 0) { lastPreds.push_back(1); }
				else { lastPreds.push_back(0); }
				cactusCount++;
			}
			else if (maxIndex == 1) {
				if (images.at(i).second == 1) { lastPreds.push_back(1); }
				else { lastPreds.push_back(0); }
				caliCount++;
			}
			else if (maxIndex == 2) {
				if (images.at(i).second == 2) { lastPreds.push_back(1); }
				else { lastPreds.push_back(0); }
				lavantaCount++;
			}
			else if (maxIndex == 3) {
				if (images.at(i).second == 3) { lastPreds.push_back(1); }
				else { lastPreds.push_back(0); }
				minimixCount++;
			}
			else if (maxIndex == 4) {
				if (images.at(i).second == 4) { lastPreds.push_back(1); }
				else { lastPreds.push_back(0); }
				redCount++;
			}



			//SFML
			
			sf::Event event;
			while (window->pollEvent(event))
			{

				if (event.type == sf::Event::Closed)
					window->close();
			}

			while (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
			{
				Sleep(50);
			}

			sf::Texture tex;
			tex.loadFromImage(toSfImage(infImage));
			free(infImage.buffer);
			//
			sf::Texture tex2;
			sf::Sprite testSprite;
			//tex2.loadFromImage(toSfImage(toImage(input)));
			testSprite.setPosition(0,500);
			testSprite.setTexture(tex2);
			//


			window->clear();
			text.setPosition(250,60);
			text.setString("epoch: " + std::to_string(Epoch));
			window->draw(text);

			text.setPosition(250, 80);
			text.setString("batchIndex: " + std::to_string(i) + "/" + std::to_string(images.size()));
			window->draw(text);

			// PERCENTAGES
			text.setPosition(500, 60);
			text.setString("cactusProb:  %" + std::to_string(output[0].item<float>()*100));
			window->draw(text);

			text.setPosition(500, 80);
			text.setString("caliProb:  %" + std::to_string(output[1].item<float>()*100));
			window->draw(text);

			text.setPosition(500, 100);
			text.setString("lavantaProb:  %" + std::to_string(output[2].item<float>() * 100));
			window->draw(text);

			text.setPosition(500, 120);
			text.setString("minimixProb:  %" + std::to_string(output[3].item<float>() * 100));
			window->draw(text);

			text.setPosition(500, 140);
			text.setString("redProb:  %" + std::to_string(output[4].item<float>() * 100));
			window->draw(text);
			// PERCENTAGES



			// COUNTERS
			text.setPosition(750, 60);
			text.setString("cactusCount:  " + std::to_string(cactusCount));
			window->draw(text);

			text.setPosition(750, 80);
			text.setString("caliCount: " + std::to_string(caliCount));
			window->draw(text);

			text.setPosition(750, 100);
			text.setString("lavantaCount:  " + std::to_string(lavantaCount));
			window->draw(text);

			text.setPosition(750, 120);
			text.setString("minimixCount: " + std::to_string(minimixCount));
			window->draw(text);

			text.setPosition(750, 140);
			text.setString("redCount: " + std::to_string(redCount));
			window->draw(text);
			//COUNTERS

			text.setPosition(1000, 80);
			text.setString("Acc: %" + std::to_string(lastPredAccuracy));

			window->draw(text);

			window->draw(sf::Sprite(tex));
			window->draw(testSprite);
			window->display();
			
			//SFML

		}
		std::cout << "Epoch: " << Epoch << std::endl;

	}



	/*  TEST
	std::cout << netP.forward(toTensor(imp.bicubicResize(image, 250, 250))) << std::endl;

	

	torch::Tensor tensor = torch::eye(3);
	for (const auto& pair : net.named_parameters()) {
		std::cout << pair.key() << ": " << pair.value() << std::endl;
	}
	std::cout << "---------------------------------------------" << std::endl;
	for (const auto& pair : netP.named_parameters()) {
		std::cout << pair.key() << ": " << pair.value() << std::endl;
	}
	std::cout << "---------------------------------------------" << std::endl;
	std::cout << netP.forward(torch::ones({ 2, 4 })) << std::endl;
	std::cout << "---------------------------------------------" << std::endl;
	*/




	/* OLD
	
	torch::optim::Adam optimizer(netP.parameters(), torch::optim::AdamOptions(0.00001));

	torch::Tensor dogTarget = torch::ones(2);
	dogTarget[0] = 0.000;
	dogTarget[1] = 1.000;

	torch::Tensor catTarget = torch::ones(2);
	catTarget[0] = 1.000;
	catTarget[1] = 0.000;

	// Train
	for (size_t i = 0; i < 100; i++) {
		optimizer.zero_grad();
		if (i % 2)
		{
			torch::Tensor output = netP.forward(toTensor(imp.bicubicResize(toInfImage(*catImages.at(i/2)), 250, 250)));
			auto loss = torch::mse_loss(output, catTarget);
			std::cout << "Output " << output << " : " << std::endl;
			std::cout << "Loss " << i << " : " << loss.item<float>() << std::endl;
			loss.backward();
			optimizer.step();
		}
		else
		{
			torch::Tensor output = netP.forward(toTensor(imp.bicubicResize(toInfImage(*dogImages.at(i/2)), 250, 250)));
			auto loss = torch::mse_loss(output, dogTarget);
			std::cout << "Output " << output << " : " << std::endl;
			std::cout << "Loss " << i << " : " << loss.item<float>() << std::endl;
			loss.backward();
			optimizer.step();
		}
	}


	*/


	//std::string model_path = "model.pt";
	//torch::serialize::OutputArchive output_archive;
	//Net.save(output_archive);
	//output_archive.save_to(model_path);

	torch::save(Net,"model.pt");

	//std::cout << Net->forward(toTensor(imp.bicubicResize(toInfImage(*catImages.at(10)), 200, 200),device_type)) << std::endl;
	//std::cout << Net->forward(toTensor(imp.bicubicResize(toInfImage(*dogImages.at(10)), 200, 200),device_type)) << std::endl;


	int a;
	std::cin >> a;
	return 0;
}
