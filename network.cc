#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace tensorflow;

class Network {

    Session* session;
    GraphDef graph_def;
  public:
    Network(std::string, std::string);
    std::vector<float> Classify(std::vector<std::vector<std::vector<std::vector<float>>>>);
    ~Network();
};

Network::Network(std::string graphPath, std::string weightsPath)
{
	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok()) {
		throw status.ToString();
	}
	status = ReadBinaryProto(Env::Default(), graphPath, &graph_def);
	if (!status.ok()) {
		throw status.ToString();
	}
	status = session->Create(graph_def);
	if (!status.ok()) {
		throw status.ToString();
	}
	Tensor restorePath(DT_STRING, TensorShape());
	restorePath.scalar<std::string>()() =  weightsPath;
	std::vector<std::pair<std::string, tensorflow::Tensor>> restoreInputs = {
		{ "save/Const:0", restorePath}
	};
	status = session->Run(restoreInputs, {}, {"save/restore_all"}, nullptr);
	if (!status.ok()) {
		throw status.ToString();
	}
}

std::vector<float> Network::Classify(std::vector<std::vector<std::vector<std::vector<float>>>> data)
{
	Tensor inputData(DT_FLOAT, TensorShape({1, 60, 40, 50}));
	Tensor dropout(DT_FLOAT, TensorShape());
	std::vector<tensorflow::Tensor> outputs;
	dropout.scalar<float>()() = 1.;
	auto inputDataAccesor = inputData.tensor<float, 4>();
	for(int x1=0; x1<60; x1++)
		for(int x2=0; x2<40; x2++)
			for(int x3=0; x3<25; x3++)
				for (int x4=0; x4<2; x4++)
				{
					inputDataAccesor(0, x1, x2, (x3*2)+x4) = data.at(x1).at(x2).at(x3).at(x4); 
				}
	std::vector<std::pair<std::string, tensorflow::Tensor>> finalInputs = {
	    { "NormalizedInput", inputData},
	    { "DropoutRate", dropout}
	};
	Status status = session->Run(finalInputs, {"finalesResult"}, {"finalesResult"}, &outputs);
	if (!status.ok()) {
		throw status.ToString();
	}
	auto output_c = outputs[0].tensor<float, 2>();
	std::vector<float> ret;
	ret.push_back(output_c(0,0));
	ret.push_back(output_c(0,1));
	return ret;
}

Network::~Network()
{
	session->Close();
}
