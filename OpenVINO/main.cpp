#include "openvino_detection.h"

int main()
{
    file_name_t input_model = "../model/Model_DAD_OpenVINO.xml";
    file_name_t input_image_path = "../model/chuanchuan_phone_interact_1MU9HI.jpg";
    std::string device_name = "CPU";

    std::shared_ptr<OpenvinoInference> infer;

    infer = std::make_shared<OpenvinoInference>(input_model, device_name);

    if (infer->Initialization())
        std::cout << "Openvino Initialization Failed, Please check ! " << std::endl;
    if (infer->Inference(OpenvinoInference::PreProcessing, OpenvinoInference::ProcessOutput, input_image_path, infer))
    {
        std::cout << "Inference Failed" << std::endl;
    }

    return 0;
}