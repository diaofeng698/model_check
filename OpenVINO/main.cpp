#include "openvino_detection.h"

int main()
{

    file_name_t input_model = "../../Model/Openvino/Model_DAD_OpenVINO.xml";

    std::string device_name = "CPU";

    std::shared_ptr<OpenvinoInference> infer;

    infer = std::make_shared<OpenvinoInference>(input_model, device_name);

    if (infer->Initialization())
        std::cout << "Openvino Initialization Failed, Please check ! " << std::endl;

    // // frame infer
    // file_name_t input_image_path = "../../Test_data/chuanchuan_phone_interact_0C82ZP.jpg";
    // if (infer->Inference(OpenvinoInference::PreProcessing, OpenvinoInference::ProcessOutput, input_image_path,
    // infer))
    // {
    //     std::cout << "Inference Failed" << std::endl;
    // }

    // Video Infer
    std::string input_video_path = "../../Test_data/dms_res_sglee.avi";
    cv::VideoCapture video(input_video_path);
    while (true)
    {
        cv::Mat input_frame;
        video >> input_frame;
        if (input_frame.empty())
        {
            break;
        }
        if (infer->Inference(OpenvinoInference::PreProcessing, OpenvinoInference::ProcessOutput, input_frame, infer))
        {
            std::cout << "Inference Failed " << std::endl;
            return EXIT_FAILURE;
        }
    }

    return 0;
}