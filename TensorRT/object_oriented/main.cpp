#include "tensorrt_detection.h"

int main()
{
    std::shared_ptr<TensorRTInference> DaD_net;
    std::string DaDModelPath = std::string("../../../Model/Tensorrt_x86/Model_DAD_TensorRT__FP32.plan");

    DaD_net = std::make_shared<TensorRTInference>(DaDModelPath, "input_1", "dense");

    if (DaD_net->TensorRTBuild())
    {
        std::cout << "Inference Engine Build Failed " << std::endl;
        return EXIT_FAILURE;
    }

    // // Frame Infer
    // std::string input_image_path = "../../../Test_data/chuanchuan_phone_interact_0C82ZP.jpg";
    // cv::Mat input_frame = cv::imread(input_image_path);

    // if (DaD_net->TensorRTInfer(TensorRTInference::ProcessInputOpenCV, TensorRTInference::VerifyOutput, input_frame,
    //                            DaD_net))
    // {
    //     std::cout << "Inference Failed " << std::endl;
    //     return EXIT_FAILURE;
    // }

    // Video Infer
    std::string input_video_path = "../../../Test_data/dms_res_sglee.avi";
    cv::VideoCapture video(input_video_path);
    while (true)
    {
        cv::Mat input_frame;
        video >> input_frame;
        if (input_frame.empty())
        {
            break;
        }
        if (DaD_net->TensorRTInfer(TensorRTInference::ProcessInputOpenCV, TensorRTInference::VerifyOutput, input_frame,
                                   DaD_net))
        {
            std::cout << "Inference Failed " << std::endl;
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
