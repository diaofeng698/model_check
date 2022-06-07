#include "tensorrt_detection.h"

double getMillisecondsTimeStamp()
{
    double retTime = 0.0;
    timespec timeNow;
    clock_gettime(CLOCK_REALTIME, &timeNow);
    retTime = (double)(timeNow.tv_sec) * 1000.0;
    retTime += (double)(timeNow.tv_nsec) / 1e6;
    return retTime;
}

static constexpr int VERBOSE = 0x0;

static constexpr int DEBUG = (0x1 << 28);

static constexpr int INFO = (0x2 << 28);

static constexpr int WARNING = (0x3 << 28);

static constexpr int ERROR = (0x4 << 28);

void publishMessage(int msgType, char prefix, int featureCode, int code, std::string featurePrefix, std::string message)
{
    constexpr int NUMBER_OF_DIGITS = 8;
    int codeMsg = featureCode | msgType | code;
    std::stringstream streamMsg;
    (streamMsg << getMillisecondsTimeStamp() << ": "
               << "[" << prefix << "][" << std::hex << std::setfill('0') << std::setw(NUMBER_OF_DIGITS) << codeMsg
               << std::dec << "][" << featurePrefix << "] " << message)
        << std::endl;
    std::cout << streamMsg.str();
}

int main()
{
    std::shared_ptr<TensorRTInference> DaD_net;

    std::string input_image_path =
        "/home/fdiao/Downloads/TensorRT-8.2.1.8/samples/python/int8_caffe_mnist/DAD/06_googlenet/model/img.png";

    std::string DaDModelPath = std::string("/home/fdiao/Downloads/TensorRT-8.2.1.8/samples/python/int8_caffe_mnist/DAD/"
                                           "06_googlenet/model/model_cpp_int8.plan");

    DaD_net = std::make_shared<TensorRTInference>(DaDModelPath, "input_1:0", "Identity:0");

    if (DaD_net->TensorRTBuild())
    {
        std::cout << "Inference Engine Build Failed " << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat input_frame = cv::imread(input_image_path);

    if (DaD_net->TensorRTInfer(TensorRTInference::ProcessInputNPPI, TensorRTInference::VerifyOutput, input_frame,
                               DaD_net))
    {
        std::cout << "Inference Failed " << std::endl;
        return EXIT_FAILURE;
    }

    publishMessage(INFO, 'I', 1, 2, "WWW", "Pointer to adapter is null.");

    return EXIT_SUCCESS;
}
