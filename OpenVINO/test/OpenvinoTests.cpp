#include "openvino_detection.h"
#include "gtest/gtest.h"
#include <iostream>

using namespace std;

TEST(OpenvinoInference, InitializationModelFileMiss)
{
    file_name_t input_model = "";
    std::string device_name = "CPU";
    OpenvinoInference infer(input_model, device_name);
    EXPECT_EQ(infer.Initialization(), EXIT_FAILURE);
}

TEST(OpenvinoInference, InitializationDeviceError)
{

    file_name_t input_model = "../../model/model_DAD_3_7.xml";
    std::string device_name = "VPU";
    OpenvinoInference infer(input_model, device_name);
    EXPECT_EQ(infer.Initialization(), EXIT_FAILURE);
}

file_name_t input_model = "../../model/model_DAD_3_7.xml";
std::string device_name = "CPU";
std::shared_ptr<OpenvinoInference> infer;

TEST(OpenvinoInference, InitializationSuccess)
{
    infer = std::make_shared<OpenvinoInference>(input_model, device_name);
    EXPECT_EQ(infer->Initialization(), EXIT_SUCCESS);
}

TEST(OpenvinoInference, InferenceImage)
{
    file_name_t input_image_path = "../../model/phone_interact.jpg";
    EXPECT_EQ(
        infer->Inference(OpenvinoInference::PreProcessing, OpenvinoInference::ProcessOutput, input_image_path, infer),
        EXIT_SUCCESS);
}

TEST(OpenvinoInference, InferenceImageError)
{
    file_name_t input_image_path = "../model/phone_interact.jpg";
    EXPECT_EQ(
        infer->Inference(OpenvinoInference::PreProcessing, OpenvinoInference::ProcessOutput, input_image_path, infer),
        EXIT_FAILURE);
}

TEST(OpenvinoInference, InferenceMat)
{
    cv::Mat input_image = cv::imread("../../model/phone_interact.jpg");
    EXPECT_EQ(infer->Inference(OpenvinoInference::PreProcessing, OpenvinoInference::ProcessOutput, input_image, infer),
              EXIT_SUCCESS);
}

TEST(OpenvinoInference, InferenceRawData)
{

    cv::Mat input_image = cv::imread("../../model/phone_interact.jpg");
    int rawdata_height = input_image.rows;
    int rawdata_width = input_image.cols;
    EXPECT_EQ(infer->Inference(OpenvinoInference::PreProcessing, OpenvinoInference::ProcessOutput, rawdata_height,
                               rawdata_width, input_image.ptr<uchar>(0), CV_8UC3, infer),
              EXIT_SUCCESS);
}
TEST(OpenvinoInference, InferenceRawDataError)
{
    uchar *data = nullptr;
    EXPECT_EQ(infer->Inference(OpenvinoInference::PreProcessing, OpenvinoInference::ProcessOutput, 1000, 1000, data,
                               CV_8UC3, infer),
              EXIT_FAILURE);
}

TEST(OpenvinoInference, InferenceError)
{
    std::shared_ptr<OpenvinoInference> infer_error;
    infer_error = std::make_shared<OpenvinoInference>("../../model/model_DAD_3_7.xml", "VPU");
    file_name_t input_image_path = "../../model/phone_interact.jpg";
    EXPECT_EQ(infer_error->Inference(OpenvinoInference::PreProcessing, OpenvinoInference::ProcessOutput,
                                     input_image_path, infer_error),
              EXIT_FAILURE);

    cv::Mat input_image = cv::imread("../../model/phone_interact.jpg");
    EXPECT_EQ(infer_error->Inference(OpenvinoInference::PreProcessing, OpenvinoInference::ProcessOutput, input_image,
                                     infer_error),
              EXIT_FAILURE);

    int rawdata_height = input_image.rows;
    int rawdata_width = input_image.cols;
    EXPECT_EQ(infer_error->Inference(OpenvinoInference::PreProcessing, OpenvinoInference::ProcessOutput, rawdata_height,
                                     rawdata_width, input_image.ptr<uchar>(0), CV_8UC3, infer_error),
              EXIT_FAILURE);
}
