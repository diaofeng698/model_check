#include <exception>
#include <inference_engine.hpp>
#include <iterator>
#include <memory>
#include <samples/classification_results.h>
#include <samples/common.hpp>
#include <samples/ocv_common.hpp>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
using namespace InferenceEngine;

#define tcout std::cout
#define file_name_t std::string
#define imread_t cv::imread
#define ClassificationResult_t ClassificationResult

struct OpenvinoInferenceResult
{
    int class_idx;
    float probability;
};

class OpenvinoInference
{
    typedef bool (*PreprocessingCallbackFun)(const cv::Mat &, std::shared_ptr<OpenvinoInference>);
    typedef void (*PostprocessingCallbackFun)(std::shared_ptr<OpenvinoInference>);

  public:
    file_name_t input_model_;
    file_name_t input_image_path_{""};
    std::string device_name_;
    OpenvinoInferenceResult openvino_result_;
    Blob::Ptr inputBlob_;
    OpenvinoInference(file_name_t input_model, std::string device_name);
    ~OpenvinoInference();
    bool Initialization();
    bool Inference(PreprocessingCallbackFun preprocessing_call_fun, PostprocessingCallbackFun postprocessing_call_fun,
                   int rawdata_height, int rawdata_width, void *rawdata, int type,
                   std::shared_ptr<OpenvinoInference> temp);
    bool Inference(PreprocessingCallbackFun preprocessing_call_fun, PostprocessingCallbackFun postprocessing_call_fun,
                   file_name_t input_image_path, std::shared_ptr<OpenvinoInference> temp);
    bool Inference(PreprocessingCallbackFun preprocessing_call_fun, PostprocessingCallbackFun postprocessing_call_fun,
                   cv::Mat image, std::shared_ptr<OpenvinoInference> temp);
    static bool PreProcessing(const cv::Mat &input, std::shared_ptr<OpenvinoInference> temp);
    static void ProcessOutput(std::shared_ptr<OpenvinoInference> temp);

  private:
    Core ie_;
    CNNNetwork network_;
    InferRequest::Ptr infer_request_;
    ExecutableNetwork executable_network_;
    bool ReadModel();
    bool ConfigureInputOutput();
    bool LoadingModel();
    void CreateInferRequest();
    bool PrepareInput(int rawdata_height, int rawdata_width, void *rawdata, int type, cv::Mat &image_out);
    bool PrepareInput(file_name_t input_image_path, cv::Mat &image_out);
    void DoSyncInference();
    void ProcessOutput();
    std::string input_name_;
    std::string output_name_;
    size_t modeldata_batch_{0};
    size_t modeldata_height_{0};
    size_t modeldata_width_{0};
    size_t modeldata_num_channels_{0};
    int class_num_{0};
};
