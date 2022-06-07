#include "openvino_detection.h"

OpenvinoInference::OpenvinoInference(file_name_t input_model, std::string device_name)
{
    std::cout << "Openvino Inference Start Initialization " << std::endl;
    input_model_ = input_model;
    device_name_ = device_name;
    infer_request_ = nullptr;
}

OpenvinoInference::~OpenvinoInference()
{
    std::cout << "OpenvinoInference Destructor Finished! " << std::endl;
}

bool OpenvinoInference::Initialization()
{

    if (ReadModel())
        return EXIT_FAILURE;
    if (ConfigureInputOutput())
        return EXIT_FAILURE;
    if (LoadingModel())
        return EXIT_FAILURE;
    CreateInferRequest();
    return EXIT_SUCCESS;
}

bool OpenvinoInference::Inference(PreprocessingCallbackFun preprocessing_call_fun,
                                  PostprocessingCallbackFun postprocessing_call_fun, file_name_t input_image_path,
                                  std::shared_ptr<OpenvinoInference> temp)
{
    if (!temp->infer_request_)
    {
        return EXIT_FAILURE;
    }
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    cv::Mat temp_image;
    if (PrepareInput(input_image_path, temp_image))
    {
        return EXIT_FAILURE;
    }
    if (preprocessing_call_fun(temp_image, temp))
    {
        return EXIT_FAILURE;
    };
    DoSyncInference();
    postprocessing_call_fun(temp);
    std::cout << "Openvino Inference Finished" << std::endl;
    gettimeofday(&tv2, NULL);
    double diff_time = ((double)(tv2.tv_usec - tv1.tv_usec) / 1000.0) + ((double)(tv2.tv_sec - tv1.tv_sec) * 1000.0);
    std::cout << "Openvino Whole Infer Time [ms] : " << diff_time << std::endl;
    return EXIT_SUCCESS;
}

bool OpenvinoInference::Inference(PreprocessingCallbackFun preprocessing_call_fun,
                                  PostprocessingCallbackFun postprocessing_call_fun, int rawdata_height,
                                  int rawdata_width, void *rawdata, int type, std::shared_ptr<OpenvinoInference> temp)
{
    if (!temp->infer_request_)
    {
        return EXIT_FAILURE;
    }

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    cv::Mat temp_image;
    if (PrepareInput(rawdata_height, rawdata_width, rawdata, type, temp_image))
    {
        return EXIT_FAILURE;
    }
    if (preprocessing_call_fun(temp_image, temp))
    {
        return EXIT_FAILURE;
    }
    DoSyncInference();
    postprocessing_call_fun(temp);
    std::cout << "Openvino Inference Finished" << std::endl;
    gettimeofday(&tv2, NULL);
    double diff_time = ((double)(tv2.tv_usec - tv1.tv_usec) / 1000.0) + ((double)(tv2.tv_sec - tv1.tv_sec) * 1000.0);
    std::cout << "Openvino Whole Infer Time [ms] : " << diff_time << std::endl;
    return EXIT_SUCCESS;
}

bool OpenvinoInference::Inference(PreprocessingCallbackFun preprocessing_call_fun,
                                  PostprocessingCallbackFun postprocessing_call_fun, const cv::Mat image,
                                  std::shared_ptr<OpenvinoInference> temp)
{
    if (!temp->infer_request_)
    {
        return EXIT_FAILURE;
    }
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    if (preprocessing_call_fun(image, temp))
    {
        return EXIT_FAILURE;
    }
    DoSyncInference();
    postprocessing_call_fun(temp);
    std::cout << "Openvino Inference Finished" << std::endl;
    gettimeofday(&tv2, NULL);
    double diff_time = ((double)(tv2.tv_usec - tv1.tv_usec) / 1000.0) + ((double)(tv2.tv_sec - tv1.tv_sec) * 1000.0);
    std::cout << "Openvino Whole Infer Time [ms] : " << diff_time << std::endl;
    return EXIT_SUCCESS;
}

bool OpenvinoInference::ReadModel()
{
    if (access(input_model_.c_str(), F_OK) != -1)
    {
        network_ = ie_.ReadNetwork(input_model_);
        if (network_.getOutputsInfo().size() != 1)
        {
            throw std::logic_error("Sample supports topologies with 1 output only");
            return EXIT_FAILURE;
        }
        if (network_.getInputsInfo().size() != 1)
        {
            throw std::logic_error("Sample supports topologies with 1 input only");
            return EXIT_FAILURE;
        }
    }
    else
    {
        std::cout << "Model File Missed, Please check ! " << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

bool OpenvinoInference::ConfigureInputOutput()
{

    // --------------------------- Prepare input blobs
    if (network_.getInputsInfo().empty())
    {
        std::cerr << "Network inputs info is empty" << std::endl;
        return EXIT_FAILURE;
    }
    InputInfo::Ptr input_info = network_.getInputsInfo().begin()->second;
    input_name_ = network_.getInputsInfo().begin()->first;

    // input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    // input_info->setLayout(Layout::NCHW);
    input_info->setPrecision(Precision::FP32);

    // --------------------------- Prepare output blobs
    if (network_.getOutputsInfo().empty())
    {
        std::cerr << "Network outputs info is empty" << std::endl;
        return EXIT_FAILURE;
    }
    DataPtr output_info = network_.getOutputsInfo().begin()->second;
    output_name_ = network_.getOutputsInfo().begin()->first;

    output_info->setPrecision(Precision::FP32);

    return EXIT_SUCCESS;
}

bool OpenvinoInference::LoadingModel()
{
    try
    {
        executable_network_ = ie_.LoadNetwork(network_, device_name_);
        return EXIT_SUCCESS;
    }
    catch (std::exception &e)
    {
        std::cout << "Check Device Plugin " << std::endl;
        return EXIT_FAILURE;
    }
}

void OpenvinoInference::CreateInferRequest()
{
    infer_request_ = executable_network_.CreateInferRequestPtr();
    inputBlob_ = infer_request_->GetBlob(input_name_);
    SizeVector dims = inputBlob_->getTensorDesc().getDims();

    modeldata_batch_ = dims[0];
    modeldata_height_ = dims[1];
    modeldata_width_ = dims[2];
    modeldata_num_channels_ = dims[3];

    Blob::Ptr outputBlob = infer_request_->GetBlob(output_name_);
    SizeVector output_dims = outputBlob->getTensorDesc().getDims();

    class_num_ = output_dims[1];
}

bool OpenvinoInference::PreProcessing(const cv::Mat &input, std::shared_ptr<OpenvinoInference> temp)
{

    cv::Mat resized_img;
    cv::resize(input, resized_img, cv::Size(temp->modeldata_width_, temp->modeldata_height_));
    std::cout << "resized inputH " << resized_img.rows << " inputW " << resized_img.cols << " inputChannel "
              << resized_img.channels() << std::endl;

    cv::Mat gray_img;
    cv::cvtColor(resized_img, gray_img, cv::COLOR_BGR2GRAY);
    std::cout << "gray inputH " << gray_img.rows << " inputW " << gray_img.cols << " inputChannel "
              << gray_img.channels() << std::endl;

    MemoryBlob::Ptr minput = as<MemoryBlob>(temp->inputBlob_);
    if (!minput)
    {
        std::cout << "Unable to Cast InputBlob to MemoryBlob" << std::endl;
        return EXIT_FAILURE;
    }
    // locked memory holder should be alive all time while access to its
    // buffer happens
    auto minputHolder = minput->wmap();

    auto data = minputHolder.as<PrecisionTrait<Precision::FP32>::value_type *>();
    if (!data)
    {
        throw std::runtime_error("Input blob has not allocated buffer");
        return EXIT_FAILURE;
    }

    for (int b = 0, volImg = temp->modeldata_num_channels_ * temp->modeldata_height_ * temp->modeldata_width_;
         b < temp->modeldata_batch_; b++)
    {
        for (int idx = 0, volChl = temp->modeldata_height_ * temp->modeldata_width_; idx < volChl; idx++)
        {

            for (int c = 0; c < temp->modeldata_num_channels_; ++c)
            {
                data[b * volImg + idx * temp->modeldata_num_channels_ + c] = gray_img.data[idx];
            }
        }
    }

    return EXIT_SUCCESS;
}

bool OpenvinoInference::PrepareInput(int rawdata_height, int rawdata_width, void *rawdata, int type, cv::Mat &image_out)
{
    if (!rawdata)
    {
        std::cout << "Input Frame Load Failed, Please check !" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "raw inputH " << rawdata_height << " inputW " << rawdata_width << " inputChannel " << 3 << std::endl;
    cv::Mat raw_mat(rawdata_height, rawdata_width, type, rawdata);
    image_out = raw_mat;

    return EXIT_SUCCESS;
}

bool OpenvinoInference::PrepareInput(file_name_t input_image_path, cv::Mat &image_out)
{

    if (access(input_image_path.c_str(), F_OK) != -1)
    {
        image_out = imread_t(input_image_path);
        // if (!image_out.data)
        // {
        //     std::cout << "Image File Load Failed" << std::endl;
        //     return EXIT_FAILURE;
        // }
        std::cout << "raw inputH " << image_out.rows << " inputW " << image_out.cols << " inputChannel "
                  << image_out.channels() << std::endl;
    }
    else
    {
        std::cout << "Image File Missed, Please check ! " << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

void OpenvinoInference::DoSyncInference()
{
    /* Running the request synchronously */
    infer_request_->Infer();
}

void OpenvinoInference::ProcessOutput(std::shared_ptr<OpenvinoInference> temp)
{
    Blob::Ptr output = temp->infer_request_->GetBlob(temp->output_name_);
    // Print classification results
    ClassificationResult_t classificationResult(output, {temp->input_image_path_}, temp->modeldata_batch_,
                                                temp->class_num_);
    classificationResult.print();
    temp->openvino_result_.class_idx = classificationResult._max_idx;
    temp->openvino_result_.probability = classificationResult._max_prob;

    std::cout << "Inference Result ID : " << temp->openvino_result_.class_idx
              << " Conf : " << temp->openvino_result_.probability << std::endl;
}
