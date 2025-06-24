#include "YoloDetection.h" // Include your header

#include <iostream>
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
static const std::vector<std::string> coco_class_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

YoloV8Detector::YoloV8Detector(const std::string& modelPath, float confThreshold, float nmsThreshold)
    : confThreshold_(confThreshold), nmsThreshold_(nmsThreshold) {

    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YoloV8");

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_CUDA
    OrtCUDAProviderOptions cudaOptions;
    cudaOptions.device_id = 0;
    sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions);

    Ort::AllocatedStringPtr input_name_ptr = session_->GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
inputName_ = std::string(input_name_ptr.get());
outputName_ = std::string(output_name_ptr.get());


    Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    inputDims_ = inputTensorInfo.GetShape();

    if (inputDims_[2] == -1 || inputDims_[3] == -1) {
        inputDims_[2] = 640;
        inputDims_[3] = 640;
    }

    Ort::TypeInfo outputTypeInfo = session_->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    outputDims_ = outputTensorInfo.GetShape();
}


    
YoloV8Detector::~YoloV8Detector() {

}

    
    // Perform object detection on a BGR image. Returns detections (filtering to persons class 0).

std::vector<Detection> YoloV8Detector::detect(const cv::Mat& imageBGR)
{

// ---------------------------------------------------------------------------
// run-time constants read from the ONNX input tensor (480 × 360)
// ---------------------------------------------------------------------------
const int netH = static_cast<int>(inputDims_[2]);   // 360
const int netW = static_cast<int>(inputDims_[3]);   // 480

// remember the native sequence size only once
static int seqW = 0, seqH = 0;
if (seqW == 0 || seqH == 0) {
    seqW = imageBGR.cols;
    seqH = imageBGR.rows;
    std::cout << "[YOLO] sequence size  = " << seqW << 'x' << seqH << '\n';
}

// ───────────────────── 1. uniform stretch (no padding) ────────────────────
cv::Mat resized;
cv::resize(imageBGR, resized, cv::Size(netW, netH));   // 480×360

// ───────────────────── 2. BGR/GRAY → RGB float32 CHW ──────────────────────
cv::Mat rgb;
if (resized.channels() == 3)
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
else
    cv::cvtColor(resized, rgb, cv::COLOR_GRAY2RGB);

rgb.convertTo(rgb, CV_32F, 1.f/255.f);

std::vector<cv::Mat> ch(3);
cv::split(rgb, ch);

std::vector<float> inData(3 * netH * netW);
const size_t chanArea = netH * netW;
for (int c = 0; c < 3; ++c)
    std::memcpy(inData.data() + c * chanArea, ch[c].data,
                chanArea * sizeof(float));

std::array<int64_t,4> inShape = {1, 3, netH, netW};
Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
                          OrtArenaAllocator, OrtMemTypeDefault);
Ort::Value inTensor = Ort::Value::CreateTensor<float>(
                          mem, inData.data(), inData.size(),
                          inShape.data(), inShape.size());

// ───────────────────── 3. inference ───────────────────────────────────────
const char* inNames[]  = { inputName_.c_str() };
const char* outNames[] = { outputName_.c_str() };

auto outs = session_->Run(Ort::RunOptions{nullptr},
                          inNames,  &inTensor, 1,
                          outNames, 1);

const float* p  = outs[0].GetTensorMutableData<float>();
const int64_t N = outs[0].GetTensorTypeAndShapeInfo().GetShape()[1]; // 300

std::vector<Detection> detections;
for (int i = 0; i < N; ++i, p += 6)
{
    float x1 = p[0];      // ← already corner coordinates
    float y1 = p[1];
    float x2 = p[2];
    float y2 = p[3];
    float conf = p[4];
    int   cls  = int(p[5]);

    if (conf < confThreshold_) continue;
    if (!(cls == 0 || cls == 2 || cls == 16)) continue;  // person / car / dog

    Detection d;
    d.classId    = cls;
    d.confidence = conf;
    d.label      = coco_class_names[cls];
    d.box        = cv::Rect( cv::Point(int(std::round(x1)),
                                      int(std::round(y1))),
                             cv::Point(int(std::round(x2)),
                                      int(std::round(y2))) );
    detections.push_back(d);
}
return detections;


}


