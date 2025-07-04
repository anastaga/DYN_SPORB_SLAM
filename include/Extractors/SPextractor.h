/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef SPEXTRACTOR_H
#define SPEXTRACTOR_H

#include <vector>
#include <list>
// #include <opencv/cv.h>
#include "YoloDetection.h"
#include <string>
#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "Tracking.h"
#include "Extractors/superpoint_onnx.h"
//#include "Extractors/super_point.h"
#ifdef EIGEN_MPL2_ONLY
#undef EIGEN_MPL2_ONLY
#endif


namespace ORB_SLAM3
{


class SPextractor
{
public:
        void SetTrackingPtr(ORB_SLAM3::Tracking* pTracker); 
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    SPextractor(int nfeatures, float scaleFactor, int nlevels,
                 float iniThFAST, float minThFAST);

    // SPextractor(const SuperPointConfig &super_point_config);

 ~SPextractor(){
    if (yolo) {
        delete yolo;
        yolo = nullptr;
    }
}

    // Compute the SP features and descriptors on an image.
    // SP are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    int operator()( cv::InputArray image,
      std::vector<cv::KeyPoint>& keypoints,
      cv::Mat& descriptors);

    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;

    SuperPointOnnxRunner* featureExtractor;
    std::string mModelstr = "onnx";//表明创建onnx模型还是tensorrt

    YoloV8Detector* yolo = nullptr;
    float lastmatchnum = 0;
        void FilterDynamicKeypoints(const cv::Mat& image,
                                std::vector<cv::KeyPoint>& keypoints,
                                cv::Mat& descriptors);
protected:
ORB_SLAM3::Tracking* mpTracking = nullptr;
    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints, cv::Mat &_desc);    
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

    // void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    // std::vector<cv::Point> pattern;

    int nfeatures;
    double scaleFactor;
    int nlevels;
    float iniThFAST;
    float minThFAST;

    int ExtractSingleLayer(const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints, 
                            cv::Mat &localDescriptors);
    
    int ExtractMultiLayers(const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints,
                           cv::Mat &Descriptors);
    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;    
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    //std::shared_ptr<SuperPoint> mModel;

};


} //namespace ORB_SLAM

#endif

