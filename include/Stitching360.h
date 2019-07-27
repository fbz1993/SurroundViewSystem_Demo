#pragma once

#if defined(_DLL_EXPORTS) // inside DLL
#   define DLL_API   __declspec(dllexport)
#else // outside DLL
#   define DLL_API   __declspec(dllimport)
#endif  // XYZLIBRARY_EXPORT

#include <opencv2\opencv.hpp>
#include <experimental/filesystem>
#include <fstream>

class SurroundView
{
public:
    /************************相机标定以及矫正****************************/
    virtual int Init(int nSrcHeight, int nSrcWidth) = 0;
    virtual cv::cuda::GpuMat Undistort(cv::cuda::GpuMat &mSrcImg) = 0;

    /************************逆投影变换*******************************/
    virtual cv::Mat PerspectiveTransform(cv::InputArray aInput, cv::Point2f *pSrcPoints, cv::Point2f *pDstPoints, cv::Size sOutputSize, int nOrientation) = 0;

    /*************************图像拼接**********************************/
    virtual cv::Mat ImageStitching(cv::Mat aInputLeft, cv::Mat aInputRight, cv::Mat aInputFront, cv::Mat aInputBack) = 0;

};

extern "C" DLL_API SurroundView *GetStitching();