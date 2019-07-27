#include <opencv2\opencv.hpp>
#include ".././include/Stitching360.h"
#include <fstream>

#pragma comment(lib,"360Stitching.lib")
#define front 0
#define back 1
#define left 2
#define right 3



int main()
{
    /****************************************图片矫正***************************************************/
    cv::Mat mSrcFront = cv::imread("..\\..\\..\\inputImg\\front.png");
    cv::Mat mSrcBack = cv::imread("..\\..\\..\\inputImg\\back.png");
    cv::Mat mSrcLeft = cv::imread("..\\..\\..\\inputImg\\left.png");
    cv::Mat mSrcRight = cv::imread("..\\..\\..\\inputImg\\right.png");
    cv::Mat mDstLeft;
    cv::Mat mDstRight;
    cv::Mat mDstFront;
    cv::Mat mDstBack;
    cv::cuda::GpuMat cmDstImageLeft;
    cv::cuda::GpuMat cmDstImageRight;
    cv::cuda::GpuMat cmDstImageFront;
    cv::cuda::GpuMat cmDstImageBack;

    SurroundView *stitching360 = GetStitching();
    stitching360->Init(mSrcFront.cols, mSrcFront.rows);

    cmDstImageFront.upload(mSrcFront);
    cv::cuda::GpuMat cmDistortionFront = stitching360->Undistort(cmDstImageFront);
    cmDistortionFront.download(mDstFront);

    cmDstImageBack.upload(mSrcBack);
    cv::cuda::GpuMat cmDistortionBack = stitching360->Undistort(cmDstImageBack);
    cmDistortionBack.download(mDstBack);

    cmDstImageLeft.upload(mSrcLeft);
    cv::cuda::GpuMat cmDistortionLeft = stitching360->Undistort(cmDstImageLeft);
    cmDistortionLeft.download(mDstLeft);

    cmDstImageRight.upload(mSrcRight);
    cv::cuda::GpuMat cmDistortionRight = stitching360->Undistort(cmDstImageRight);
    cmDistortionRight.download(mDstRight);

    /***************************************投影变换*****************************************************/
    // 左侧
    cv::Point2f pSrcPointsLeft[] =
    {
        cv::Point2f(797, 696),// C->D
        cv::Point2f(2010, 722),// A->B
        cv::Point2f(1026, 480),
        cv::Point2f(1590, 468)

    };

    cv::Point2f pDstPointsLeft[] =
    {
        cv::Point2f(200, 150 + 250),
        cv::Point2f(200 + 770, 150 + 250),
        cv::Point2f(200, 150 + 30),
        cv::Point2f(200 + 770, 150 + 30)
    };
    cv::Mat mPerspectiveLeft = stitching360->PerspectiveTransform(mDstLeft, pSrcPointsLeft, pDstPointsLeft, cv::Size(1080, 500), left);
    cv::imshow("left", mPerspectiveLeft);

    // 右侧
    cv::Point2f pSrcPointsRight[] =
    {
        cv::Point2f(739, 692),// C->D
        cv::Point2f(1925, 683),// A->B
        cv::Point2f(995, 463),
        cv::Point2f(1572, 454)
    };

    cv::Point2f mDstPointsRight[] =
    {
        cv::Point2f(200, 150 + 250),
        cv::Point2f(200 + 770, 150 + 250),
        cv::Point2f(200, 150 + 30),
        cv::Point2f(200 + 770, 150 + 30)
    };
    cv::Mat mPerspectiveRight = stitching360->PerspectiveTransform(mDstRight, pSrcPointsRight, mDstPointsRight, cv::Size(1080, 500), right);
    cv::imshow("right", mPerspectiveRight);

    // 前方
    cv::Point2f mSrcPointsFront[] =
    {
        cv::Point2f(645, 666),// C->D
        cv::Point2f(1714, 680),// A->B
        cv::Point2f(927, 471),
        cv::Point2f(1492, 471)
    };

    cv::Point2f mDstPointsFront[] =
    {
        cv::Point2f(200, 150 + 250),
        cv::Point2f(200 + 770, 150 + 250),
        cv::Point2f(200, 150 + 30),
        cv::Point2f(200 + 770, 150 + 30)
    };
    cv::Mat mPerspectiveFront = stitching360->PerspectiveTransform(mDstFront, mSrcPointsFront, mDstPointsFront, cv::Size(1080, 500), front);
    cv::imshow("front", mPerspectiveFront);

    // 后方
    cv::Point2f pSrcPointsBack[] =
    {
        cv::Point2f(566, 686),
        cv::Point2f(1807, 705),
        cv::Point2f(896, 460),
        cv::Point2f(1527, 460)
    };

    cv::Point2f pDstPointsBack[] =
    {
        cv::Point2f(200, 150 + 250),
        cv::Point2f(200 + 770, 150 + 250),
        cv::Point2f(200, 150 + 30),
        cv::Point2f(200 + 770, 150 + 30)
    };
    cv::Mat mPerspectiveBack = stitching360->PerspectiveTransform(mDstBack, pSrcPointsBack, pDstPointsBack, cv::Size(1080, 500), back);
    cv::imshow("back", mPerspectiveBack);

    /**************************************拼接******************************************************/

    cv::Mat mCombine = stitching360->ImageStitching(mPerspectiveLeft, mPerspectiveRight, mPerspectiveFront, mPerspectiveBack);
    cv::imshow("Combined Image++", mCombine);
    cv::imwrite("combine.png", mCombine);
    cv::waitKey(0);
}


