#pragma once
#include <opencv2\opencv.hpp>
#include <experimental/filesystem>
#include <fstream>


class Stitching360 {
private:
	std::string                             m_sImageRoot;	/* 图片文件夹 */
    std::string                             m_sLastName;    /* 图片后缀名 */
    std::string                             m_sCaliResult; /* 存标定数据的文件名*/
    cv::Size                                m_szImage;
    cv::Size                                m_szBoard;	/****    定标板上每行、列的角点数       ****/
	int                                     m_nImageCount;	/****    标定图像数量     ****/
	int                                     m_nSuccessImageNum;                /****   成功提取角点的棋盘图数量    ****/
	cv::Matx33d                             m_mIntrinsicMatrix;    /*****    摄像机内参数矩阵    ****/
	cv::Matx33d                             m_mNewIntrinsicMat;   /** 摄像头新的内参用于矫正 **/
	cv::Vec4d                               m_vDistortionCoeffs;     /* 摄像机的4个畸变系数：k1,k2,k3,k4*/
	std::vector<cv::Mat>                    m_vImageSeq;					/* 保存图像 */
    std::vector<std::vector<cv::Point2f>>   m_vCornersSeq;    /****  保存检测到的所有角点       ****/
    std::vector<cv::Point2f>                n_vCorners;                  /****    缓存每幅图像上检测到的角点       ****/
	std::vector<cv::Vec3d>                  m_vRotationVectors;                           /* 每幅图像的旋转向量 */
	std::vector<cv::Vec3d>                  m_vTranslationVectors;                        /* 每幅图像的平移向量 */
    cv::cuda::GpuMat                        m_cmMap1; /* 最终矫正的映射表 */
    cv::cuda::GpuMat                        m_cmMap2; /* 最终矫正的映射表 */

	int findCorners();
	int cameraCalibrate(int count);
	int savePara();
	

public:
	Stitching360();
    ~Stitching360();
    /************************相机标定以及矫正****************************/
	int Init(int nSrcHeight, int nSrcWidth);
	cv::cuda::GpuMat Undistort(cv::cuda::GpuMat &mSrcImg);

    /************************逆投影变换*******************************/
    cv::Mat PerspectiveTransform(cv::InputArray aInput, cv::Point2f *pSrcPoints, cv::Point2f *pDstPoints, cv::Size sOutputSize, int nOrientation);

    /*************************图像拼接**********************************/
    cv::Mat ImageStitching(cv::Mat aInputLeft, cv::Mat aInputRight, cv::Mat aInputFront, cv::Mat aInputBack);

};