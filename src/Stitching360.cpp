#define _DLL_EXPORTS
#include ".././include/Stitching360.h"
#define front 0
#define back 1
#define left 2
#define right 3

class Stitching360 :public SurroundView
{
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
    void OnMouseAction(int event, int x, int y, int flags, void *para);


public:
    Stitching360();
    ~Stitching360();
    /************************相机标定以及矫正****************************/
    virtual int Init(int nSrcHeight, int nSrcWidth);
    virtual cv::cuda::GpuMat Undistort(cv::cuda::GpuMat &mSrcImg);

    /************************逆投影变换*******************************/
    virtual cv::Mat PerspectiveTransform(cv::InputArray aInput, cv::Point2f *pSrcPoints, cv::Point2f *pDstPoints, cv::Size sOutputSize, int nOrientation);

    /*************************图像拼接**********************************/
    virtual cv::Mat ImageStitching(int nWidth, int nHeight, cv::Mat aInputLeft, cv::Mat aInputRight, cv::Mat aInputFront, cv::Mat aInputBack,
        std::vector<cv::Point> vPtsInputLeft, std::vector<cv::Point> vPtsInputRight, std::vector<cv::Point> vPtsInputFront, std::vector<cv::Point> vPtsInputBack);

};

Stitching360::Stitching360():m_sImageRoot("..\\..\\..\\CaliImg\\"), m_sLastName(".png"), m_sCaliResult("..\\..\\..\\src\\result.txt"), m_szBoard(cv::Size(7, 7)), m_nImageCount(15), m_nSuccessImageNum(0)
{
}

Stitching360::~Stitching360() 
{
    cv::destroyAllWindows();
}

int Stitching360::Init(int nSrcHeight, int nSrcWidth)
{
	bool isExist = std::experimental::filesystem::exists(m_sCaliResult);
	m_szImage = cv::Size(nSrcHeight, nSrcWidth);
	if (!isExist) 
	{
		int count = findCorners();
		cameraCalibrate(count);
		savePara();
	}
	else
	{
		std::ifstream fin(m_sCaliResult);
		float d;
		int i = 0;
		// 这里还要处理一下读入的方式
		while (fin >> d) {
			if (i <= 8)
				m_mIntrinsicMatrix(i / 3, i % 3) = d;
			else if (i <= 12)
				m_vDistortionCoeffs(i - 9) = d;
			else
				m_mNewIntrinsicMat((i - 13) / 3, (i - 13) % 3) = d;
			i++;
		}
	}
	cv::Mat map1, map2;
	cv::fisheye::initUndistortRectifyMap(m_mIntrinsicMatrix, m_vDistortionCoeffs, cv::Matx33d::eye(), m_mNewIntrinsicMat, m_szImage + cv::Size(200,200), CV_32FC1, map1, map2);
	m_cmMap1.upload(map1);
	m_cmMap2.upload(map2);
    return 1;
}

int Stitching360::findCorners() 
{
	int count = 0;
	for (int i = 0; i != m_nImageCount; i++)
	{
        std::cout << "Frame #" << i + 1 << "..." << std::endl;
        std::string imageFileName;
		std::stringstream StrStm;
		StrStm << i + 1;
		StrStm >> imageFileName;
		imageFileName += m_sLastName;
		cv::Mat image = cv::imread(m_sImageRoot + imageFileName);
		/* 提取角点 */
		cv::Mat imageGray;
		cvtColor(image, imageGray, CV_RGB2GRAY);
		/*输入图像，角点数，检测到的角点，寻找角点前对图像做的调整*/
		bool patternfound = cv::findChessboardCorners(image, m_szBoard, n_vCorners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE +
			cv::CALIB_CB_FAST_CHECK);

		if (!patternfound)
		{
            std::cout << "找不到角点，需删除图片文件" << imageFileName << "重新排列文件名，再次标定" << std::endl;
			getchar();
			exit(1);
		}
		else
		{
			/* 亚像素精确化,对检测到的整数坐标角点精确化，精确化后的点存在corners中， 最小二乘迭代100次，误差在0.001*/
			cornerSubPix(imageGray, n_vCorners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100, 0.001));
            std::cout << "Frame corner#" << i + 1 << "...end" << std::endl;

			count = count + n_vCorners.size();
			m_nSuccessImageNum = m_nSuccessImageNum + 1;
			m_vCornersSeq.push_back(n_vCorners);
		}
		m_vImageSeq.push_back(image);
	}
	return count;
    std::cout << "角点提取完成！\n";
    return 1;
}

int Stitching360::cameraCalibrate(int count) {
    std::cout << "开始定标………………" << std::endl;
	cv::Size square_size = cv::Size(20, 20); /**** 每一个格子是20m*20mm ****/
    std::vector<std::vector<cv::Point3f>>  object_Points;        /****  保存定标板上角点的三维坐标   ****/

	cv::Mat image_points = cv::Mat(1, count, CV_32FC2, cv::Scalar::all(0));  /*****   保存提取的所有角点   *****/
    std::vector<int>  point_counts;
	/* 初始化定标板上角点的三维坐标 */
	for (int t = 0; t<m_nSuccessImageNum; t++)
	{
        std::vector<cv::Point3f> tempPointSet;
		for (int i = 0; i<m_szBoard.height; i++)
		{
			for (int j = 0; j<m_szBoard.width; j++)
			{
				/* 假设定标板放在世界坐标系中z=0的平面上 */
				cv::Point3f tempPoint;
				tempPoint.x = i*square_size.width;
				tempPoint.y = j*square_size.height;
				tempPoint.z = 0;
				tempPointSet.push_back(tempPoint);
			}
		}
		object_Points.push_back(tempPointSet);
	}
	for (int i = 0; i< m_nSuccessImageNum; i++)
	{
		point_counts.push_back(m_szBoard.width*m_szBoard.height);
	}
	/* 开始定标 */
	cv::Size image_size = m_vImageSeq[0].size();
	int flags = 0;
	flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	flags |= cv::fisheye::CALIB_CHECK_COND;
	flags |= cv::fisheye::CALIB_FIX_SKEW;
	/* 角点在标定板内的坐标， 角点在图像中的坐标， 图片大小， 内参矩阵， 四个畸变参数， 输出的旋转向量， 输出的平移向量， 迭代次数 误差*/
	cv::fisheye::calibrate(object_Points, m_vCornersSeq, image_size, m_mIntrinsicMatrix, m_vDistortionCoeffs, m_vRotationVectors, m_vTranslationVectors, flags, cv::TermCriteria(3, 20, 1e-6));
	cv::fisheye::estimateNewCameraMatrixForUndistortRectify(m_mIntrinsicMatrix, m_vDistortionCoeffs, image_size, cv::noArray(), m_mNewIntrinsicMat, 0.8, image_size, 1.0);
    std::cout << "定标完成！\n";
    return 1;
}

int Stitching360::savePara() {
    std::cout << "开始保存定标结果………………" << std::endl;
    std::ofstream fout(m_sCaliResult);
	/*相机内参数矩阵*/
    fout << m_mIntrinsicMatrix(0,0) << ' ' << m_mIntrinsicMatrix(0, 1) << ' ' << m_mIntrinsicMatrix(0, 2) << ' ' << m_mIntrinsicMatrix(1, 0) << ' ' << m_mIntrinsicMatrix(1, 1)
		<< ' ' << m_mIntrinsicMatrix(1, 2) << ' ' << m_mIntrinsicMatrix(2, 0) << ' ' << m_mIntrinsicMatrix(2, 1) << ' ' << m_mIntrinsicMatrix(2, 2) << std::endl;
	/*畸变系数*/
	fout << m_vDistortionCoeffs(0) << ' '<< m_vDistortionCoeffs(1) <<' '<< m_vDistortionCoeffs(2) << ' ' << m_vDistortionCoeffs(3) << std::endl;
	/*矫正内参矩阵*/
	fout << m_mNewIntrinsicMat(0, 0) << ' ' << m_mNewIntrinsicMat(0, 1) << ' ' << m_mNewIntrinsicMat(0, 2) << ' ' << m_mNewIntrinsicMat(1, 0) << ' ' << m_mNewIntrinsicMat(1, 1)
		<< ' ' << m_mNewIntrinsicMat(1, 2) << ' ' << m_mNewIntrinsicMat(2, 0) << ' ' << m_mNewIntrinsicMat(2, 1) << ' ' << m_mNewIntrinsicMat(2, 2) << std::endl;
    std::cout << "完成保存" << std::endl;
	fout << std::endl;
    return 1;
}

cv::cuda::GpuMat Stitching360::Undistort(cv::cuda::GpuMat &mSrcImg)
{
	//cv::Mat map1, map2;
	//fisheye::initUndistortRectifyMap();
	
	//cv::remap();
	cv::cuda::GpuMat mDstImg;
	//cv::remap(mSrcImg, mDstImg, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
	cv::cuda::remap(mSrcImg, mDstImg, m_cmMap1, m_cmMap2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
	return mDstImg;
}

cv::Mat Stitching360::PerspectiveTransform(cv::InputArray aInput, cv::Point2f *pSrcPoints, cv::Point2f *pDstPoints, cv::Size sOutputSize, int nOrientation) 
{
    cv::Mat mPerspective = cv::getPerspectiveTransform(pSrcPoints, pDstPoints);
    cv::Mat mPerspectiveImg;
    cv::warpPerspective(aInput, mPerspectiveImg, mPerspective, sOutputSize, cv::INTER_LINEAR);
    if (nOrientation == front)
    {

    }
    else if (nOrientation == back)
    {
        cv::flip(mPerspectiveImg, mPerspectiveImg, -1);
    }
    else if (nOrientation == left)
    {
        cv::transpose(mPerspectiveImg, mPerspectiveImg);
        cv::flip(mPerspectiveImg, mPerspectiveImg, 0);
    }
    else if (nOrientation == right)
    {
        cv::transpose(mPerspectiveImg, mPerspectiveImg);
        cv::flip(mPerspectiveImg, mPerspectiveImg, 1);
    }

    return mPerspectiveImg;
}

cv::Mat Stitching360::ImageStitching(int nWidth, int nHeight, cv::Mat mInputLeft, cv::Mat mInputRight, cv::Mat mInputFront, cv::Mat mInputBack,
    std::vector<cv::Point> vPtsLeft, std::vector<cv::Point> vPtsRight, std::vector<cv::Point> vPtsFront, std::vector<cv::Point> vPtsBack)
{
    cv::Mat mCombine = cv::Mat::zeros(1600, 1600, mInputRight.type());
    cv::Mat mRoiInputRight = cv::Mat::zeros(mInputRight.size(), CV_8U);
    cv::Mat mRoiInputLeft = cv::Mat::zeros(mInputLeft.size(), CV_8U);
    cv::Mat mRoiInputFront = cv::Mat::zeros(mInputFront.size(), CV_8U);
    cv::Mat mRoiInputBack = cv::Mat::zeros(mInputBack.size(), CV_8U);

    std::vector<cv::Point> vStitchFront;
    std::vector<cv::Point> vStitchBack;

    std::vector<std::vector<cv::Point>> vContourInputRight;
    std::vector<std::vector<cv::Point>> vContourInputLeft;
    std::vector<std::vector<cv::Point>> vContourInputFront;
    std::vector<std::vector<cv::Point>> vContourInputBack;

    cv::Mat mImgRoiInputFront;
    cv::Mat mImgRoiInputBack;
    cv::Mat mImgRoiInputLeft;
    cv::Mat mImgRoiInputRight;

    /***********************************切割**************************************************/
    // front切割
    vStitchFront.push_back(cv::Point(vPtsFront.at(1).x - nHeight + vPtsFront.at(1).y, nHeight));
    vStitchFront.push_back(cv::Point(vPtsFront.at(0).x + nHeight - vPtsFront.at(0).y, nHeight));
    if (vPtsFront.at(0).x > vPtsFront.at(0).y)
    {
        vStitchFront.push_back(cv::Point(vPtsFront.at(0).x - vPtsFront.at(0).y, 0));
    }
    else
    {
        vStitchFront.push_back(cv::Point(0, vPtsFront.at(0).y - vPtsFront.at(0).x));
        vStitchFront.push_back(cv::Point(0, 0));
    }
    if (nWidth - vPtsFront.at(1).x > vPtsFront.at(1).y)
    {
        vStitchFront.push_back(cv::Point(vPtsFront.at(1).x + vPtsFront.at(0).y, 0));
    }
    else
    {
        vStitchFront.push_back(cv::Point(nWidth, 0));
        vStitchFront.push_back(cv::Point(nWidth, vPtsFront.at(1).y - nWidth + vPtsFront.at(1).x));
    }

    // back切割
    vStitchBack.push_back(cv::Point(vPtsBack.at(1).x - vPtsBack.at(1).y, 0));
    vStitchBack.push_back(cv::Point(vPtsBack.at(0).x + vPtsBack.at(0).y, 0));
    if (nHeight - vPtsBack.at(0).y < vPtsBack.at(0).x)
    {
        vStitchBack.push_back(cv::Point(vPtsBack.at(0).x - nHeight + vPtsBack.at(0).y, nHeight));
    }
    else
    {
        vStitchBack.push_back(cv::Point(0, vPtsBack.at(0).x + vPtsBack.at(0).y));
        vStitchBack.push_back(cv::Point(0, nHeight));
    }
    if (nWidth - vPtsBack.at(1).x > vPtsBack.at(1).y)
    {
        vStitchBack.push_back(cv::Point(vPtsBack.at(1).x + nHeight - vPtsBack.at(1).y, nHeight));
    }
    else
    {
        vStitchBack.push_back(cv::Point(nWidth, nHeight));
        vStitchBack.push_back(cv::Point(nWidth, vPtsBack.at(1).y - nWidth + vPtsBack.at(1).x));
    }


    /*****************************计算边缘************************************/
    int nDiffFL_x = vPtsLeft.at(0).x - vPtsFront.at(0).x;
    int nDiffBL_x = vPtsLeft.at(1).x - vPtsBack.at(0).x;
    if (nDiffFL_x < nDiffBL_x)
    {
        if (nDiffFL_x <= 0)
        {
            mImgRoiInputFront = mCombine(cv::Rect(0, 0, nWidth, nHeight));
            mImgRoiInputLeft = mCombine(cv::Rect(-nDiffFL_x, vPtsFront.at(0).y - vPtsLeft.at(0).y, nHeight, nWidth));
            mImgRoiInputBack = mCombine(cv::Rect(-nDiffFL_x + nDiffBL_x, vPtsFront.at(0).y + vPtsLeft.at(1).y - vPtsLeft.at(0).y - vPtsBack.at(0).y, nWidth, nHeight));
            mImgRoiInputRight = mCombine(cv::Rect(vPtsLeft.at(0).x + vPtsFront.at(1).x - vPtsFront.at(0).x - vPtsRight.at(0).x - nDiffFL_x, vPtsFront.at(1).y - vPtsRight.at(0).y, nHeight, nWidth));
        }
        else
        {
            mImgRoiInputFront = mCombine(cv::Rect(nDiffFL_x, 0, nWidth, nHeight));
            mImgRoiInputLeft = mCombine(cv::Rect(0, vPtsLeft.at(0).y - vPtsFront.at(0).y, nHeight, nWidth));
            mImgRoiInputBack = mCombine(cv::Rect(nDiffBL_x, vPtsFront.at(0).y + vPtsLeft.at(1).y - vPtsLeft.at(0).y - vPtsBack.at(0).y, nWidth, nHeight));
            mImgRoiInputRight = mCombine(cv::Rect(vPtsLeft.at(0).x + vPtsFront.at(1).x - vPtsFront.at(0).x - vPtsRight.at(0).x, vPtsFront.at(1).y - vPtsRight.at(0).y, nHeight, nWidth));
        }
    }
    else
    {
        if (nDiffBL_x <= 0)
        {
            mImgRoiInputFront = mCombine(cv::Rect(-nDiffBL_x + nDiffFL_x, 0, nWidth, nHeight));
            mImgRoiInputLeft = mCombine(cv::Rect(-nDiffBL_x, vPtsFront.at(0).y - vPtsLeft.at(0).y, nHeight, nWidth));
            mImgRoiInputBack = mCombine(cv::Rect(0, vPtsFront.at(0).y + vPtsLeft.at(1).y - vPtsLeft.at(0).y - vPtsBack.at(0).y, nWidth, nHeight));
            mImgRoiInputRight = mCombine(cv::Rect(vPtsLeft.at(0).x + vPtsFront.at(1).x - vPtsFront.at(0).x - vPtsRight.at(0).x - nDiffBL_x, vPtsFront.at(1).y - vPtsRight.at(0).y, nHeight, nWidth));
        }
        else
        {
            mImgRoiInputFront = mCombine(cv::Rect(nDiffFL_x, 0, nWidth, nHeight));
            mImgRoiInputLeft = mCombine(cv::Rect(0, vPtsLeft.at(0).y - vPtsFront.at(0).y, nHeight, nWidth));
            mImgRoiInputBack = mCombine(cv::Rect(nDiffBL_x, vPtsFront.at(0).y + vPtsLeft.at(1).y - vPtsLeft.at(0).y - vPtsBack.at(0).y, nWidth, nHeight));
            mImgRoiInputRight = mCombine(cv::Rect(vPtsLeft.at(0).x + vPtsFront.at(1).x - vPtsFront.at(0).x - vPtsRight.at(0).x, vPtsFront.at(1).y - vPtsRight.at(0).y, nHeight, nWidth));
        }
    }

    vContourInputBack.push_back(vStitchBack);
    drawContours(mRoiInputBack, vContourInputBack, 0, cv::Scalar::all(255), -1);

    vContourInputFront.push_back(vStitchFront);
    drawContours(mRoiInputFront, vContourInputFront, 0, cv::Scalar::all(255), -1);
    
    // 先放left与right, 因为没有切割. front 与back切割了
    mInputRight.copyTo(mImgRoiInputRight);
    mInputLeft.copyTo(mImgRoiInputLeft);
    mInputFront.copyTo(mImgRoiInputFront, mRoiInputFront);
    mInputBack.copyTo(mImgRoiInputBack, mRoiInputBack);
    //mCombine = mCombine(cv::Rect(0, 0, 900, 910));
    return mCombine;
}


extern "C" DLL_API SurroundView *GetStitching()
{
    return new Stitching360;
}