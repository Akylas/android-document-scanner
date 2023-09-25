#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;

const double contoursApproxEpsilonFactor = 0.03;

bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2)
{
	double i = fabs(contourArea(cv::Mat(contour1)));
	double j = fabs(contourArea(cv::Mat(contour2)));
	return (i > j);
}

bool compareXCords(Point p1, Point p2)
{
	return (p1.x < p2.x);
}

bool compareYCords(Point p1, Point p2)
{
	return (p1.y < p2.y);
}

bool compareDistance(pair<Point, Point> p1, pair<Point, Point> p2)
{
	return (norm(p1.first - p1.second) < norm(p2.first - p2.second));
}

double _distance(Point p1, Point p2)
{
	return sqrt(((p1.x - p2.x) * (p1.x - p2.x)) +
				((p1.y - p2.y) * (p1.y - p2.y)));
}

void resizeToHeight(Mat src, Mat &dst, int height)
{
	Size s = Size(src.cols * (height / double(src.rows)), height);
	resize(src, dst, s, INTER_AREA);
}

void orderPoints(vector<Point> inpts, vector<Point> &ordered)
{
	sort(inpts.begin(), inpts.end(), compareXCords);
	vector<Point> lm(inpts.begin(), inpts.begin() + 2);
	vector<Point> rm(inpts.end() - 2, inpts.end());

	sort(lm.begin(), lm.end(), compareYCords);
	Point tl(lm[0]);
	Point bl(lm[1]);
	vector<pair<Point, Point>> tmp;
	for (size_t i = 0; i < rm.size(); i++)
	{
		tmp.push_back(make_pair(tl, rm[i]));
	}

	sort(tmp.begin(), tmp.end(), compareDistance);
	Point tr(tmp[0].second);
	Point br(tmp[1].second);

	ordered.push_back(tl);
	ordered.push_back(tr);
	ordered.push_back(br);
	ordered.push_back(bl);
}

void fourPointTransform(Mat src, Mat &dst, vector<Point> pts)
{
	vector<Point> ordered_pts;
	orderPoints(pts, ordered_pts);

	double wa = _distance(ordered_pts[2], ordered_pts[3]);
	double wb = _distance(ordered_pts[1], ordered_pts[0]);
	double mw = max(wa, wb);

	double ha = _distance(ordered_pts[1], ordered_pts[2]);
	double hb = _distance(ordered_pts[0], ordered_pts[3]);
	double mh = max(ha, hb);

	Point2f src_[] =
		{
			Point2f(ordered_pts[0].x, ordered_pts[0].y),
			Point2f(ordered_pts[1].x, ordered_pts[1].y),
			Point2f(ordered_pts[2].x, ordered_pts[2].y),
			Point2f(ordered_pts[3].x, ordered_pts[3].y),
		};
	Point2f dst_[] =
		{
			Point2f(0, 0),
			Point2f(mw - 1, 0),
			Point2f(mw - 1, mh - 1),
			Point2f(0, mh - 1)};
	Mat m = getPerspectiveTransform(src_, dst_);
	warpPerspective(src, dst, m, Size(mw, mh));
}

void preProcess(Mat src, Mat &dst)
{
	cv::Mat imageGrayed;
	cv::Mat imageOpen, imageClosed, imageBlurred;

	cvtColor(src, imageGrayed, COLOR_BGR2GRAY);

	GaussianBlur(imageGrayed, imageBlurred, Size(11, 11), 0);
	cv::Mat structuringElmt = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	morphologyEx(imageBlurred, imageClosed, cv::MORPH_CLOSE, structuringElmt);
	Canny(imageClosed, dst, 0, 200);

	structuringElmt = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
	dilate(dst, dst, structuringElmt);


	// cv::Mat structuringElmt = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4, 4));
	// morphologyEx(imageGrayed, imageOpen, cv::MORPH_OPEN, structuringElmt);
	// morphologyEx(imageOpen, imageClosed, cv::MORPH_CLOSE, structuringElmt);

	// GaussianBlur(imageClosed, imageBlurred, Size(7, 7), 0);
	// Canny(imageBlurred, dst, 75, 100);
}

string getOutputFileName(string path, string name)
{
	std::string fname, ext;

	size_t sep = path.find_last_of("\\/");
	if (sep != std::string::npos)
	{
		path = path.substr(sep + 1, path.size() - sep - 1);

		size_t dot = path.find_last_of(".");
		if (dot != std::string::npos)
		{
			fname = path.substr(0, dot);
			ext = path.substr(dot, path.size() - dot);
		}
		else
		{
			fname = path;
			ext = "";
		}
	}

	return fname + "_" + name + ext;
}

double getContrastValue(int n)
{
	switch (n)
	{
	case 1:
		return 1.5;
	case 2:
		return 1.4;
	case 3:
		return 1.3;
	case 4:
		return 1.25;
	case 5:
		return 1.2;
	}
	return 1.3;
}

double getContrastLevel(Mat srcArry)
{
	const float range[] = {0, 256}; // the upper boundary is exclusive
	const int histSize = 256;
	const float *histRange[] = {range};

	int sum = 0;
	int brightnessVal[5];
	int ans = -1;
	int max = std::numeric_limits<int>::min();
	if (srcArry.channels() >= 3)
	{
		// cvtColor(srcArry, srcArry, COLOR_BGR2HSV);

		vector<Mat> bgrPlanes;
		split(srcArry, bgrPlanes);

		// calculating histogram of only V channel
		bool uniform = true, accumulate = false;
		Mat valueHist;
		calcHist(&bgrPlanes[2], 1, 0, Mat(), valueHist, 1, &histSize, histRange, uniform, accumulate);
		// calcHist(bgrPlanes, 1, channels, Mat(), valueHist, histSizeDims, histSize, histRanges, uniform, accumulate);

		int histW = 512, histH = 400;

		Mat histImage(histH, histW, CV_8UC3, {0, 0, 0});
		normalize(valueHist, valueHist, 0, histImage.rows, NORM_MINMAX);

		// the following loop analyzes the histogram and looks for the region in graph with highest concentrated
		// peaks and returns that region as integer.
		// this histogram is divided into five regions.
		for (int i = 1; i < 256; i++)
		{

			int p = std::round(valueHist.at<float>(i));
			int divideBy = 51;

			if (i < 52)
			{
				sum += p;
				if (i == 51)
				{
					brightnessVal[0] = sum / divideBy;
					sum = 0;
				}
			}
			else if (i < 103)
			{
				sum += p;
				if (i == 102)
				{
					brightnessVal[1] = sum / divideBy;
					sum = 0;
				}
			}
			else if (i < 154)
			{
				sum += p;
				if (i == 153)
				{
					brightnessVal[2] = sum / divideBy;
					sum = 0;
				}
			}
			else if (i < 204)
			{
				sum += p;
				if (i == 203)
				{
					brightnessVal[3] = sum / divideBy;
					sum = 0;
				}
			}
			else
			{
				sum += p;
				if (i == 255)
				{
					brightnessVal[4] = sum / divideBy;
					sum = 0;
				}
			}
		}

		for (int i = 0; i < 5; i++)
		{
			if (max < brightnessVal[i])
			{
				max = brightnessVal[i];
				ans = i;
			}
		}
	}
	return getContrastValue(ans);
}

void removeShadows(Mat src, Mat &dst)
{
	cvtColor(src, dst, COLOR_BGR2HSV);
	vector<Mat> bgrPlanes;
	vector<Mat> result;
	vector<Mat> list;
	split(dst, bgrPlanes);
	// processing the V channel for shadow removal
	cv::Mat zero = cv::Mat::zeros(dst.size(), CV_8UC1);
	Mat kernel = Mat::ones(7, 7, CV_32F);
	list.push_back(bgrPlanes[2]);	// adding the V channel for processing in list
	result.push_back(bgrPlanes[0]); // adding H channel in result
	result.push_back(bgrPlanes[1]); // adding S channel in result
	for (auto mat : list)
	{
		Mat dilated_img;
		dilate(mat, dilated_img, kernel);
		medianBlur(dilated_img, dilated_img, 21);
		Mat diff;
		absdiff(mat, dilated_img, diff);
		bitwise_not(diff, diff);
		Mat norm;
		normalize(diff, norm, 0, 255, NORM_MINMAX, CV_8UC1, noArray());
		result.push_back(norm); // completely processed --> adding V channel into result
		dilated_img.release();
		diff.release();
	}

	merge(result, dst);

	cvtColor(dst, dst, COLOR_HSV2BGR); // converting image back to RGB
}
void applyContrastFilter(Mat src, Mat &dst)
{
	cvtColor(src, dst, COLOR_BGR2HSV);
	double contrast_value = getContrastLevel(dst);
	Mat channel;
	// cvtColor(warped, warped, COLOR_BGR2HSV);
	extractChannel(dst, channel, 2);
	cv::Ptr<cv::CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(1);
	// apply the CLAHE algorithm to the L channel
	clahe->apply(channel, channel);
	// Merge the color planes back into an HSV image
	insertChannel(channel, dst, 2);

	// Extract the S channel
	extractChannel(dst, channel, 1);
	// apply the CLAHE algorithm to the S channel
	// clahe = createCLAHE();
	// clahe->setClipLimit(1);
	// apply the CLAHE algorithm to the L channel
	clahe->apply(channel, channel);
	// Merge the color planes back into an HSV image
	insertChannel(channel, dst, 1);

	cvtColor(dst, dst, COLOR_HSV2BGR);
	dst.convertTo(dst, -1, contrast_value * 0.9, 29);
	// cvtColor(warpedHSV, warpedHSV, COLOR_BGR2GRAY);

	channel.release();
}

// int main(int argc, char **argv) {
// 	static const char *const keys = "{ @image |<none>| }";
// 	CommandLineParser parser(argc, argv, keys);

// 	if (!parser.has("@image"))
// 	{
// 		parser.printMessage();
// 		return -1;
// 	}

// 	printf("OpenCV version: %s\n", CV_VERSION);

// 	string image_name(parser.get<String>("@image"));
// 	Mat image = imread(image_name);
// 	if (image.empty())
// 	{
// 		printf("Cannot read image file: %s\n", image_name.c_str());
// 		return -1;
// 	}

// 	double ratio = image.rows / 500.0;
// 	Mat orig = image.clone();
// 	resizeToHeight(image, image, 500);

// 	Mat gray, edged, warped;
// 	preProcess(image, edged);
// #ifndef NDEBUG
// 	imwrite(getOutputFileName(image_name, "edged"), edged);
// #endif

// 	vector<vector<Point>> contours;
// 	vector<Vec4i> hierarchy;
// 	vector<vector<Point>> approx;
// 	findContours(edged, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

// 	approx.resize(contours.size());
// 	size_t i, j;
// 	for (i = 0; i < contours.size(); i++)
// 	{
// 		double peri = arcLength(contours[i], true);
// 		approxPolyDP(contours[i], approx[i], 0.02 * peri, true);
// 	}
// 	sort(approx.begin(), approx.end(), compareContourAreas);

// 	for (i = 0; i < approx.size(); i++)
// 	{
// 		drawContours(image, approx, i, Scalar(255, 255, 0), 2);
// 		if (approx[i].size() == 4)
// 		{
// 			break;
// 		}
// 	}

// 	if (i < approx.size())
// 	{
// 		drawContours(image, approx, i, Scalar(0, 255, 0), 2);
// #ifndef NDEBUG
// 		imwrite(getOutputFileName(image_name, "outline"), image);
// #endif
// 		for (j = 0; j < approx[i].size(); j++)
// 		{
// 			approx[i][j] *= ratio;
// 		}

// 		fourPointTransform(orig, warped, approx[i]);
// #ifndef NDEBUG
// 		imwrite(getOutputFileName(image_name, "flat"), warped);
// #endif
// 		// cvtColor(warped, warped, COLOR_BGR2GRAY, 1);
// 		// adaptiveThreshold(warped, warped, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 15);
// 		// GaussianBlur(warped, warped, Size(3, 3), 0);
// 		// imwrite(getOutputFileName(image_name, "scanned"), warped);
// 		Mat warpedHSV;
// 		removeShadows(warped, warpedHSV);
// 		// applyContrastFilter(warped, warpedHSV);
// 		// cvtColor(warpedHSV, warpedHSV, COLOR_HSV2BGR); // converting image back to RGB

// 		cvtColor(warpedHSV, warpedHSV, COLOR_BGR2GRAY, 1);
// 		imwrite(getOutputFileName(image_name, "scanned"), warpedHSV);
// 		imshow("Display window", warpedHSV);
// 		int k = cv::waitKey(0); // Wait for a keystroke in the window

// 		orig.release();
// 		gray.release();
// 		image.release();
// 		edged.release();
// 		warped.release();
// 		warpedHSV.release();
// 	}
// }

void testProcess(Mat &image, Mat &edged, Mat &warped)
{
	Mat gray;
	double ratio = image.rows / 300.0;
	Mat orig = image.clone();
	resizeToHeight(image, image, 300);

	preProcess(image, edged);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<vector<Point>> approx;
	findContours(edged, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

	approx.resize(contours.size());
	size_t i, j;
	for (i = 0; i < contours.size(); i++)
	{
		double peri = arcLength(contours[i], true);
		approxPolyDP(contours[i], approx[i], 0.02 * peri, true);
	}
	sort(approx.begin(), approx.end(), compareContourAreas);

	for (i = 0; i < approx.size(); i++)
	{
		// drawContours(image, approx, i, Scalar(255, 255, 0), 2);
		if (approx[i].size() == 4)
		{
			break;
		}
	}

	if (i < approx.size())
	{
		drawContours(image, approx, i, Scalar(0, 255, 0), 2);
		for (j = 0; j < approx[i].size(); j++)
		{
			approx[i][j] *= ratio;
		}
		fourPointTransform(orig, warped, approx[i]);
	}

	orig.release();
	gray.release();
}
double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

void findCannySquares(
	cv::Mat srcGray,
	double scaledWidth,
	double scaledHeight,
	std::vector<std::vector<cv::Point>> &cannySquares,
	int indice,
	std::vector<int> &indices)
{
	// contours search
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(srcGray, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	std::vector<Point> approx;
	for (int i = 0; i < contours.size(); i++)
	{
		std::vector<Point> contour = contours[i];
		// contour.fromArray(contours[i].data(), contours[i].size());
		// detection of geometric shapes
		cv::approxPolyDP(contour, approx, cv::arcLength(contour, true) * contoursApproxEpsilonFactor, true);
		// std::vector<Point> approx1f;
		// detection of quadrilaterals among geometric shapes
		if (approx.size() == 4 && std::abs(cv::contourArea(approx)) > scaledWidth / 5 * (scaledHeight / 5) && cv::isContourConvex(approx))
		{
			double maxCosine = 0.0;
			for (int j = 2; j <= 4; j++)
			{
				double cosine = std::abs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
				maxCosine = std::max(maxCosine, cosine);
			}
			// selection of quadrilaterals with large enough angles
			if (maxCosine < 0.5)
			{
				// indices.push_back(cannySquares.size());
				cannySquares.push_back(approx);
			}
		}
	}
}

void findSquares(cv::Mat srcGray, double scaledWidth, double scaledHeight, std::vector<std::pair<std::vector<cv::Point>, double>> &squares)
{
	// Contours search
	std::vector<std::vector<cv::Point>> contours;
	vector<Vec4i> hierarchy;
	cv::findContours(srcGray, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	std::vector<Point> approx;
	for (size_t i = 0; i < contours.size(); i++)
	{
		std::vector<Point> contour = contours[i];

		// Detection of geometric shapes
		double epsilon = cv::arcLength(contour, true) * contoursApproxEpsilonFactor;
		cv::approxPolyDP(contour, approx, epsilon, true);

		// Detection of quadrilaterals among geometric shapes
		if (approx.size() == 4 && cv::isContourConvex(approx)) {
			const double area = std::abs(contourArea(approx));
			if (area > scaledWidth / 5 * (scaledHeight / 5) )
			{
				double maxCosine = 0.0;
				for (int j = 2; j < 5; j++)
				{
					double cosine = std::abs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
					maxCosine = std::max(maxCosine, cosine);
				}
				// Selection of quadrilaterals with large enough angles
				if (maxCosine < 0.5)
				{
					squares.push_back(std::pair<std::vector<cv::Point>, double>(approx, area));
				}
			}
		}
	}
}
void testProcess2(Mat &image, Mat &edged, Mat &warped)
{
	double width;
	double height;


	double ratio = image.rows / 500.0;
	Mat orig = image.clone();
	resizeToHeight(image, image, 500);
	cv::Size size = image.size();
	width = size.width;
	height = size.height;

    // convert photo to LUV colorspace to avoid glares caused by lights
    // cvtColor(image, image, COLOR_BGR2Luv);

	cv::Mat blurred;

	cv::medianBlur(image, blurred, 9);
	cv::Mat threshOutput = cv::Mat(blurred.size(), CV_8U);
	std::vector<std::vector<cv::Point>> squares;
	std::vector<std::pair<std::vector<cv::Point>, double>> threshSquares;
	std::vector<std::pair<std::vector<cv::Point>, double>> cannySquares;
	std::vector<int> indices;

	for (int c = 2; c >= 0; c--)
	{
		Mat lBlurred[] = {blurred};
		Mat lOutput[] = {threshOutput};
		int ch[] = {c, 0};
		cv::mixChannels(lBlurred, 1, lOutput, 1, ch, 1);

		int thresholdLevel = 3;

		int t = 60;
		// for (int l = 0 ; l < thresholdLevel; l++)
		for (int l = thresholdLevel-1 ; l >= 0; l--)
		{
			// if (l == 0)
			// {	
			// 	t = 60;
			// 	while (t >= 10)
			// 	{
			// 		cv::Canny(threshOutput, edged, t, t * 2);
			// 		cv::dilate(edged, edged, cv::Mat(), cv::Point(-1, -1), 2);
			// 		findSquares(
			// 			edged,
			// 			width,
			// 			height,
			// 			cannySquares);
			// 		if (cannySquares.size() > 0)
			// 		{
			// 			break;
			// 		}
			// 		// Call findCannySquares here with appropriate parameters
			// 		t -= 10;
			// 	}
			// }
			// else
			// {
				cv::threshold(threshOutput, edged, (200 - 175 / (l + 2.0)), 256.0, cv::THRESH_BINARY);
				findSquares(edged, width, height, threshSquares);
				if (threshSquares.size() > 0)
					{
						break;
					}
				// Call findThreshSquares here with appropriate parameters
			// }

			if (cannySquares.size() > 0 || threshSquares.size() > 0)
			{
				// stop as soon as find some
				break;
			}
		}
	}

	int indiceMax = -1;
	double maxArea = -1.0;
	for (size_t i = 0; i < cannySquares.size(); i++)
	{
		double area = cannySquares[i].second;
		if (area > maxArea && area < width * height)
		{
			maxArea = area;
			indiceMax = static_cast<int>(i);
		}
	}

	if (indiceMax != -1)
	{
		squares.push_back(cannySquares[indiceMax].first);
	}

	int marge = static_cast<int>(width * 0.01);
	std::vector<std::pair<std::vector<cv::Point>, double>> squaresProba;
	bool probable;

	for (size_t i = 0; i < threshSquares.size(); i++)
	{
		probable = true;
		std::vector<cv::Point> pointsProba = threshSquares[i].first;
		for (const cv::Point &p : pointsProba)
		{
			if (p.x < marge || p.x >= width - marge || p.y < marge || p.y >= height - marge)
			{
				probable = false;
				break;
			}
		}
		if (probable)
		{
			squaresProba.push_back(std::pair<std::vector<cv::Point>, double>(pointsProba,  threshSquares[i].second));
		}
	}

	int largestContourIndex = 0;

	if (!squaresProba.empty())
	{
		double largestArea = -1.0;
		for (size_t i = 0; i < squaresProba.size(); i++)
		{
			double a = squaresProba[i].second;
			if (a > largestArea && a < width * height)
			{
				largestArea = a;
				largestContourIndex = static_cast<int>(i);
			}
		}
		squares.push_back(squaresProba[largestContourIndex].first);
	}
	else
	{
		double largestArea = -1.0;
		for (size_t i = 0; i < threshSquares.size(); i++)
		{
			double a = threshSquares[i].second;
			if (a > largestArea && a < width * height)
			{
				largestArea = a;
				largestContourIndex = static_cast<int>(i);
			}
		}
		if (!threshSquares.empty())
		{
			squares.push_back(threshSquares[largestContourIndex].first);
		}
		// else
		// {
		// 	std::vector<cv::Point> pts = {cv::Point(0, 0), cv::Point(width, 0), cv::Point(0, height), cv::Point(width, height)};
		// 	squares.push_back(pts);
		// }
	}

	for (int id : indices)
	{
		if (id != indiceMax)
		{
			squares.push_back(cannySquares[id].first);
		}
	}

	for (size_t id = 0; id < threshSquares.size(); id++)
	{
		if (static_cast<int>(id) != largestContourIndex)
		{
			squares.push_back(threshSquares[id].first);
		}
	}

	for (size_t id = 0; id < squares.size(); id++)
	{
		drawContours(image, squares, id, Scalar(0, 255, 0), 2);
		if (id == 0)
		{
			std::vector<Point> square = squares[id];
			for (size_t j = 0; j < square.size(); j++)
			{
				square[j] *= ratio;
			}
			fourPointTransform(orig, warped, square);
		}
	}
	blurred.release();
	// std::vector<cv::Point> unsorted = squares[0];
	// std::vector<std::vector<cv::Point>> contours;
	// contours.push_back(getOrderedPoints(unsorted));
}

int main(int argc, char **argv)
{
	Mat image;					 // Declaring a matrix to load the frames//
	namedWindow("Video Player"); // Declaring the video to show the video//
	VideoCapture cap(0);		 // Declaring an object to capture stream of frames from default camera//
	if (!cap.isOpened())
	{ // This section prompt an error message if no video stream is found//
		cout << "No video stream detected" << endl;
		system("pause");
		return -1;
	}
	bool useMorph = false;
	while (true)
	{ // Taking an everlasting loop to show the video//
		cap >> image;
		if (image.empty())
		{ // Breaking the loop if no video frame is detected//
			break;
		}
		if (useMorph) {
			cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    		cv::morphologyEx(image, image, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 3);
		}


		Mat edged, warped;
		testProcess(image, edged, warped);

		imshow("Video Player", image);
		imshow("Edges", edged);
		if (!warped.empty())
		{
			imshow("Warped", warped);
		}
		edged.release();
		warped.release();
		char c = (char)waitKey(25); // Allowing 25 milliseconds frame processing time and initiating break condition//
		if (c == 27)
		{ // If 'Esc' is entered break the loop//
			break;
		}else if (c == 32) {
			useMorph = !useMorph;
		}
	}
	cap.release(); // Releasing the buffer memory//
	return 0;
}
