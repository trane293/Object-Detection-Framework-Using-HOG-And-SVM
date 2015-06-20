/*=====================================================================================
  =====================================================================================
                 MAIN FILE TO LOAD SVM MODEL FILE AND TEST DETECTIONS
  =====================================================================================
  =====================================================================================  
                                AUTHOR - Anmol Sharma
  =====================================================================================
  =====================================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string>
#include <ios>
#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>

#define SVMLIGHT 1
#define LIBSVM 2

//#define TRAINHOG_USEDSVM SVMLIGHT
#define TRAINHOG_USEDSVM SVMLIGHT

#if TRAINHOG_USEDSVM == SVMLIGHT
    #include "svmlight/svmlight.h"
    #define TRAINHOG_SVM_TO_TRAIN SVMlight
#elif TRAINHOG_USEDSVM == LIBSVM
    #include "libsvm/libsvm.h"
    #define TRAINHOG_SVM_TO_TRAIN libSVM
#endif

using namespace std;
using namespace cv;

static string svmModelFile24_hn = "genfiles/svmlight_24hn/svmlightmodel.dat";
static string svmModelFile64    = "genfiles/svmlight_64/svmlightmodel.dat";
static string svmModelFile24    = "genfiles/svmlight_24/svmlightmodel.dat";
static string svmTest           = "genfiles/svmlight_24hn_test/svmlightmodel.dat";
static string svmMoreHn         = "genfiles/svmlight_24morehn/svmlightmodel.dat";

const string useSVMModel = svmMoreHn;

// HOG parameters for training that for some reason are not included in the HOG class
static const Size trainingPadding = Size(0, 0);
static const Size winStride = Size(8, 8);
static const Size windowSize = Size(24, 24);
/* Functions Start */

static void showDetections(const vector<Rect>& found, Mat& imageData) {
    vector<Rect> found_filtered;
    size_t i, j;
    for (i = 0; i < found.size(); ++i) {
        Rect r = found[i];
        for (j = 0; j < found.size(); ++j)
            if (j != i && (r & found[j]) == r)
                break;
        if (j == found.size())
            found_filtered.push_back(r);
    }
    for (i = 0; i < found_filtered.size(); i++) {
        Rect r = found_filtered[i];
        rectangle(imageData, r.tl(), r.br(), Scalar(64, 255, 64), 3);
    }
}


static void detectTest(const HOGDescriptor& hog, const double hitThreshold, Mat& imageData, float scaleValue, int minNeighbours) {
    vector<Rect> found;
    Size padding(Size(32, 32));
    Size winStride(Size(8, 8));

    // Tweaked parameters can be 1.05 and 70
    hog.detectMultiScale(imageData, found, hitThreshold, winStride, padding, scaleValue, minNeighbours); 
    showDetections(found, imageData);
}

int main()	{

	TRAINHOG_SVM_TO_TRAIN::getInstance()->loadModelFromFile(useSVMModel);
	cout<<"Successfully Loaded Model File!"<<endl;
	
	printf("Generating Representative Single HOG Feature Vector using SVMlight! \n");
    vector<float> descriptorVector;
    vector<unsigned int> descriptorVectorIndices;
    // Generate a single detecting feature vector (v1 | b) from the trained support vectors, for use e.g. with the HOG algorithm
    TRAINHOG_SVM_TO_TRAIN::getInstance()->getSingleDetectingVector(descriptorVector, descriptorVectorIndices);
    const double hitThreshold = TRAINHOG_SVM_TO_TRAIN::getInstance()->getThreshold();

    if (descriptorVector.empty())
        cout<<"Descriptor Vector is empty!";

	HOGDescriptor hog; // Use standard parameters here
    hog.winSize = windowSize;
	
	hog.setSVMDetector(descriptorVector);
	VideoCapture capture(0);

	
	if( !capture.isOpened() )
	    throw "Error when reading steam";

    float scale = 1.01;
    int minN = 50;
    while ( char(waitKey(1) != 'q' && capture.isOpened()) ) {

        // getFilesInDirectory(testPosSamplesDir, positiveTestImages, validExtensions);
           
        double t0 = getTickCount();
        Mat frame;
        capture >> frame;
        resize(frame, frame, Size(512, 288));
        detectTest(hog, hitThreshold, frame, scale, minN);
        imshow("HOG custom detection", frame);

        if (char(waitKey(30)) == 'c') {
            cout<<"Current Parameters = "<<endl;
            cout<<"\tScaling Coefficient = "<<scale<<endl;
            cout<<"\tMinimum Neighbours  = "<<minN<<endl;
            cout<<"Enter New Parameters!"<<endl;
            cout<<"\tEnter Scale value ->";
            cin>>scale;
            cout<<"\tEnter Minimum Neighbours ->";
            cin>>minN;
        }
        double fps = getTickFrequency() / (getTickCount() - t0);
        printf("\tFrame Rate: %2.2f \n", fps );
    }
	return 0;
}
