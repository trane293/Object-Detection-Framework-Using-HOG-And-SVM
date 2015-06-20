/*=====================================================================================
  =====================================================================================
   PROTOTYPE CODE TO TEST NEW FEATURES AND OTHER SHIT THAT I FEAR WILL BREAK EVERYTHING
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
string fileNameToWriteTo = "hard/";
static string testSamplesDir = "test/";
int c = 1;
// Directory containing negative sample images

const string useSVMModel = svmModelFile24_hn;

// HOG parameters for training that for some reason are not included in the HOG class
static const Size trainingPadding = Size(0, 0);
static const Size winStride = Size(8, 8);
static const Size windowSize = Size(24, 24);
/* Functions Start */

static string toLowerCase(const string& in) {
    string t;
    for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
        t += tolower(*i);
    }
    return t;
}

static void getFilesInDirectory(const string& dirName, vector<string>& fileNames, const vector<string>& validExtensions) {
    printf("Opening directory %s\n", dirName.c_str());
    struct dirent* ep;
    size_t extensionLocation;
    DIR* dp = opendir(dirName.c_str());
    if (dp != NULL) {
        while ((ep = readdir(dp))) {
            // Ignore (sub-)directories like . , .. , .svn, etc.
            if (ep->d_type & DT_DIR) {
                continue;
            }
            extensionLocation = string(ep->d_name).find_last_of("."); // Assume the last point marks beginning of extension like file.ext
            // Check if extension is matching the wanted ones
            string tempExt = toLowerCase(string(ep->d_name).substr(extensionLocation + 1));
            if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end()) {
                printf("Found matching data file '%s'\n", ep->d_name);
                fileNames.push_back((string) dirName + ep->d_name);
            } else {
                printf("Found file does not match required file type, skipping: '%s'\n", ep->d_name);
            }
        }
        (void) closedir(dp);
    } else {
        printf("Error opening directory '%s'!\n", dirName.c_str());
    }
    return;
}

static void showDetections(const vector<Rect>& found, Mat& imageData) {
    Point p(1, 1);
    vector<Rect> found_filtered;
    Mat imageDataCopy = imageData.clone();
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
        //cout<<r.x<<" "<<r.y<<" "<<r.width<<" "<<r.height<<endl<<endl;
        if (r.width > 24 && r.height > 24)  {
            Rect new_size(r);
            Rect imgBounds(0,0,imageDataCopy.cols,imageDataCopy.rows);
            new_size = new_size & imgBounds;
        // Now you can do the following without worrying (except in the case that new_size is empty!!)

            if (new_size.x != 0 && new_size.y != 0 && new_size.width >= 24 && new_size.height >= 24)  {
                Mat tmp = imageDataCopy(new_size);
                //Mat tempMat = imageData(p1,p2);
                string temp_file = fileNameToWriteTo + to_string(c) + ".png";
                resize(tmp, tmp, Size(24, 24));
                imwrite(temp_file, tmp);
                c++;
            }
        }
        //imshow("Image", imageData);
        rectangle(imageData, r.tl() - p, r.br() + p, Scalar(64, 255, 64), 1);
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

int main()  {

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

    static vector<string> testImages;
    static vector<string> validExtensions;
    validExtensions.push_back("jpg");
    validExtensions.push_back("png");
    validExtensions.push_back("ppm");

    float scale = 1.02;
    int minN = 22;
    
    VideoCapture capture("/home/anmol/lane2.avi");
    //VideoCapture capture(0);
    int count = 1;
    Mat frame;
    getFilesInDirectory(testSamplesDir, testImages, validExtensions);

    cout<<endl;
    
    while(1)    {
              
        vector<string>::iterator iter;
        
        static int i = 0;
        
        for (iter = testImages.begin();iter<testImages.end();iter++)    {
            // Read the image

            Mat imageToTest = imread(*iter, 1);
            resize(imageToTest, imageToTest, Size(560, 315)); //640x360, 480x270, Maintain aspect ratio of 1:7 for STS detection IJCNN dataset
            if (!imageToTest.data)
                cout<<"Error in opening image!";
            
            // Detect the traffic signs in the image and display the bounding boxes.
        
            detectTest(hog, hitThreshold, imageToTest, scale, minN);
            imshow("HOG custom detection", imageToTest);
            if (char(waitKey(1000)) == 'c') {

                cout<<"Current Parameters = "<<endl;
                cout<<"\tScaling Coefficient = "<<scale<<endl;
                cout<<"\tMinimum Neighbours  = "<<minN<<endl;
                cout<<"Enter New Parameters!"<<endl;
                cout<<"\tEnter Scale value ->";
                cin>>scale;
                cout<<"\tEnter Minimum Neighbours ->";
                cin>>minN;
            
            }
        }
    }
    return 0;
}





















//         if (count % 40 == 0) {
//         double t0 = getTickCount();
        
//         capture >> frame;
//         //frame = frame(Rect(0, 0, 640, 360));
//         resize(frame, frame, Size(512, 288)); //320x180 for 16:9, 512x288 // 640x480 for 4:3, 320x240
//         detectTest(hog, hitThreshold, frame, scale, minN);
//         imshow("HOG custom detection", frame);

//         if (char(waitKey(1)) == 'c') {
//             cout<<"Current Parameters = "<<endl;
//             cout<<"\tScaling Coefficient = "<<scale<<endl;
//             cout<<"\tMinimum Neighbours  = "<<minN<<endl;
//             cout<<"Enter New Parameters!"<<endl;
//             cout<<"\tEnter Scale value ->";
//             cin>>scale;
//             cout<<"\tEnter Minimum Neighbours ->";
//             cin>>minN;
//         }
//         double fps = getTickFrequency() / (getTickCount() - t0);
//         printf("\tFrame Rate: %2.2f \n", fps );
//         }
//         count++;
// }
    













