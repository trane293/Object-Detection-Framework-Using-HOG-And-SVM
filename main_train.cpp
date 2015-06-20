/*=====================================================================================
  =====================================================================================
           CODE TO TRAIN THE SVM DETECTOR AND SAVE THE MODEL FILE TO GENFILES
  =====================================================================================
  =====================================================================================  
                                AUTHOR - Anmol Sharma
  =====================================================================================
  =====================================================================================
*/

#include <stdio.h>
#include <dirent.h>
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


// Directory containing positive sample images
static string posSamplesDir = "pos/";
// Directory containing negative sample images
static string negSamplesDir = "neg(more_hard_negs)(plus_earliers)/";

/* File Paths  */
static string featuresFile = "genfiles/svmlight_24morehn/features.dat";
static string descriptorVectorFile = "genfiles/svmlight_24morehn/descriptorvector.dat";
static string svmModelFile = "genfiles/svmlight_24morehn/svmlightmodel.dat";

// HOG parameters for training that for some reason are not included in the HOG class
static const Size trainingPadding = Size(0, 0);
static const Size winStride = Size(8, 8);
static const Size windowSize = Size(24, 24);
// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="Helper functions">
/* Helper functions */

static string toLowerCase(const string& in) {
    string t;
    for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
        t += tolower(*i);
    }
    return t;
}

static void storeCursor(void) {
    printf("\033[s");
}

static void resetCursor(void) {
    printf("\033[u");
}

/**
 * Saves the given descriptor vector to a file
 * @param descriptorVector the descriptor vector to save
 * @param _vectorIndices contains indices for the corresponding vector values (e.g. descriptorVector(0)=3.5f may have index 1)
 * @param fileName
 * @TODO Use _vectorIndices to write correct indices
 */
static void saveDescriptorVectorToFile(vector<float>& descriptorVector, vector<unsigned int>& _vectorIndices, string fileName) {
    printf("Saving descriptor vector to file '%s'\n", fileName.c_str());
    string separator = " "; // Use blank as default separator between single features
    fstream File;
    float percent;
    File.open(fileName.c_str(), ios::out);
    if (File.good() && File.is_open()) {
        printf("Saving %lu descriptor vector features:\t", descriptorVector.size());
        storeCursor();
        for (int feature = 0; feature < descriptorVector.size(); ++feature) {
            if ((feature % 10 == 0) || (feature == (descriptorVector.size()-1)) ) {
                percent = ((1 + feature) * 100 / descriptorVector.size());
                printf("%4u (%3.0f%%)", feature, percent);
                fflush(stdout);
                resetCursor();
            }
            File << descriptorVector.at(feature) << separator;
        }
        printf("\n");
        File << endl;
        File.flush();
        File.close();
    }
}

/**
 * For unixoid systems only: Lists all files in a given directory and returns a vector of path+name in string format
 * @param dirName
 * @param fileNames found file names in specified directory
 * @param validExtensions containing the valid file extensions for collection in lower case
 */
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

/**
 * This is the actual calculation from the (input) image data to the HOG descriptor/feature vector using the hog.compute() function
 * @param imageFilename file path of the image file to read and calculate feature vector from
 * @param descriptorVector the returned calculated feature vector<float> , 
 *      I can't comprehend why openCV implementation returns std::vector<float> instead of cv::MatExpr_<float> (e.g. Mat<float>)
 * @param hog HOGDescriptor containin HOG settings
 */
static void calculateFeaturesFromInput(const string& imageFilename, vector<float>& featureVector, HOGDescriptor& hog) {
    /** for imread flags from openCV documentation, 
     * @see http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#Mat imread(const string& filename, int flags)
     * @note If you get a compile-time error complaining about following line (esp. imread),
     * you either do not have a current openCV version (>2.0) 
     * or the linking order is incorrect, try g++ -o openCVHogTrainer main.cpp `pkg-config --cflags --libs opencv`
     */
    Mat imageData = imread(imageFilename, 0);
    if (imageData.empty()) {
        featureVector.clear();
        printf("Error: HOG image '%s' is empty, features calculation skipped!\n", imageFilename.c_str());
        return;
    }
    // Check for mismatching dimensions
    if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height) {
        featureVector.clear();
        printf("Error: Image '%s' dimensions (%u x %u) do not match HOG window size (%u x %u)!\n", imageFilename.c_str(), imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
        return;
    }
    vector<Point> locations;
    hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
    imageData.release(); // Release the image again after features are extracted
}

/**
 * Shows the detections in the image
 * @param found vector containing valid detection rectangles
 * @param imageData the image in which the detections are drawn
 */
static void showDetections(const vector<Point>& found, Mat& imageData) {
    size_t i, j;
    for (i = 0; i < found.size(); ++i) {
        Point r = found[i];
        // Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
        rectangle(imageData, Rect(r.x-16, r.y-32, 32, 64), Scalar(64, 255, 64), 3);
    }
}

/**
 * Shows the detections in the image
 * @param found vector containing valid detection rectangles
 * @param imageData the image in which the detections are drawn
 */

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

/**
 * Test the trained detector against the same training set to get an approximate idea of the detector.
 * Warning: This does not allow any statement about detection quality, as the detector might be overfitting.
 * Detector quality must be determined using an independent test set.
 * @param hog
 */
static void detectTrainingSetTest(const HOGDescriptor& hog, const double hitThreshold, const vector<string>& posFileNames, const vector<string>& negFileNames) {
    unsigned int truePositives = 0;
    unsigned int trueNegatives = 0;
    unsigned int falsePositives = 0;
    unsigned int falseNegatives = 0;
    vector<Point> foundDetection;
    // Walk over positive training samples, generate images and detect
    for (vector<string>::const_iterator posTrainingIterator = posFileNames.begin(); posTrainingIterator != posFileNames.end(); ++posTrainingIterator) {
        const Mat imageData = imread(*posTrainingIterator, 0);
        hog.detect(imageData, foundDetection, hitThreshold, winStride, trainingPadding);
        if (foundDetection.size() > 0) {
            ++truePositives;
            falseNegatives += foundDetection.size() - 1;
        } else {
            ++falseNegatives;
        }
    }
    // Walk over negative training samples, generate images and detect
    for (vector<string>::const_iterator negTrainingIterator = negFileNames.begin(); negTrainingIterator != negFileNames.end(); ++negTrainingIterator) {
        const Mat imageData = imread(*negTrainingIterator, 0);
        hog.detect(imageData, foundDetection, hitThreshold, winStride, trainingPadding);
        if (foundDetection.size() > 0) {
            falsePositives += foundDetection.size();
        } else {
            ++trueNegatives;
        }        
    }
    
    printf("Results:\n\tTrue Positives: %u\n\tTrue Negatives: %u\n\tFalse Positives: %u\n\tFalse Negatives: %u\n", truePositives, trueNegatives, falsePositives, falseNegatives);
    float Accuracy = (truePositives + trueNegatives)*100/(trueNegatives + truePositives + falsePositives + falseNegatives);
    float falsePositiveRate = falsePositives*100/(trueNegatives + falsePositives);
    float falseNegativeRate = falseNegatives*100/(falseNegatives + truePositives);
    printf("\tAccuracy = %2.2f %% \n \tFalse Positive Rate = %2.2f %% \n \tFalse Negative Rate = %2.2f %% \n", Accuracy, falsePositiveRate, falseNegativeRate );
}

/**
 * Test detection with custom HOG description vector
 * @param hog
 * @param hitThreshold threshold value for detection
 * @param imageData
 */
static void detectTest(const HOGDescriptor& hog, const double hitThreshold, Mat& imageData, float scaleValue, int minNeighbours) {
    vector<Rect> found;
    Size padding(Size(32, 32));
    Size winStride(Size(8, 8));

    // Tweaked parameters can be 1.05 and 70
    hog.detectMultiScale(imageData, found, hitThreshold, winStride, padding, scaleValue, minNeighbours); 
    showDetections(found, imageData);
}
// </editor-fold>

/**
 * Main program entry point
 * @param argc unused
 * @param argv unused
 * @return EXIT_SUCCESS (0) or EXIT_FAILURE (1)
 */
int main(int argc, char** argv) {

    // <editor-fold defaultstate="collapsed" desc="Init">
    HOGDescriptor hog; // Use standard parameters here
    hog.winSize = windowSize; // Default training images size as used in paper = 64*128. Changed to 24*24 ANMOL
    
/*
    win_size – Detection window size. Align to block size and block stride.
block_size – Block size in pixels. Align to cell size. Only (16,16) is supported for now.
block_stride – Block stride. It must be a multiple of cell size.
cell_size – Cell size. Only (8, 8) is supported for now.
nbins – Number of bins. Only 9 bins per cell are supported for now.
win_sigma – Gaussian smoothing window parameter.
threshold_L2hys – L2-Hys normalization method shrinkage.
gamma_correction – Flag to specify whether the gamma correction preprocessing is required or not.
nlevels – Maximum number of detection window increases.
*/

    // Get the files to train from somewhere
    static vector<string> positiveTrainingImages;
    static vector<string> negativeTrainingImages;
    static vector<string> validExtensions;
    validExtensions.push_back("jpg");
    validExtensions.push_back("png");
    validExtensions.push_back("ppm");
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Read image files">
    getFilesInDirectory(posSamplesDir, positiveTrainingImages, validExtensions);
    getFilesInDirectory(negSamplesDir, negativeTrainingImages, validExtensions);
    /// Retrieve the descriptor vectors from the samples
    unsigned long overallSamples = positiveTrainingImages.size() + negativeTrainingImages.size();
    // </editor-fold>
    
    // <editor-fold defaultstate="collapsed" desc="Calculate HOG features and save to file">
    // Make sure there are actually samples to train
    if (overallSamples == 0) {
        printf("No training sample files found, nothing to do!\n");
        return EXIT_SUCCESS;
    }

    /// @WARNING: This is really important, some libraries (e.g. ROS) seems to set the system locale which takes decimal commata instead of points which causes the file input parsing to fail
    setlocale(LC_ALL, "C"); // Do not use the system locale
    setlocale(LC_NUMERIC,"C");
    setlocale(LC_ALL, "POSIX");

    printf("Reading files, generating HOG features and save them to file '%s':\n", featuresFile.c_str());
    float percent;
    /**
     * Save the calculated descriptor vectors to a file in a format that can be used by SVMlight for training
     * @NOTE: If you split these steps into separate steps: 
     * 1. calculating features into memory (e.g. into a cv::Mat or vector< vector<float> >), 
     * 2. saving features to file / directly inject from memory to machine learning algorithm,
     * the program may consume a considerable amount of main memory
     */ 
    fstream File;
    File.open(featuresFile.c_str(), ios::out);
    if (File.good() && File.is_open()) {
		// Remove following line for libsvm which does not support comments
        // File << "# Use this file to train, e.g. SVMlight by issuing $ svm_learn -i 1 -a weights.txt " << featuresFile.c_str() << endl;
        // Iterate over sample images
        for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
            storeCursor();
            vector<float> featureVector;
            // Get positive or negative sample image file path
            const string currentImageFile = (currentFile < positiveTrainingImages.size() ? positiveTrainingImages.at(currentFile) : negativeTrainingImages.at(currentFile - positiveTrainingImages.size()));
            // Output progress
            if ( (currentFile+1) % 10 == 0 || (currentFile+1) == overallSamples ) {
                percent = ((currentFile+1) * 100 / overallSamples);
                printf("%5lu (%3.0f%%):\tFile '%s'", (currentFile+1), percent, currentImageFile.c_str());
                fflush(stdout);
                resetCursor();
            }
            // Calculate feature vector from current image file
            calculateFeaturesFromInput(currentImageFile, featureVector, hog);
            if (!featureVector.empty()) {
                /* Put positive or negative sample class to file, 
                 * true=positive, false=negative, 
                 * and convert positive class to +1 and negative class to -1 for SVMlight
                 */
                File << ((currentFile < positiveTrainingImages.size()) ? "+1" : "-1");
                // Save feature vector components
                for (unsigned int feature = 0; feature < featureVector.size(); ++feature) {
                    File << " " << (feature + 1) << ":" << featureVector.at(feature);
                }
                File << endl;
            }
        }
        printf("\n");
        File.flush();
        File.close();
    } else {
        printf("Error opening file '%s'!\n", featuresFile.c_str());
        return EXIT_FAILURE;
    }
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Pass features to machine learning algorithm">
    /// Read in and train the calculated feature vectors
    printf("Calling %s\n", TRAINHOG_SVM_TO_TRAIN::getInstance()->getSVMName());
    TRAINHOG_SVM_TO_TRAIN::getInstance()->read_problem(const_cast<char*> (featuresFile.c_str()));
    TRAINHOG_SVM_TO_TRAIN::getInstance()->train(); // Call the core libsvm training procedure
    printf("Training done, saving model file!\n");
    TRAINHOG_SVM_TO_TRAIN::getInstance()->saveModelToFile(svmModelFile);
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Generate single detecting feature vector from calculated SVM support vectors and SVM model">
    printf("Generating representative single HOG feature vector using svmlight!\n");
    vector<float> descriptorVector;
    vector<unsigned int> descriptorVectorIndices;
    // Generate a single detecting feature vector (v1 | b) from the trained support vectors, for use e.g. with the HOG algorithm
    TRAINHOG_SVM_TO_TRAIN::getInstance()->getSingleDetectingVector(descriptorVector, descriptorVectorIndices);
    // And save the precious to file system
    saveDescriptorVectorToFile(descriptorVector, descriptorVectorIndices, descriptorVectorFile);
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Test detecting vector">
    // Detector detection tolerance threshold
    const double hitThreshold = TRAINHOG_SVM_TO_TRAIN::getInstance()->getThreshold();
    // Set our custom detecting vector
    hog.setSVMDetector(descriptorVector);
    
    printf("Testing training phase using training set as test set (just to check if training is ok - no detection quality conclusion with this!)\n");
    detectTrainingSetTest(hog, hitThreshold, positiveTrainingImages, negativeTrainingImages);

    vector<string> positiveTestImages;
    vector<string> negativeTestImages;


    VideoCapture cap(0); // open the default camera

    if(!cap.isOpened()) // check if we succeeded
        return -1;
    float scale = 1.01;
    int minN = 40;
    
    while ( char(waitKey(1) != 'q' && cap.isOpened())) {

        // getFilesInDirectory(testPosSamplesDir, positiveTestImages, validExtensions);
           
        double t0 = getTickCount();
        Mat frame;
        cap >> frame;
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
        printf("Frame Rate: %2.2f \n", fps );
    }
        // getFilesInDirectory(testPosSamplesDir, positiveTestImages, validExtensions);
        // getFilesInDirectory(testNegSamplesDir, negativeTestImages, validExtensions);
        // cout<<endl;
        // vector<string>::iterator iter;
        
        // static int i = 0;
        
        // for (iter = positiveTestImages.begin();iter<positiveTestImages.end();iter++)    {
        //     // Read the image

        //     Mat testImage = imread(*iter, 1);
        //     resize(testImage, testImage, Size(400, 400)); 
        //     if (!testImage.data)
        //         cout<<"Error in opening image!";
            
        //     // Detect the traffic signs in the image and display the bounding boxes.
        
        //     //namedWindow("Opened File!");
        //     //imshow("Opened File!", image);
        //     waitKey(0);

    // testPosSamplesDir = "test/pos" < a string>
    // positiveTestImages = vector of strings, each strng containing file name
    // I need a pathToImage = string containing testPosSamplesDir and positiveTestImages

    // For positive test set
       
        // Now we have directory name in testPosSamplesDir and image names in positiveTestImages
        
        // I need to find a way to open these files in a Mat object and then pass it to detectTest function
    return EXIT_SUCCESS;
}




        // for(int i = 1;i<=positiveTestImages.size();i++)    {
        //     vector<string>::iterator it;
        //     vector<string> fileNameToRead;
    
        //     fileNameToRead.push_back(testPosSamplesDir);
        //     fileNameToRead.push_back(*it);//iterator to filename vector);
        //     imread(//iterator to filenametoread)
        //     // Test the positive images
        //     printf("=======================================");
        //     for (it=positiveTestImages.begin(); it<positiveTestImages.end(); it++)  {
        //         char test = testPosSamplesDir.append(*it);
        //         printf("%s", test);
        //     }
        // }





            // for (it=negativeTestImages.begin(); it<negativeTestImages.end(); it++)  {
            //     std::cout << ' ' << *it;
            //     std::cout << '\n';
            // }
            // Use imread to read the positive test sample
            // Then use the below function to detect

            //detectTest(hog, hitThreshold, positiveTestImage);

        // for(int i = 1;i<=negativeTestImages.size();i++)    {
        //     // Test the negative images
        //     // Use imread to read to read the negative test sample
        //     // Then use the below function to detect

        //     //detectTest(hog, hitThreshold, testImage);
        // }


        //getFilesInDirectory(testPosSamplesDir, positiveTestImages, validExtensions);
        //getFilesInDirectory(testNegSamplesDir, negativeTestImages, validExtensions);

        // find the number of images in positive test set
        
        // detect signs in positive images using a for loop

        // output the prediction of positive test set
            // True positive = total predicted positive/total test pos images

        // detect signs in negative images using a for loop

        // output prediction
            // True negative = total predicted negative/total test neg images



        

