# Object Detection Framework Using HOG And SVM
##AUTHOR - Anmol Sharma

-------------------------------------------------------------------------------------
#Description of Files and Folders


        genfiles - FOLDER - Will contain SVM models that are trained on different training sets
		    specified by the naming scheme of different folders in which they 
		    will be stored. For example - 
		
		    svmlight_24   -  contains SVM model trained on 24x24 positive and
		                     negative images. This model is not trained using 
		                     hard negatives.

        svmlight_64   -  contains SVM model trained on 64x64 positive and 
            			       negative images. This model is not trained using 
            			       hard negatives.
        
        svmlight_24hn  - contained SVM model trained on 24x24 positive and
		         negative images. This model is also trained on a 
		         number of hard negatives obtained from a road video
		         and few from indoor scenes.

        svmlight_24hn_test - this folder will be used for storing test SVM models
			     which is not finalized and for prototyping purposes. 

        libsvm - FOLDER - should contain the libsvm.h header file. 

        nbproject - FOLDER - contains some important files that are used to configure the 
                            HOG training process. 

        pos - FOLDER - should contain positive images for training. 

        neg - FOLDER - should contain negative images for training. 

        sidefiles - FOLDER - contains some screenshots on how to set up this code. Useless.

        svmlight - FOLDER - contains important svmlight header files. These files were downloaded
					               from official website and then make build here. 

        test - FOLDER - should contain test data if required. 

        buildcode_main.sh - SHELL SCRIPT - Used to build the project using main.cpp as main file.

        buildcode_train.sh - SHELL SCRIPT - Used to build the project using main_train.cpp as main file.

        buildcode_prototype - SHELL SCRIPT - Used to build the project using main_prototype as main file.

        main_train.cpp - CPP FILE - Should contain the main training code to get training files, train an 
		                     SVM classifier and save the model to genfiles. 

        main.cpp - CPP FILE - Should contain the main code to test a pre-trained classifier ideally trained 
	                        using main_train.cpp

        main_prototype.cpp - CPP FILE - Should contain prototype code in case user doesn't want to mess up 
			                  with main.cpp branch. 

--------------------------------------------------------------------------------------
###HOW TO BUILD CODE 

Open terminal. 
Write:
        chmod +x <name_of_build_file>
		example: chmod +x buildcode_main.sh
        ./buildcode_main.sh
------------------------
        
