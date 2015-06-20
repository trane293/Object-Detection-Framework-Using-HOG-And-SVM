# Object Detection Framework Using HOG And SVM
##AUTHOR - Anmol Sharma
--------------------------------------------------------------------------------------<br />
			#Description of Files and Folders<br />
--------------------------------------------------------------------------------------<br />

genfiles - FOLDER - Will contain SVM models that are trained on different training sets<br />
		    specified by the naming scheme of different folders in which they <br />
		    will be stored. For example - <br />
		
		    svmlight_24   -  contains SVM model trained on 24x24 positive and<br />
		                     negative images. This model is not trained using <br />
		                     hard negatives.<br /><br />

        svmlight_64   -  contains SVM model trained on 64x64 positive and <br />
            			       negative images. This model is not trained using <br />
            			       hard negatives.<br /><br />

		    svmlight_24hn  - contained SVM model trained on 24x24 positive and<br />
		    				         negative images. This model is also trained on a <br />
		    				         number of hard negatives obtained from a road video<br />
		    				        and few from indoor scenes.<br /><br />

		svmlight_24hn_test - this folder will be used for storing test SVM models<br />
							           which is not finalized and for prototyping purposes. <br /><br />

       libsvm - FOLDER - should contain the libsvm.h header file. <br /><br />

    nbproject - FOLDER - contains some important files that are used to configure the <br />
					 HOG training process. <br /><br />

          pos - FOLDER - should contain positive images for training. <br /><br />

          neg - FOLDER - should contain negative images for training. <br /><br />

    sidefiles - FOLDER - contains some screenshots on how to set up this code. Useless.<br /><br />

     svmlight - FOLDER - contains important svmlight header files. These files were downloaded<br />
					               from official website and then make build here. <br /><br />

         test - FOLDER - should contain test data if required. <br /><br />

     buildcode_main.sh - SHELL SCRIPT - Used to build the project using main.cpp as main file.<br /><br />

    buildcode_train.sh - SHELL SCRIPT - Used to build the project using main_train.cpp as main file.<br /><br />

  buildcode_prototype - SHELL SCRIPT - Used to build the project using main_prototype as main file.<br /><br />

       main_train.cpp - CPP FILE - Should contain the main training code to get training files, train an <br />
							                     SVM classifier and save the model to genfiles. <br /><br />

            main.cpp - CPP FILE - Should contain the main code to test a pre-trained classifier ideally trained <br />
					                        using main_train.cpp<br /><br />

  main_prototype.cpp - CPP FILE - Should contain prototype code in case user doesn't want to mess up <br />
								                  with main.cpp branch. <br /><br />

--------------------------------------------------------------------------------------<br />
--------------------------------------------------------------------------------------<br />
            		       #HOW TO BUILD CODE <br />
--------------------------------------------------------------------------------------<br />
--------------------------------------------------------------------------------------<br />

Open terminal. <br />
Write:<br />
           --> chmod +x <name_of_build_file><br />
			example: chmod +x buildcode_main.sh<br />
           --> ./buildcode_main.sh<br />

        
