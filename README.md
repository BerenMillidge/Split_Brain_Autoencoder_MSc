# Split_Brain_Autoencoder_MSc

Here is a cleaned up and simply working version of the split brain autoencoder model. By following the steps below you should be able to get the model running on the MIT saliency dataset with the colour-channel model where each autoencoder model gets either the red, green, or blue channel of the image.

Steps to get it working:

0.) Clone/download this git repository onto your computer.

1.) Go to the MIT saliency benchmark website (http://saliency.mit.edu/datasets.html) and download the data. The CAT2000 is the largest but you can also experiment with the others if you want. It's probably best to save the data into a subdirectory of wherever you saved the code.

2.) Run the file_reader.py to process the data to be fed into the autoencoder model. When running, you need to specify the directory you have saved the data into, as well as what you want the processed data file to be called. For instance, if you have saved the MIT images into a folder called 'images' in the same directory as the code, and want to call the results - 'data' then type: 'python file_reader.py images/ data' into the terminal and run it.

3.) Once this has run there should be some data files in the same folder as the code. If you downloaded the CAT2000 images, there are multiple categories so the file_reader saves each category separately as well as the big combined file of all of them. The combined file should be called something like *your_name*_images_combined. There should also be a *your_name*_outputs_combined - this is the gold-standard human salience maps for comparison.

4.) Run the model on the data. The code for running the model is in the file 'experiments.py'. It requires that you give it on the command line the name of the data file and the name of the file you want it to save the results to. So, if you called the data 'data', and want to call the output results 'results', then run: 'python experiment.py data_images_combined results' on the commandline and the model should run. You can also specify the number of epochs you want the model to run for by typing it as the last commandline argument, or just changing it in the code.

After you've got the MIT images combined into one data file you shouldn't need anything from the file_reader code. The model itself is contained in autoencoder.py. The routines for splitting the images into channels and running the model are in experiments.py.

Hope this helps.

If you run into problems getting it running or have any other questions, feel free to email me!
