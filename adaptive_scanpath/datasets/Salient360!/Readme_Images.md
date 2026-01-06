**Description of the images dataset**

The dataset is structured as follows:

* Stimuli: 85 omnidirectional images in equi-rectangular format. 
* H: Folder containing the saliency maps and scanpaths from head-only movements. 
* HE: Folder containing the saliency maps and scanpaths (for each eye) from head and eye movements. 
* Tools: Python scripts to parse the saliency map binary files, and to compute saliency and scapanth measures. 

The details about the saliency map files and the scanpath files are:

* Saliency maps from head-only movements: Binary files with a resolution of 2048x1024, containing the saliency values (float32) of the 360-image in equi-rectangular format, organised row-wise. The saliency maps are normalized, so the sum of the saliency values must be equal to 1. For each sampled head position, the center of the viewport is considered. Then, an isotropic 3.34-degree Gaussian foveation filter centered in the view-port is applied.

* Scanpaths from head-only movements: Text files are provided with the scanpaths from head movement with 100 samples per observer. Each line contains a quadruple vector that indicates the fixation number, longitude, latitude and fixation timestamp, respectively. The fixation number increments serially for a particular observer and resets to 0  when we reach the next observer, after all of the fixations of the given observer are reported. The fixation starting time is indicated in seconds, and latitude and longitude positions are normalized between 0 and 1 (so they should be multiplied  according to the resolution of the desired  equi-rectangular image output dimension).

* Saliency maps from head and eye movements: Binary files with a resolution of 2048x1024, containing the saliency values (float32) of the 360-image in equi-rectangular format, organised row-wise. The saliency maps are normalized, so the sum of the saliency values must be equal to 1. For each eye fixation, an isotropic 2-degree gaussian foveation filter centered at the fixation position is applied. This process is applied to fixations from both left and right eyes, and then combined in the final saliency map.

* Scanpaths from head and eye movements: Text files are provided with the scanpaths from both left and right eyes. Each line contains a quadruple vector that indicates the fixation number, longitude, latitude and fixation timestamp, respectively. The fixation number increments serially for a particular observer and resets to 0  when we reach the next observer, after all of the fixations of the given observer are reported. The fixation starting time is indicated in seconds, and latitude and longitude positions are normalized between 0 and 1 (so they should be multiplied  according to the resolution of the desired  equi-rectangular image output dimension).

**Experiment**

The head mounted display (HMD) Oculus-DK2 was used for this test. It has a frame refresh rate of 75Hz, resolution of 960x1080 per eye and a total viewing angle of 100x100 degrees. The gyroscopic sensors within the device are able to transmit the orientation data at a rate equal to the device frame refresh rate. A small eye-tracking camera from Sensomotoric Instruments (SMI) was integrated into the device and was able to transmit eye-tracking data binocularly at 60Hz.

The software setup included a custom build unity software along with the Oculus-DK2 driver version 2.0. The software had a feature to check for calibration accuracy every two minutes and re-calibrated each time if necessary.

A total of 63 observers in the age group of 19-52 participated in the test. Observers were tested for visual acuity using the Snellen Test and their dominant eye was also determined using the cardboard technique. 

To maintain a natural (free-viewing like) gaze pattern, subjects were made to view the scene normally without the need to provide explicit quantitative measurements. They were instructed to watch the scene as normally as possible with a combination of head and eye-movement. Observers were also free to stop the test anytime in case they felt fatigued or had a sensation of vertigo. There were five images used as a training for the observers before starting the actual test.

A total of 60 stimuli were shown to the observers in a sequence. Each stimuli lasted for 25 seconds and there was a 5 second gray screen between two stimuli. Every two minutes there was a calibration performed to check the accuracy of the eye-tracker. The test itself lasted for about 35 minutes and the observers had a pause of 5 minutes at the half point of the experiment. The observers were themselves seated comfortably in a turn-chair and were free to rotate the full 360 degrees and also move the chair within the room if necessary. The position of each 360 image was reset to the equirectangular image center at the start of each viewing (irrespective of their position). This was done to ensure that all observers start at the same starting position in the panorama.

**Citing the Database**

Please cite the following paper in your publications making use of the Salient360! database:

* Yashas Rai, Patrick Le Callet and Philippe Guillotel. “Which saliency weighting for omni directional image quality assessment?”. In Proceedings of the IEEE Ninth International Conference on Quality of Multimedia Experience (QoMEX’17). Erfurt, Germany, pp. 1-6, June 2017.

* Yashas Rai, Jesús Gutiérrez, and Patrick Le Callet. 2017. “A Dataset of Head and Eye Movements for 360 Degree Images“. In Proceedings of the 8th ACM on Multimedia Systems Conference (MMSys'17). ACM, New York, NY, USA, pp. 205-210, June 2017.