# Log file for meeting with Prof.
## 20190128 11:00AM WEB224H
1. Image Compression
	Get familiar with MPEG-2/H.263/H.264/H.265
1. iPhone portrait mode project
  1. Using OpenCV to identify square box of people or face
  1. Segment the object and the background
  2. Face recognition (we don't need recognize different people)
  1. Blur the background
  2. compare it with the iPhone portrait output
  3. I can use it as the DIP class project
1. Some feature of iPhone portrait mode
	1. Face detaction
	2. Facial landmarking
	3. Facial light effect adjusting
	4. Object Segmentaion
	5. Blur the Background

	
## 20190206 11:00AM WEB224H
1. If face detaction fails, do a body detaction
2. If the difference between object is large enough, omit the smaller one.
3. Find a upper body harr feature file to do the upperbody detection

## 20190211 11:00AM WEB224H
1. Do a background filter then save as jpeg
2. Compare the file size of different image based on the level of blur
3. Blur should be undetectable and object of interest should be in good shape


# My research
#### Face Detaction
Encode a picture using the HOG algorithm to create a simplified version of the image. Using this simplified image, find the part of the image that most looks like a generic HOG encoding of a face.

HOG, Histogram of Oriented Gradients

Deep Learning based face detaction, shipped with OpenCV
Single Shot Detector (SSD) framework with ResNet as the base network

**(done)** Haar-like feature is what is project focused

#### Face recognition (didn't achieve)
Figure out the pose of the face by finding the main landmarks in the face. Once we find those landmarks, use them to warp the image so that the eyes and mouth are centered.

Deep learning to extract feature

Pass the centered face image through a neural network that knows how to measure features of the face. Save those 128 measurements.

Looking at all the faces we’ve measured in the past, see which person has the closest measurements to our face’s measurements. That’s our match!

#### Image Matting
**(done)** A Closed Form Solution to Natural Image Matting, but it's off the table beacuse its half-automatic

