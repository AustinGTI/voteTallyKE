# VoteTallyKE

This repo provides a complete system that is capable of extracting handwritten vote tallies from images of the polling station electoral forms with up to 95% accuracy

A custom neural network based on the YOLOv5 object detection architecture is used to read the handwritten digits and a separate neural network also based on YOLOv5 is used to locate the position of the QR code from which the location of the tallies can be calculated. As a backup, the paddleOCR character recognition engine is used to locate keywords on all forms that can be used to calculate the location of the tallies (this is only a backup as it is slower though more accurate)

---



## Training the neural networks

### Handwritten digits

The first neural network was trained to detect handwritten digits accurately using custom training data that is generated using the popular MNIST digit dataset. 
The digits are placed in a 1000 by 1000 px image that replicates the appearance of that area of the electoral form as closely as possible. 
This is done by :

* replicating the  background pattern of the form
* drawing horizontal gridlines around the digits
* placing the digits in 4 lines of 3 digits each as is consistent with the 4 electoral candidates
* Adding image artifacts common in the electoral form eg. stamp marks,
* Adding random perlin noise and blur to the image 
  
  

1000 such images are generated with the bounding locations of each digit stored in corresponding text files. The images are split into training, testing and validation datasets and a model is trained on them using the YOLOv5 nano architecture to recognize handwritten digits as they would appear in the forms.



![img](https://github.com/AustinGTI/voteTallyKE/blob/3848dbc6e762b2adcfc3a620debfef0c78476211/metaAssets/dgtTrain/dgtTrain_v2.png?raw=true)

## QR Code

The entire form is too large and unpredictable to reliably read the correct digits on a regular basis. For this reason, the rotation,scale and position of the form in the image need to be calculated and accounted for in order to crop out the required area. This is done by training a neural network to recognize the QR code and logo that are present in every form.

A form template is loaded the populated with a randomly generated QR code and the logo and slightly varying locations in the general area where they would expect to be found. The resultant image is then randomly offset and slightly rotated to simulate the real form and some background noise and blur is added to simulate different cameras and lighting conditions. The positions of the QR code and logo are saved and stored along with 1000 such images.

A neural network based on the YOLOv5 architecture is trained on these images and positions to recognize the QR code and logo, the squares on 3 corners of the QR code are used to calculate rotation

![process of generating QR code and logo training data](https://github.com/AustinGTI/voteTallyKE/blob/master/metaAssets/qrTrain/qrTrain_v1.png?raw=true)

## Character recognition

As a backup, the [paddleOCR · PyPI](https://pypi.org/project/paddleocr/) library is used to read any characters on the form and calculate their scale, position and orientation. This data can be used similarly to the QR code and logo to locate the position of the vote tally numbers. PaddleOCR is not reliable at reading handwritten digits and is rather slow at reading characters (at least 15 seconds) which is why the first 2 custom neural networks are trained and used as the primary engines of the system.

---

# The System

The input to the system is a directory containing sub directories of every county in the country which subsequently contain photo scanned pdf files of every electoral form. This is the format in which the forms are available of the public IEBC portal.

Each of these pdf files are iterated through from county to county.

The procedure carried out on each file is as follows:

### 1. pdf to np array

A pdf file is converted to a png image using the [pdf2image · PyPI](https://pypi.org/project/pdf2image/) library which requires the pdf rendering software [Poppler](https://poppler.freedesktop.org/) as a dependency. The png image is converted into an array with [numpy · PyPI](https://pypi.org/project/numpy/) that can be processed using the [opencv-python · PyPI](https://pypi.org/project/opencv-python/) computer vision library. These 2 libraries openCV and numpy will be the main libraries used for image manipulation and processing.

### 2. locate QR and logo

The previously trained YOLOv5 model is used to locate the positions of the QR code and logo on the form. This is successful at least 90% of the time. This number could be improved with a more robust model. The y positions of the logo and QR code relative to each other are used to estimate the rotation of the form. In a perfectly oriented form, the center of the logo and that of the QR code are on the same level horizontally.

Any rotation higher than 5&deg; in either direction is corrected by rotating the form by the same amount in the opposite direction. Any rotation lower than 5&deg; is ignored for 2 reasons.

* The rotation estimation is not very accurate, and those less than 5 are likely negligible noise

* Rotating the image is computationally expensive when 46,000 forms need to be processed so it is best to do it as little as possible

After the rotation. The scale and position of the QR code and logo relative to that of the vote tally area are used to estimate the location of the vote tallies. The location is averaged out based on the confidence of each prediction and cropped out

### 2.5. character recognition

In the 10 - 15% of situations where the logo and QR code cannot be located with more than 90% confidence for each, the slower character recognition method is used instead.

There are certain key sentences that are present in all forms. For the highest level of accuracy, sentences that are long yet also printed in large bold fonts are preferred. In my system, 3 specific sentences are chosen.

* **PRESIDENTIAL ELECTION RESULTS AT THE POLLING STATION**

* **Number of votes cast in favour of each candidate**

* **No of valid votes obtained**

If at least one of these sentences is among those detected by PaddleOCR and the accuracy is at least 90%, the same procedure is carried out as with the QR code and logo method.

PaddleOCR returns bounding boxes as well that make it possible to calculate the angle of rotation with trigonometry and rotate the image back to an upright position. The size and scale of the sentences is used to estimate the location of the tallies and that general area is cropped out.

This is successful more than 90% of the time when the QR code and logo cannot be found which adds up to a success rate of up to 99% at finding and cropping out the estimated area where the tallies are located with a reasonable degree of certainty.

### 3. digit detection and recognition

Given the cropped out general area where the handwritten vote tallies lie, some image processing is performed to increase the contrast and hence visibility of the handwritten digits as well as reduce background noise.

Subsequently, the handwritten digit recognition model can locate and recognize digits reliably though there are occassionally false positives on account of the random noise and artifacts that may be on the image.

In order to parse out the correct digits from false positives, digits clustered along the same general y position are grouped together.

From these clusters, several possibilities may arise:

* if there are exactly 4 and none of them has more than 3 digits, they are read as the tallies of the 4 candidates and stored (this is the ideal situation)

* If there are exactly 4 clusters and some of them have more than 3 digits, the digits with the highest confidence that do not overlap are chosen in each cluster and considered to be the tallies of the 4 candidates

* If there are less than 4 clusters, it is considered a bad form and the system moves to the next one, There would be no point in recording tallies for some candidates and not others (this is quite rare)

* If there are more than 4 clusters, all combinations of 4 clusters from the total are extracted and weighed based on location, width and number of digits. The ideal set of 4 will each have the same number of digits along roughly the same x position and of the same width. The set of 4 clusters that fit this criteria the closest are chosen and read as the tallies for each of the 4 candidates.
  
  

### 4. Saving the data

Each file name conveniently has a polling station code that can be used to determine the county, constituency , ward and polling center where the form came from. All of these details are saved for each form along with the tallies in a csv file if the previous operations were all successful. An image is generated displaying the extracted tallies next to the handwritten digits for easy confirmation and the system moves to the next form.

---

This process takes about 0.7 seconds per form at best and up to 16 seconds at worst when character recognition is used. For the roughly 46,000 polling stations this adds up to 10 - 24 hours for all the forms. If a GPU is used, the character recognition can be alot faster

![system process diagram](https://github.com/AustinGTI/voteTallyKE/blob/master/metaAssets/readForm/readForm_v1.png?raw=true)

# The Code


