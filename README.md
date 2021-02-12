# Detection and recognition of graphic elements in a window

### Objectives achieved in the project

- detection of buttons 
- detection of text fields 
- detection of radial buttons 
- detection of check 
- boxes and its status (checked, unchecked) 
- representation of results in a json format 
- tool for viewing results

![Pipeline](https://github.com/PetricaP/ProiectPIMPY/blob/master/Documentation/ss/pipeline.png)


### GUI
Python's tkinter library was used to create the viewing tool.

![GUI](https://github.com/PetricaP/ProiectPIMPY/blob/master/Documentation/ss/gui.png)


## Implementation of graphic element detection functions

#### Steps for detecting rectangles
1) Finding contours using the function in opencv findContours with the approximation parameter CHAIN_APPROX_SIMPLE to identify only significant points
2) Approximating contours with a polygon using the approxPolyDP function and filtering all polygons that do not have 4 sides
3) Filtering polygons that are not parallelograms (program verification that opposite sides are parallel)
4) Filtering parallelograms that are not rectangles (checking by program that the difference of the coordinates associated with the opposite sides is less than 0.1 * the length of the corresponding side)
5) Approximating the polygon with a rectangle using boundingRect
6) Elimination of rectangles with an area smaller than a specified area
7) Elimination of redundant values

![Rectangle](https://github.com/PetricaP/ProiectPIMPY/blob/master/Documentation/ss/rectangle.png)


#### Steps for detection of text-containing regions (EAST neural network)
1) resize the image so that the length and width are a multiple of 32 (EAST detector
accept only such images)
2) image preprocessing for its classification in the neural network using the function
blobFromImage (decreasing the average of the neural network (123.68, 116.78, 103.94) and scaling
image)
3) transmitting the obtained layers to the network and obtaining the regions with text and some probabilities
presence of the text in those regions
4) decoding predictions using the minimum probability of 10%
5) applying the non_max_supression function (imutils.object_detection) on bounding boxes
to remove their overlap
6) rescaling to the original dimensions

![EAST](https://github.com/PetricaP/ProiectPIMPY/blob/master/Documentation/ss/text.png)


#### Button Detection
The results obtained from the previous functions (detection of rectangles in contours and 
detection of text) are used to detect the buttons. 

Button detection steps:
1) Removal of outer rectangles (if a rectangle D1 contains another rectangle D2 inside, then D1 is considered an outer rectangle and is removed from the set of rectangles)
2) Elimination of rectangles for which the length is less than the height (buttons represent horizontal rectangles, ie w> h)
3) Impose that the ratio between (the intersecting rectangle between the contour rectangle and the text rectangle) and (the minimum between the contour rectangle and the text rectangle) be greater than 0.8. This condition excludes rectangles that do not intersect or intersect
partially intersects.
4) Imposing the condition that the absolute difference between the heights of the 2 rectangles be less than half of their maximum height (condition that forces the observance of the text rectangle to be inside the contour rectangle)

![Button](https://github.com/PetricaP/ProiectPIMPY/blob/master/Documentation/ss/button.png)


#### Checkbox detection
The results obtained from the previous functions (detection of contour rectangles and 
text detection) are used to detect checkboxes. 

![Checkbox](https://github.com/PetricaP/ProiectPIMPY/blob/master/Documentation/ss/checkbox_1.png)


Checkbox detection steps:
1) Remove contour rectangles for which the absolute difference between the length and width of the rectangle is greater than 20 pixels (this condition forces the rectangle to be approximately one square)
2) Removal of contour rectangles for which the area of the rectangle > 200 (pixels^2) (by this condition we require that the rectangle be relatively small in size)
3) Determining the text associated with the check-box by imposing the conditions:
   * the ordinate of the text rectangle should be at most 10 pixels different from the ordinate of the checkbox (code optimization condition)
   * imposing the condition that the ordinate of the text rectangle be contained in the circle centered in the center of the check-box with radius
4 * checkbox.width
4) Determining the state of the checkbox (checked, unchecked) by applying the OTSU binary method and verifying that the black pixels represent more than 20% of the checkbox area (works for darker intensity of check mark against the background)

![Checkbox](https://github.com/PetricaP/ProiectPIMPY/blob/master/Documentation/ss/checkbox_2.png)


#### Radialbox detection
The results obtained from the previous functions (detection of contour rectangles and text detection) are used to detect radialboxes. 
Radialbox detection steps:

![Radialbox](https://github.com/PetricaP/ProiectPIMPY/blob/master/Documentation/ss/radial_1.png)


1) Detecting circles using the HoughCircles OpenCV method (first the original image is transformed into a Gray image, then it is blurred with the middle filter to remove noise).
2) Impose the condition that the ordinate of the center of the circle be greater than the ordinate of the text rectangle (circle.center.y> rect.y)
3) The point of the text rectangle should be inside the circle determined by the central radial button and the radius equal to 3 * the radius of the radial button
4) Determining the state of the check-box (checked, unchecked) by applying the OTSU binary method and verifying that the black pixels represent more than 25% of the
checkbox area (works for darker tick intensity than background)

![Radialbox](https://github.com/PetricaP/ProiectPIMPY/blob/master/Documentation/ss/radial_2.png)


## How to run the project

1. `pip3 install -r requirements.txt`
2. `python3 gui_analyzer.py --gui`





