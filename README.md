# Third-Eye
Suspicious activities are of a problem when it comes to the potential risk it brings to humans. With the increase in criminal activities in urban and suburban areas, it is necessary to detect them to be able to minimize such events. Early days surveillance was done manually by humans and were a tiring task as suspicious activities were uncommon compared to the usual activities. With the arrival of intelligent surveillance systems, various approaches were introduced in surveillance.</br>
Thus,the basic idea is using the camera to detect suspicious activity inside a bounsing box.

# Teammates
Pranav Chopdekar</br>
Priyanshu Agarkar</br>
Komal Chitnis</br>
Sahil Gujar</br>

# Link to download frozen_inference_graph.pb
https://github.com/methylDragon/opencv-python-reference/tree/master/Resources/Models/mask-rcnn-coco

# Working 
In our project, we are using the normal everyday usage camera and positioning it in a way such that it acts like a static  surveillance camera. This camera would be used to display the real time footage of the area. This real time footage would be used by using the OpenCV module in python. The restricted area would thus defined by the user using the select ROI( Region of interest) method. This allows the user the choose any area in the real-time view to be defined as restricted.
Ones defined, the area would be highlighted from the rest of the view thus making it look like a bounding box, Hence, if any entity enters this bounding box, it would trigger a alarm, alerting the authorities that a suspicious activity took place. 


