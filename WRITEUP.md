# Project Write-Up

## Explaining Custom Layers

The process behind converting custom layers,there are a few differences depending on the original model framework.
- In both **TensorFlow** and **Caffe** :
  - First,register the custom layers as extensions to the Model Optimizer.
  - For **Caffe** :
  	  - Second, register the layers as Custom, then use **Caffe** to calculate the output shape of the layer.
	  - You’ll need **Caffe** on your system to do this option.

  - For **TensorFlow** :
	  - Second,replace the unsupported subgraph with a different subgraph.
	  - The final **TensorFlow** option is to actually offload the computation of the subgraph back to **TensorFlow** during inference.
		
Some of the potential reasons for handling custom layers are :
- If any Layer which is not present in list of **Supported Layer** of any particular model Framework,The **Model Optimizer** 
  automatically classifies it as **Custom Layer**.
- **Custom Layers** are mainly used for handling UnSupported Layers.

## Comparing Model Performance

- My method(s) to compare models before and after conversion to Intermediate Representations
are :

  - The difference between model accuracy pre- and post-conversion was :
    - The model accuracy before convertion will be slightly higher than the model which is converted into Intermediate Representation. 
  - The size of the model pre- and post-conversion was :
    - The size of the model after converting into Intermediate Representation will be smaller than the model before convertion.
	- When a model is converted into IR It'll convert into .xml file which contains model architecture and .bin file which contains weights and baises.
  - The inference time of the model pre- and post-conversion was :
	- The inference time of the model after converting to IR will be shorter than the model before converting.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are :
- Calculate your store's conversion ratio.
  - Retail conversion measures the proportion of tourists to a retail outlet who make a sale.If 400 people visit your store in a very day, but only 75 buy something, the conversion rate is 18.75 percent.
  - To measure retail conversion, you would like to live numbers of tourists and understand the thanks to interpret the information.
- Measuring your footfall patterns.
  - Counting the quantity of individuals who enter your store tells you the way many buying opportunities you've got had in-store for any particular day.
  - By dividing total transactional sales by footfall data for that very same period, you'll be able to determine what number shoppers were persuaded to place their hand in their pocket and make a purchase: i.e. your conversion rate.
  -  By measuring customer footfall and establishing your conversion rate, you'll be able to specialise in key areas to extend store performance.
- People counting apps forms the premise for a variety of high-tech solutions like :
  - Including retail analytics, queue management and security systems
## Assess Effects on End User Needs

- Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows :
  - We need a model which has been trained and tested on a dataset which gives a good accuracy before deploying the model at edge.
  - Lighting is a important factor, where a model cannot detect human in darkness.providing enough lighting will enable the model to detect precisely. 
  - The focal length of a camera has to be good enough to capture the people,If the camera's focal length is low, the model performs very poorly and may give incorrect detections.


## Pre-converted IR OpenVINO Model Used

- Model : [person-detection-retail-0013]
  - To navigate to the directory containing the Model Downloader:
    - ` cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader `
  - Downloading person-detection-retail-0013
    - ` sudo ./downloader.py --name person-detection-retail-0013 --precisions FP16 -o /home/workspace `
  - Verifying Download:
	- Get back to ` /home/workspace` by entering ` cd ..` command. And then enter ` cd intel ` and there you'll find **person-detection-retail-0013** file which contains both .xml and .bin files.
  - This model is sufficient for the app beacause it datects the people in the frame and counts them correctly and gives us the exact average time duration of the person in the frame along with the stats.
  
		
## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [ssd_inception_v2_coco_2018_01_28]
  - ssd_inception_v2_coco_2018_01_28 is downloade from this [link](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments :
    - ` tar -xvf ssd_inception_v2_coco_2018_01_28.tar.gz `
	- `cd ssd_inception_v2_coco_2018_01_28`
	- `python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`
  - The model was insufficient for the app because,The model is producing incorrect detection of people by doubling/tripling the count for a single person who's present in the frame. 
  - I tried to improve the model for the app by :
    - tuning the confidence threshold with diffenent values like `0.6,0.625,0.7,0.85,0.9`
	- By incrementing the frame start_frame_number with `3,4,5,6`
  
- Model 2: [ssd_mobilenet_v2_coco_2018_03_29]
  - ssd_mobilenet_v2_coco_2018_03_29 is downloade from this [link](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments :
    - `tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz`
	- `cd ssd_mobilenet_v2_coco_2018_03_29`
	- `python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`
 - The model was insufficient for the app because,The model is producing incorrect detection of people by doubling/tripling the count for a single person who's present in the frame. 
  - I tried to improve the model for the app by :
    - tuning the confidence threshold with diffenent values like `0.6,0.625,0.7,0.85,0.9`
	- By incrementing the frame start_frame_number with `3,4,5,6`

- Model 3: [ssdlite_mobilenet_v2_coco_2018_05_09]
  - ssdlite_mobilenet_v2_coco_2018_05_09 is downloade from this [link](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments...
  - `tar -xvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz`
	- `cd ssdlite_mobilenet_v2_coco_2018_05_09`
	- `python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`
  - The model was insufficient for the app because,The model is producing incorrect detection of people by doubling/tripling the count for a single person who's present in the frame. 
  - I tried to improve the model for the app by :
    - tuning the confidence threshold with diffenent values like `0.6,0.625,0.7,0.85,0.9`
	- By incrementing the frame start_frame_number with `3,4,5,6`