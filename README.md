# 🧠 Deep Learning Fundus Image Analysis for Early Detection of Diabetic Retinopathy

## 📖 Overview
Diabetic Retinopathy (DR) is a serious eye disease caused by prolonged diabetes that can lead to vision loss or blindness. Early detection is essential, but manual screening is time-consuming and requires expert ophthalmologists.

This project presents a **Deep Learning-based system** that analyzes retinal (fundus) images using a **Convolutional Neural Network (CNN)** with the **Xception model**. The system is integrated into a **Flask web application** to provide an easy-to-use interface for prediction.

---

## ⚠️ Important Note
Some files (such as trained model files or datasets) are not included in this repository due to GitHub size limitations.

👉 Full project files are available here:  
**https://drive.google.com/drive/folders/1c_TF6V9flufmMloMGYHUI01u34CiWiJy?usp=drive_link**

---

## 🎯 Objectives
- Develop a deep learning model for DR detection  
- Use transfer learning with Xception  
- Perform image preprocessing and augmentation  
- Build a Flask-based web application  
- Store user data and results using IBM Cloudant DB  

---

## 🏗️ Project Structure

├───mla_env
│   ├───Include
│   ├───Lib
│   │   └───site-packages
│   │       ├───blinker
│   │       │   └───__pycache__
│   │       ├───blinker-1.9.0.dist-info
│   │       ├───click
│   │       │   └───__pycache__
│   │       ├───click-8.3.1.dist-info
│   │       │   └───licenses
│   │       ├───colorama
│   │       │   ├───tests
│   │       │   │   └───__pycache__
│   │       │   └───__pycache__
│   │       ├───colorama-0.4.6.dist-info
│   │       │   └───licenses
│   │       ├───flask
│   │       │   ├───json
│   │       │   │   └───__pycache__
│   │       │   ├───sansio
│   │       │   │   └───__pycache__
│   │       │   └───__pycache__
│   │       ├───flask-3.1.2.dist-info
│   │       │   └───licenses
│   │       ├───itsdangerous
│   │       │   └───__pycache__
│   │       ├───itsdangerous-2.2.0.dist-info
│   │       ├───jinja2
│   │       │   └───__pycache__
│   │       ├───jinja2-3.1.6.dist-info
│   │       │   └───licenses
│   │       ├───markupsafe
│   │       │   └───__pycache__
│   │       ├───markupsafe-3.0.3.dist-info
│   │       │   └───licenses
│   │       ├───pip
│   │       │   ├───_internal
│   │       │   │   ├───cli
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───commands
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───distributions
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───index
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───locations
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───metadata
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───models
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───network
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───operations
│   │       │   │   │   ├───build
│   │       │   │   │   │   └───__pycache__
│   │       │   │   │   ├───install
│   │       │   │   │   │   └───__pycache__
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───req
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───resolution
│   │       │   │   │   ├───legacy
│   │       │   │   │   │   └───__pycache__
│   │       │   │   │   ├───resolvelib
│   │       │   │   │   │   └───__pycache__
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───utils
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───vcs
│   │       │   │   │   └───__pycache__
│   │       │   │   └───__pycache__
│   │       │   ├───_vendor
│   │       │   │   ├───cachecontrol
│   │       │   │   │   ├───caches
│   │       │   │   │   │   └───__pycache__
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───certifi
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───chardet
│   │       │   │   │   ├───cli
│   │       │   │   │   │   └───__pycache__
│   │       │   │   │   ├───metadata
│   │       │   │   │   │   └───__pycache__
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───colorama
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───distlib
│   │       │   │   │   ├───_backport
│   │       │   │   │   │   └───__pycache__
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───html5lib
│   │       │   │   │   ├───filters
│   │       │   │   │   │   └───__pycache__
│   │       │   │   │   ├───treeadapters
│   │       │   │   │   │   └───__pycache__
│   │       │   │   │   ├───treebuilders
│   │       │   │   │   │   └───__pycache__
│   │       │   │   │   ├───treewalkers
│   │       │   │   │   │   └───__pycache__
│   │       │   │   │   ├───_trie
│   │       │   │   │   │   └───__pycache__
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───idna
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───msgpack
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───packaging
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───pep517
│   │       │   │   │   ├───in_process
│   │       │   │   │   │   └───__pycache__
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───pkg_resources
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───progress
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───requests
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───resolvelib
│   │       │   │   │   ├───compat
│   │       │   │   │   │   └───__pycache__
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───tenacity
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───tomli
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───urllib3
│   │       │   │   │   ├───contrib
│   │       │   │   │   │   ├───_securetransport
│   │       │   │   │   │   │   └───__pycache__
│   │       │   │   │   │   └───__pycache__
│   │       │   │   │   ├───packages
│   │       │   │   │   │   ├───backports
│   │       │   │   │   │   │   └───__pycache__
│   │       │   │   │   │   ├───ssl_match_hostname
│   │       │   │   │   │   │   └───__pycache__
│   │       │   │   │   │   └───__pycache__
│   │       │   │   │   ├───util
│   │       │   │   │   │   └───__pycache__
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───webencodings
│   │       │   │   │   └───__pycache__
│   │       │   │   └───__pycache__
│   │       │   └───__pycache__
│   │       ├───pip-21.2.3.dist-info
│   │       ├───pkg_resources
│   │       │   ├───extern
│   │       │   │   └───__pycache__
│   │       │   ├───tests
│   │       │   │   └───data
│   │       │   │       └───my-test-package-source
│   │       │   │           └───__pycache__
│   │       │   ├───_vendor
│   │       │   │   ├───packaging
│   │       │   │   │   └───__pycache__
│   │       │   │   └───__pycache__
│   │       │   └───__pycache__
│   │       ├───setuptools
│   │       │   ├───command
│   │       │   │   └───__pycache__
│   │       │   ├───extern
│   │       │   │   └───__pycache__
│   │       │   ├───_distutils
│   │       │   │   ├───command
│   │       │   │   │   └───__pycache__
│   │       │   │   └───__pycache__
│   │       │   ├───_vendor
│   │       │   │   ├───more_itertools
│   │       │   │   │   └───__pycache__
│   │       │   │   ├───packaging
│   │       │   │   │   └───__pycache__
│   │       │   │   └───__pycache__
│   │       │   └───__pycache__
│   │       ├───setuptools-57.4.0.dist-info
│   │       ├───werkzeug
│   │       │   ├───datastructures
│   │       │   │   └───__pycache__
│   │       │   ├───debug
│   │       │   │   ├───shared
│   │       │   │   └───__pycache__
│   │       │   ├───middleware
│   │       │   │   └───__pycache__
│   │       │   ├───routing
│   │       │   │   └───__pycache__
│   │       │   ├───sansio
│   │       │   │   └───__pycache__
│   │       │   ├───wrappers
│   │       │   │   └───__pycache__
│   │       │   └───__pycache__
│   │       ├───werkzeug-3.1.5.dist-info
│   │       │   └───licenses
│   │       └───_distutils_hack
│   │           └───__pycache__
│   └───Scripts
├───model
│   └───__pycache__
├───static
│   ├───css
│   ├───images
│   └───uploads
├───templates
└───uploads


---

## ⚙️ Technologies Used
- **Programming Language:** Python  
- **Frameworks:** TensorFlow, Keras, Flask  
- **Model:** Xception (Transfer Learning)  
- **Database:** IBM Cloudant  
- **Tools:** PyCharm / Spyder  
- **Platform:** IBM Cloud  

---

## 🧠 System Workflow
1. User uploads a fundus image via the web interface  
2. Flask server receives and preprocesses the image  
3. The trained deep learning model analyzes the image  
4. Prediction results are displayed  
5. User data and results are stored in the database  

---

## 📊 Dataset
- Retinal fundus images categorized by DR stages  
- Images resized to **299×299 pixels**  
- Divided into training and testing sets  

---

## 🔄 Data Preprocessing
- Image resizing  
- Normalization  
- Data augmentation:
  - Rotation  
  - Zoom  
  - Flipping  
  - Brightness adjustment  

---

## 🤖 Model Details
- Pretrained **Xception model** (ImageNet)  
- Custom layers added:
  - Flatten layer  
  - Dense layer  
  - Softmax output layer  

### Training Configuration
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Metrics: Accuracy  
- Epochs: 30  

---

## 🌐 Web Application Features
- User registration & login  
- Image upload functionality  
- Prediction display  
- Data storage in Cloudant DB  

---

## ✅ Advantages
- Early detection of diabetic retinopathy  
- Reduces dependency on specialists  
- Fast and scalable solution  
- Useful for remote healthcare  

---

## 📌 Applications
- Hospitals and clinics  
- Screening programs  
- Telemedicine platforms  
- Rural healthcare services  

---

## ⚠️ Limitations
- Depends on image quality  
- Requires internet connectivity  
- Cannot replace professional medical diagnosis  

---

## 🚀 Future Enhancements
- Multi-disease detection  
- Mobile application integration  
- Real-time image capture  
- Improved accuracy with larger datasets  

---

## 🧾 Conclusion
This project demonstrates how deep learning can assist in early detection of diabetic retinopathy. By integrating AI with a web application, it provides a practical and scalable healthcare solution.

---

## 📚 References
- TensorFlow Documentation  
- Keras Documentation  
- IBM Cloud Documentation  
- Research papers on Diabetic Retinopathy Detection  
- Towards Data Science articles  





but in this github file some of the files are not here due to size of them and also the hidden property of it hich is used in this code
