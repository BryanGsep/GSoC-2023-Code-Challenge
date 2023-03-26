# Machine learning Assistant for caMicroscope GSoC 2023 code Challenge

## Requirement
Create a frontend web application which uses tensorflow-js to provide some form of analysis on a user-supplied image unobtrusively. This should function on multiple timescales as possible, so that some information can be displayed immediately, while other slower-running calculations can be returned on completion.

## Ideas
Data preparation is a crucial but time-consuming task in machine learning, requiring specialized expertise to ensure accurate and reliable results. Therefore, having support systems in place is essential to streamline this process.

The frontend web service is designed with simplicity, effectiveness, and easy monitoring in mind, providing multi timescale support for efficient data preparation. Leveraging caMicroscope's pretraining model and the robust operation of tensorflow.js, the service can accurately label cell types, including Normal, Benign, In-situ Carcinoma, and Invasive Carcinoma, and extract features from image cell segments. Moreover, the web service is specifically tailored to support Whole Slide Images (WSI), making it an even more valuable tool for researchers and medical professionals.

## Models
### Two-Stage Convolutional Neural Network (Classification Model)
Input image size: 512x512
Classification list: Normal, Benign, In-situ Carcinoma, and Invasive Carcinoma.
Training dataset: https://rdm.inesctec.pt/dataset/nis-2017-003

### Y-Net (Segmentation Model)
Input image size: 256x256
Training dataset: https://www.bcsc-research.org/data/variables

## Image Usage Example (Most suitable)
Same as image use in training for Two-Stage Convolutional Neural Network
https://rdm.inesctec.pt/dataset/nis-2017-003

or you can download it from /src/images folder of this respo

## Usage Guidance
Step 0: Frontend Link 

Step 1: Upload unclassified images (support .tiff, .png, .jpeg) (Right after that image would be showed)

Step 2: You can choose "Classify Image", "Segment Image" to process image if you want.

Step 3: After finishing classifying process, "Download" button would be available. Download folder would classify images into corresponding folders.

## Deployment Method
Prequesite: Node.js installation globally

Step 1: Clone code from this repos
```bash
git clone https://github.com/BryanGsep/GSoC-2023-Code-Challenge.git

cd GSoC-2023-Code-Challenge
```

Step 2: Open terminal and Install dependency
```npm install```

Step 3: Run frontend web locally
```npm run start```

Step 4: Open live server url.
