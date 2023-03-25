// Using a pre-trained TensorFlow.js model based on the Two-Stage Convolutional Neural Network for Breast Cancer Histology Image Classification paper
// Based on: Nazeri, K., Aminpour, A., & Ebrahimi, M. (2018). Two-Stage Convolutional Neural Network for Breast Cancer Histology Image Classification. In International Conference Image Analysis and Recognition (pp. 717-726). Springer.

// Using Y-Net, a joint segmentation and classification method for diagnosis of breast biopsy images
// Based on: Mehta, S., Mercan, E., Bartlett, J., Weaver, D., Elmore, J., & Shapiro, L. (2018). Y-Net: Joint Segmentation and Classification for Diagnosis of Breast Biopsy Images. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. -). Springer.

// This code is based on research from Teresa Araújo et al.
// See: Araújo, T., Aresta, G., Castro, E., Rouco, J., Aguiar, P., Eloy, C., Polónia, A., & Campilho, A. (2017).
// Classification of Breast Cancer Histology Images Using Convolutional Neural Networks. PLOS ONE.

// Load the pre-trained model
const CLASSIFY_IMAGE_SIZE = 512;
const SEGMENTATION_IMAGE_SIZE = 256;
const DISPLAY_IMAGE_WIDTH = 256;
const DISPLAY_IMAGE_HEIGHT = 256*3/4;
const classifyModelUrl = './models/BACH/model.json';
const segmentationModelUrl = './models/YNet/model.json';

const fileInput = document.getElementById('fileInput');
const classifyButton = document.getElementById('classify-button');
const segmentationButton = document.getElementById('segmentation-button');
const downloadButton = document.getElementById('download-button');
const imageContainers = document.getElementById('image-container');
const classifyProgressBar = document.getElementById('classify-progress-bar');
const segmentProgressBar = document.getElementById('segment-progress-bar');

downloadButton.disabled = true;
// Load the model and weights using the URLs
async function loadClassifyModel() {
    const model = await tf.loadLayersModel(classifyModelUrl);
    return model;
}

async function loadSegmentationModel() {
    const model = await tf.loadLayersModel(segmentationModelUrl);
    return model;
}

function cropImage(img, x, y, width, height) {
    return new Promise((resolve, reject) => {
        // create a canvas element with the desired dimensions
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, x, y, width, height, 0, 0, width, height);

        const croppedImg = new Image();
        croppedImg.width = width;
        croppedImg.height = height;
        croppedImg.onload = () => {
            resolve(croppedImg);
        };
        croppedImg.onerror = () => {
            reject(new Error('Error loading cropped image'));
        };
        croppedImg.src = canvas.toDataURL();
    });
}

class PatchExtractor {
    constructor(img, patch_size, stride) {
        this.img = img;
        this.size = patch_size;
        this.stride = stride;
    }

    extract_patches() {
        let [wp, hp] = this.shape();
        let patches = [];
        for (let h = 0; h < hp; h++) {
            for (let w = 0; w < wp; w++) {
                patches.push(this.extract_patch([w, h]));
            }
        }
        return patches;
    }

    extract_patch(patch) {
        return cropImage(this.img, patch[0] * this.stride, patch[1] * this.stride, this.size, this.size);
    }

    shape() {
        let wp = Math.floor((this.img.width - this.size) / this.stride + 1);
        let hp = Math.floor((this.img.height - this.size) / this.stride + 1);
        return [wp, hp];
    }
}

// Classify all uploaded images
async function classifyImages() {
    downloadButton.disabled = true;
    // Load the model
    let counter = 0;
    classifyProgressBar.setAttribute("max", `${fileInput.files.length}`);
    classifyProgressBar.setAttribute("value", `${counter}`);
    classifyProgressBar.setAttribute("style", "");
    const classifyModel = await loadClassifyModel();

    // Loop over all selected files
    for (const file of fileInput.files) {
        const imgElement = new Image();
        imgElement.alt = file.name;
        imgElement.onload = async () => {
            const extractor = new PatchExtractor(imgElement, CLASSIFY_IMAGE_SIZE, CLASSIFY_IMAGE_SIZE);
            const patches = extractor.extract_patches();
            Promise.all(patches)
                .then(async (patchElements) => {
                    const predictions = []
                    for (const patch of patchElements) {
                        const imageTensor = tf.browser.fromPixels(patch).reshape([-1, CLASSIFY_IMAGE_SIZE, CLASSIFY_IMAGE_SIZE, 3]);
                        const prediction = await classifyModel.predict(imageTensor).data().catch((error) => console.log(error));
                        if (prediction) {
                            predictions.push(prediction);
                        }
                    }
                    let averagePredictions = [0,0,0,0];
                    for (const prediction of predictions) {
                        for (let i=0; i<prediction.length; i++) {
                            averagePredictions[i] += prediction[i];
                        }
                    }
                    averagePredictions = averagePredictions.map((predictionSum) => predictionSum/predictions.length)
                    // choose class with highest possibility
                    const highestIndex = averagePredictions.indexOf(Math.max(...averagePredictions));
                    const classNames = ['Normal', 'Benign', 'In-situ Carcinoma', 'Invasive Carcinoma'];
                    
                    // Create a new <p> element to display the classification result
                    const resultElement = document.createElement('h5');
                    const displayImageElement = document.getElementById("display-image-"+file.name.toString());
                    displayImageElement.setAttribute("classify", `${classNames[highestIndex]}`);
                    resultElement.innerText = `${classNames[highestIndex]}`;
                    resultElement.setAttribute("class", "card-title");
                    
                    // Append the result to the corresponding image container
                    const imageContainerbody = document.getElementById("body-"+file.name.toString());
                    imageContainerbody.insertBefore(resultElement, imageContainerbody.children[0]);
                    counter++;
                    console.log(counter);
                    console.log(fileInput.files.length);
                    if (counter < fileInput.files.length) {
                        classifyProgressBar.setAttribute("max", `${fileInput.files.length}`);
                        classifyProgressBar.setAttribute("value", `${counter}`);
                    } else {
                        classifyProgressBar.setAttribute("style", "display: none;")
                        downloadButton.disabled = false;
                    }
                })
        };
        if (file.name.endsWith("tiff") || file.name.endsWith("tif")){
            imgElement.src = await getTiffImageURL(URL.createObjectURL(file)).then();
        } else {
            imgElement.src = URL.createObjectURL(file);
        }
    }
}

function maskCanvas(img, mask) {
    const canvas = document.createElement("CANVAS");
    const ctx = canvas.getContext("2d");
    canvas.height = img.height;
    canvas.width = img.width;
    ctx.drawImage(img, 0, 0);
    const imgData = ctx.getImageData(0, 0, img.width, img.height);
    for(let i=0; i<img.width; i++) {
        for(let j=0; j<img.height; j++) {
            if (mask[j*img.width+i] > 0.3) {
                imgData.data[(j*img.width+i)*4] = 0;
                imgData.data[(j*img.width+i)*4+1] = 0;
                imgData.data[(j*img.width+i)*4+2] = 0;
                imgData.data[(j*img.width+i)*4+3] = 255;
            }
        }
    }
    ctx.putImageData(imgData, 0, 0);
    return canvas;
}

function handleImageOnLoad(ctx, image, dx, dy) {
    ctx.drawImage(image, 0, 0, SEGMENTATION_IMAGE_SIZE, SEGMENTATION_IMAGE_SIZE, dx, dy, SEGMENTATION_IMAGE_SIZE, SEGMENTATION_IMAGE_SIZE);
}

async function segmentateImage() {
    let counter = 0;
    segmentProgressBar.setAttribute("max", `${fileInput.files.length}`);
    segmentProgressBar.setAttribute("value", `${counter}`);
    segmentProgressBar.setAttribute("style", "");
    const segmentationModel = await loadSegmentationModel();
    // run segmentation model
    
    for (let file of fileInput.files) {
        const imgElement = new Image();
        imgElement.alt = file.name;
        imgElement.onload = async () => {
            // change image data into tensor
            const imgOrigWidth = imgElement.width;
            const imgOrigHeight = imgElement.height;
            const extractor = new PatchExtractor(imgElement, SEGMENTATION_IMAGE_SIZE, SEGMENTATION_IMAGE_SIZE);
            const patches = extractor.extract_patches();
            const shapes = extractor.shape();
            // Create new canvas
            const mainCanvas = document.createElement("CANVAS");
            const mainCtx = mainCanvas.getContext("2d");
            mainCanvas.setAttribute("id", "main-canvas");
            mainCanvas.width = imgOrigWidth;
            mainCanvas.height = imgOrigHeight;
            Promise.all(patches)
                .then((patchElements) => {
                    const promiseList = []
                    for (let i=0; i<patchElements.length; i++) {
                        const segImageTensor = tf.browser.fromPixels(patchElements[i]).reshape([-1, SEGMENTATION_IMAGE_SIZE, SEGMENTATION_IMAGE_SIZE, 3]);
                        promiseList.push(segmentationModel.predict(segImageTensor).data());
                    }
                    Promise.all(promiseList)
                        .then((masks) => {
                            const loadedImages = masks.map(
                                    (mask, index) => {
                                        return new Promise((resolve, reject) => {
                                            const maskedCanvas = maskCanvas(patchElements[index], mask);
                                            const tmpImage = new Image();
                                            const dx = (index%shapes[0])*SEGMENTATION_IMAGE_SIZE;
                                            const dy = (Math.floor(index/shapes[0]))*SEGMENTATION_IMAGE_SIZE;
                                            tmpImage.width = SEGMENTATION_IMAGE_SIZE;
                                            tmpImage.height = SEGMENTATION_IMAGE_SIZE;
                                            tmpImage.onload = () => {
                                                handleImageOnLoad(mainCtx, tmpImage, dx, dy);
                                                resolve();
                                            }
                                            tmpImage.onerror = () => {
                                                reject();
                                            }
                                            tmpImage.src = maskedCanvas.toDataURL();
                                        })
                                    }
                                )

                            const displayImageElement = document.getElementById("display-image-"+file.name.toString());
                            Promise.all(loadedImages)
                                    .then(() => {
                                        counter++;
                                        if (counter < fileInput.files.length) {
                                            segmentProgressBar.setAttribute("max", `${fileInput.files.length}`);
                                            segmentProgressBar.setAttribute("value", `${counter}`);
                                        } else {
                                            segmentProgressBar.setAttribute("style", "display: none;")
                                        }
                                        displayImageElement.src = mainCanvas.toDataURL();
                                    })
                        })
                })
        };
        if (file.name.endsWith("tiff") || file.name.endsWith("tif")){
            imgElement.src = await getTiffImageURL(URL.createObjectURL(file)).then();
        } else {
            imgElement.src = URL.createObjectURL(file);
        };
    }
}

function getTiffImageURL(URL) {
    return new Promise((resolve, reject) => {
        var xhr = new XMLHttpRequest();
        xhr.responseType = 'arraybuffer';
        xhr.open('GET', URL);
        xhr.onload = function (e) {
            var tiff = new Tiff({buffer: xhr.response});
            var canvas = tiff.toCanvas();
            resolve(canvas.toDataURL())
        };
        xhr.send();
    })
}

Tiff.initialize({TOTAL_MEMORY: 19777216 * 10});

// Show iamge after loading
fileInput.addEventListener('change', async () => {
    // Clear any previous results
    imageContainers.innerHTML = '';
    for (const file of fileInput.files) {
        // create and image container element
        const imageContainer = document.createElement("DIV");
        imageContainer.setAttribute("id", file.name.toString());
        imageContainer.setAttribute("class", "card");
        imageContainer.setAttribute("style", `display: inline ;width: ${DISPLAY_IMAGE_WIDTH}px;`)

        // create card-body element
        const imageContainerBody = document.createElement("DIV");
        imageContainerBody.setAttribute("class", "card-body");
        imageContainerBody.setAttribute("id", "body-"+file.name.toString());

        // create and display image element
        const displayImageElement = new Image();
        displayImageElement.width = DISPLAY_IMAGE_WIDTH;
        displayImageElement.height = DISPLAY_IMAGE_HEIGHT;
        if (file.name.endsWith("tiff") || file.name.endsWith("tif")){
            displayImageElement.src = await getTiffImageURL(URL.createObjectURL(file)).then();
        } else {
            displayImageElement.src = URL.createObjectURL(file);
        }
        displayImageElement.setAttribute("id", "display-image-"+file.name.toString());
        displayImageElement.setAttribute("class", "card-img-top");

        // Create a new <p> element to display the filename
        const filenameElement = document.createElement('p');
        filenameElement.innerText = `Filename: ${file.name}`;
        filenameElement.setAttribute("class", "card-text")
        
        // Append the filename and image elements to the image container
        imageContainer.appendChild(displayImageElement);
        imageContainer.appendChild(imageContainerBody);
        imageContainerBody.appendChild(filenameElement);

        // Append the imageContainer into main containter
        imageContainers.appendChild(imageContainer);
    }
})

function download() {
    const images = document.querySelectorAll('img');
    const zip = new JSZip();
    const imagesByCategory = {};
    images.forEach((image) => {
        const category = image.getAttribute("classify");
        if (!imagesByCategory[category]) {
            imagesByCategory[category] = [];
        }
        imagesByCategory[category].push(image);
    });

    const categoryImagePromiseList = Object.entries(imagesByCategory).map((image_cat) => {
        return new Promise((resolve, reject) => {
            const category = image_cat[0];
            const categoryImages = image_cat[1];
            const categoryFolder = zip.folder(category);
            const imagePromiseList = categoryImages.map((image, index) => {
                return new Promise((resolve, reject) => {
                    const filename = `${category}_image_${index}.tiff`;
                    fetch(image.src)
                        .then((response) => response.blob())
                        .then((blob) => {
                            categoryFolder.file(filename, blob);
                            resolve();
                        })
                        .catch(()=> {
                            reject();
                        });
                })
            });

            Promise.all(imagePromiseList)
                .then(() => {
                    resolve();
                })
                .catch(() => {
                    reject();
                });
        });
    })

    Promise.all(categoryImagePromiseList)
        .then(() => {
            zip.generateAsync({ type: 'blob' }).then((blob) => {
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = 'caMicroscope.zip';
                document.body.appendChild(link);
                link.click();
            });
        })
}

classifyButton.addEventListener('click', classifyImages);
segmentationButton.addEventListener('click', segmentateImage);
downloadButton.addEventListener('click', download);
