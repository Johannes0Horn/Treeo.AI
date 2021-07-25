const MODEL_URL = 'deeplab_model_trained/tensorflowjs_model.pb';
const WEIGHTS_URL = 'deeplab_model_trained/weights_manifest.json';
const img_dir = '/trunk_diameter_testdata/';
const testdatatxt = '/trunk_diameter_testdata.txt'

var closingFactorCard = 10//10;
var openingFactorCard = 10//10;
var closingFactorTrunk = 60//60;
var openingFactorTrunk = 20//30;

var paddingInPixels = 100;
var orig_imgWidth = 224;
var orig_imgHeight = 224;
var allAccuracies = [];
var goneWrong = 0;
var txtFile = new XMLHttpRequest();
var lines = [];
var prozentualeAbweichungen = [];
var absolutAbweichungen = [];
var realDiameters = [];
var model;
var currentImageId = 0
var highestImageId = -1
var pause = 0

var meaErrorAdjustmentFactor = 1 //1 = unchanged

//todo:  Get diamtere of trunk every 10th line in between smallest rectangle
async function loadModel() {
    console.log("model loading..");
    let load_button = document.getElementById("load-button");
    load_button.innerHTML = "Model loading...";
    //model = await tf.loadGraphModel(MODEL_URL);
    model = await tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL);
    disableButton('load-button');
    load_button.disabled = true;
    load_button.innerHTML = "Model loaded!";
    console.log("model loaded successfully:");
    console.log(model);
}


async function Inference(currentImageId) {
    return new Promise((resolve, reject) => {
        //for each image!!
        let realDiameter = lines[currentImageId].split('_')[1] / Math.PI;
        let img_path = img_dir + lines[currentImageId] + ".png";
        $("#currentImage").html("ImgId: " + currentImageId + "   Img Name: " + lines[currentImageId]);


        let img = document.getElementById("test-image");
        img.onload = async function () {

            //let tensor = tf.browser.fromPixels(img)
            let tensor = tf.fromPixels(img)
                .resizeNearestNeighbor([orig_imgWidth, orig_imgHeight])
                .toFloat();
            tensor = tensor.expandDims();
            console.log("start network");
            let predictions = await model.execute(tensor).data();   //{["image_tensor"]: tensor}, "ImageTensor"
            console.log("network ran succesful");
            //CREATE PICTURE FROM PREDICTIONS
            var width = orig_imgWidth,
                height = orig_imgHeight,
                buffer = new Uint8ClampedArray(width * height * 4);
            //save all occurences
            let classifications = [];
            // row after row
            for (var y = 0; y < height; y++) {
                for (var x = 0; x < width; x++) {
                    var classification = predictions[y * width + x];
                    if (!classifications.includes(classification)) {
                        classifications.push(classification);
                    }
                    var pos = (y * width + x) * 4; // position in buffer based on x and y
                    if (classification == 0) {
                        buffer[pos] = 0;           // some R value [0, 255]
                        buffer[pos + 1] = 0;                     // some G value
                        buffer[pos + 2] = 0;                     // some B value
                        buffer[pos + 3] = 255;                 // set alpha channel

                    } else if (classification == 1) {

                        let val = classification + 50;
                        buffer[pos] = 127;           // some R value [0, 255]
                        buffer[pos + 1] = 127;                     // some G value
                        buffer[pos + 2] = 127;                     // some B value
                        buffer[pos + 3] = 255;                 // set alpha channel

                    } else if (classification == 2) {

                        let val = classification + 50;
                        buffer[pos] = 255;           // some R value [0, 255]
                        buffer[pos + 1] = 255;                     // some G value
                        buffer[pos + 2] = 255;                     // some B value
                        buffer[pos + 3] = 255;                 // set alpha channel

                    }

                }
            }
            // create off-screen canvas element
            var resultcanvas = document.createElement('canvas');
            var resultctx = resultcanvas.getContext('2d');

            resultcanvas.width = width;
            resultcanvas.height = height;

            // create imageData object
            var idata = resultctx.createImageData(width, height);

            // set our buffer as source
            idata.data.set(buffer);
            // update canvas with new data
            resultctx.putImageData(idata, 0, 0);
            // produce a PNG file
            var dataUri = resultcanvas.toDataURL();
            console.log(dataUri);
            document.getElementById("output-image").src = dataUri;


            //orig greyscale masks
            let mat = cv.imread(resultcanvas, 0);
            //binary card mask
            let card = new cv.Mat();
            //binary trunk mask
            let trunk = mat;

            //means every value below 254, will be set to 0, and above 254 to the value of 255

            cv.threshold(mat, card, 254, 255, cv.THRESH_BINARY);

            //for trunk its in range from 127 to 127 only
            for (let i = 0; i < trunk.rows; i++) {

                for (let j = 0; j < trunk.cols; j++) {

                    let editValue = trunk.ucharPtr(i, j);


                    if (editValue[0] != 127) //check whether value is within range.
                    {
                        for (let r = 0; r < 3; r++) {
                            trunk.ucharPtr(i, j)[r] = 0;
                        }
                    } else {
                        for (let r = 0; r < 3; r++) {
                            trunk.ucharPtr(i, j)[r] = 255;
                        }
                    }
                }
            }
            console.log("masks seperated");


            ///CARD
            //1.) RGBA to ONE CHANNEL
            cv.cvtColor(card, card, cv.COLOR_RGBA2GRAY, 0);
            console.log('card width: ' + card.cols + '\n' +
                'card height: ' + card.rows + '\n' +
                'card size: ' + card.size().width + '*' + card.size().height + '\n' +
                'card depth: ' + card.depth() + '\n' +
                'card channels ' + card.channels() + '\n' +
                'card type: ' + card.type() + '\n');

            //pad card so opening and closing doesnt affect edge of image
            let s = new cv.Scalar(0);
            cv.copyMakeBorder(card, card, paddingInPixels, paddingInPixels, paddingInPixels, paddingInPixels, cv.BORDER_CONSTANT, s);

            //2.) CLOSE AND OPEN OPERATION TO KILL NOISE

            //OPENING-->remove foreground noise
            let cardOpenFilter = cv.Mat.ones(openingFactorCard, openingFactorCard, cv.CV_8U);
            let openedCard = new cv.Mat();
            cv.morphologyEx(card, openedCard, cv.MORPH_OPEN, cardOpenFilter);

            //CLOSING--> remove background noise
            let cardCloseFilter = cv.Mat.ones(closingFactorCard, closingFactorCard, cv.CV_8U);
            let closedCard = new cv.Mat();
            cv.morphologyEx(openedCard, closedCard, cv.MORPH_CLOSE, cardCloseFilter);

            //unpad card so opening and closing doesnt affect edge of image
            let unpaddedCard = new cv.Mat();
            let orig_rect = new cv.Rect(paddingInPixels + 1, paddingInPixels + 1, orig_imgWidth, orig_imgHeight);
            unpaddedCard = closedCard.roi(orig_rect);

            cv.imshow('output-cardclosed', unpaddedCard);

            //3.) FIND COUNTOURS
            let cardCountoursDrawn = cv.Mat.zeros(orig_imgWidth, orig_imgHeight, cv.CV_8UC3);
            let contours = new cv.MatVector();
            let hierarchy = new cv.Mat();
            cv.findContours(unpaddedCard, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            console.log("card contours found");

            //choose biggest card contour as vaild one:
            //more than one contour?
            //get biggest area contours
            biggestContoursIndex = -1;
            biggestArea = 0;
            for (let i = 0; i < contours.size(); i++) {
                let area = cv.contourArea(contours.get(i));
                if (area > biggestArea) {
                    biggestArea = area;
                    biggestContoursIndex = i
                }
            }
            // DRAW COUNTOURS
            let color = new cv.Scalar(Math.round(Math.random() * 255), Math.round(Math.random() * 255),
                Math.round(Math.random() * 255));
            cv.drawContours(cardCountoursDrawn, contours, biggestContoursIndex, color, 1, cv.LINE_8, hierarchy, 100);

            console.log("card contours drawn");

            //4.) FIND MIN AREA RECT OF CONTOURS
            let cardRect = cv.Mat.zeros(orig_imgWidth, orig_imgHeight, cv.CV_8UC3);


            let cardRotatedRect = cv.minAreaRect(contours.get(biggestContoursIndex));
            let cardVertices = cv.RotatedRect.points(cardRotatedRect);
            let cardRectangleColor = new cv.Scalar(255, 0, 0);
            //DRAW MIN AREA RECT OF CONTOURS
            console.log("card begin draw rectangle");
            for (let i = 0; i < 4; i++) {
                cv.line(cardRect, cardVertices[i], cardVertices[(i + 1) % 4], cardRectangleColor, 2, cv.LINE_AA, 0);
            }

            // 5.) GET SIZE OF CARD_RECTANGLE IN PIXELS
            function getRange(x1, y1, x2, y2) {
                return Math.sqrt(Math.pow((x2 - x1), 2) + Math.pow((y2 - y1), 2));
            }

            //check range to each point from point [0], second most far away is point to longer side
            let range0_to_1 = getRange(cardVertices[0]["x"], cardVertices[0]["y"], cardVertices[1]["x"], cardVertices[1]["y"]);
            let range0_to_2 = getRange(cardVertices[0]["x"], cardVertices[0]["y"], cardVertices[2]["x"], cardVertices[2]["y"]);
            let range0_to_3 = getRange(cardVertices[0]["x"], cardVertices[0]["y"], cardVertices[3]["x"], cardVertices[3]["y"]);
            //console.log(range0_to_1);
            //console.log(range0_to_2);
            //console.log(range0_to_3);
            //get second biggest
            let distances = [range0_to_1, range0_to_2, range0_to_3];
            distances.sort(function (a, b) {
                return a - b
            });
            let cardlongerSide = distances[1];
            let cardshorterSide = distances [0];
            //console.log("cardlong: ", cardlongerSide);
            //console.log("cardshort: ", cardshorterSide);


            ///TRUNK
            //1.) RGBA to ONE CHANNEL
            cv.cvtColor(trunk, trunk, cv.COLOR_RGBA2GRAY, 0);
            console.log('trunk width: ' + trunk.cols + '\n' +
                'trunk height: ' + trunk.rows + '\n' +
                'trunk size: ' + trunk.size().width + '*' + trunk.size().height + '\n' +
                'trunk depth: ' + trunk.depth() + '\n' +
                'trunk channels ' + trunk.channels() + '\n' +
                'trunk type: ' + trunk.type() + '\n');

            //pad card so opening and closing doesnt affect edge of image
            cv.copyMakeBorder(trunk, trunk, paddingInPixels, paddingInPixels, paddingInPixels, paddingInPixels, cv.BORDER_CONSTANT, s);

            //2.) CLOSE OPERATION TO KILL NOISE AND CONNECT MASKS
            //OPENING
            let trunkOpenFilter = cv.Mat.ones(openingFactorTrunk, openingFactorTrunk, cv.CV_8U);
            let openedTrunk = new cv.Mat();
            cv.morphologyEx(trunk, openedTrunk, cv.MORPH_OPEN, trunkOpenFilter);

            //CLOSING
            let trunkCloseFilter = cv.Mat.ones(closingFactorTrunk, closingFactorTrunk, cv.CV_8U);
            let closedTrunk = new cv.Mat();
            cv.morphologyEx(openedTrunk, closedTrunk, cv.MORPH_CLOSE, trunkCloseFilter);

            //unpad card so opening and closing doesnt affect edge of image
            let unpaddedTrunk = new cv.Mat();
            unpaddedTrunk = closedTrunk.roi(orig_rect);

            //3.) FIND COUNTOURS
            let trunkCountoursDrawn = cv.Mat.zeros(orig_imgWidth, orig_imgHeight, cv.CV_8UC3);
            contours = new cv.MatVector();
            hierarchy = new cv.Mat();
            cv.findContours(unpaddedTrunk, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            console.log("trunk contours found");
            // DRAW COUNTOURS
            for (let i = 0; i < contours.size(); ++i) {
                let color = new cv.Scalar(Math.round(Math.random() * 255), Math.round(Math.random() * 255),
                    Math.round(Math.random() * 255));
                cv.drawContours(trunkCountoursDrawn, contours, i, color, 1, cv.LINE_8, hierarchy, 100);
            }
            console.log("trunk contours drawn");
            //4.) FIND MIN AREA RECT OF CONTOURS
            //TODO: MAYBE USE ONLY ONE CONTOUR:merge 2 biggest Area contours(if more than 1) to one remaining contour to feed to convexHull and get average diameter from it
            // TODO: If 2 masks, check if both have realistic size
            let trunkRects = cv.Mat.zeros(orig_imgWidth, orig_imgHeight, cv.CV_8UC3);
            let trunkRectangleColor = new cv.Scalar(255, 0, 0);

            let trunklongerSide;
            let trunkshorterSide;
            let diameter1;
            let diameter2;
            //console.log("number of contours: ", contours.size());


            if (contours.size() > 1) {
                //more than one contour
                //get 2 biggest area contours
                biggestContoursIndexes = [];
                biggestAreas = [];
                for (let i = 0; i < contours.size(); i++) {
                    let area = cv.contourArea(contours.get(i));
                    if (biggestAreas.length >= 2) {
                        if (area > biggestAreas[0]) {
                            biggestAreas[0] = area;
                            biggestContoursIndexes[0] = i;
                        } else if (area > biggestAreas[1]) {
                            biggestAreas[1] = area;
                            biggestContoursIndexes[1] = i;
                        }
                    } else {
                        biggestAreas.push(area);
                        biggestContoursIndexes.push(i);
                    }
                }

                //get rectangles from contours
                let trunkRotatedRect1 = cv.minAreaRect(contours.get(biggestContoursIndexes[0]));
                let trunkVertices1 = cv.RotatedRect.points(trunkRotatedRect1);

                let trunkRotatedRect2 = cv.minAreaRect(contours.get(biggestContoursIndexes[1]));
                let trunkVertices2 = cv.RotatedRect.points(trunkRotatedRect2);
                //draw rectangels

                //DRAW MIN AREA RECT OF CONTOURS
                console.log("card begin draw rectangle");
                for (let i = 0; i < 4; i++) {
                    cv.line(trunkRects, trunkVertices1[i], trunkVertices1[(i + 1) % 4], trunkRectangleColor, 2, cv.LINE_AA, 0);
                    cv.line(trunkRects, trunkVertices2[i], trunkVertices2[(i + 1) % 4], trunkRectangleColor, 2, cv.LINE_AA, 0);

                }

                console.log("above each other");
                //assume above each other
                //RECTANGLE 1
                //get 2 highest points
                let highestPoint1 = [trunkVertices1[0]["x"], trunkVertices1[0]["y"]];
                let highestPoint2 = [trunkVertices1[1]["x"], trunkVertices1[1]["y"]];

                for (let i = 2; i < 4; i++) {
                    if (trunkVertices1[i]["y"] > highestPoint1[1]) {
                        highestPoint1[0] = trunkVertices1[i]["x"];
                        highestPoint1[1] = trunkVertices1[i]["y"];
                    } else if (trunkVertices1[i]["y"] > highestPoint2[1]) {
                        highestPoint2[0] = trunkVertices1[i]["x"];
                        highestPoint2[1] = trunkVertices1[i]["y"];
                    }
                }
                console.log("highestPoint1.1", highestPoint1);
                console.log("highestPoint1.2", highestPoint2);


                //get side above trunk
                diameter1 = getRange(highestPoint1[0], highestPoint1[1], highestPoint2[0], highestPoint2[1]);
                //RECTANGLE 2
                //get 2 highest points
                highestPoint1 = [trunkVertices2[0]["x"], trunkVertices2[0]["y"]];
                highestPoint2 = [trunkVertices2[1]["x"], trunkVertices2[1]["y"]];

                for (let i = 2; i < 4; i++) {
                    if (trunkVertices2[i]["y"] > highestPoint1[1]) {
                        highestPoint1[0] = trunkVertices2[i]["x"];
                        highestPoint1[1] = trunkVertices2[i]["y"];
                    } else if (trunkVertices2[i]["y"] > highestPoint2[1]) {
                        highestPoint2[0] = trunkVertices2[i]["x"];
                        highestPoint2[1] = trunkVertices2[i]["y"];
                    }
                }
                console.log("highestPoint2.1", highestPoint1);
                console.log("highestPoint2.2", highestPoint2);

                //get side above trunk
                diameter2 = getRange(highestPoint1[0], highestPoint1[1], highestPoint2[0], highestPoint2[1]);
                console.log("diamter1:", diameter1);
                console.log("diamter2:", diameter2);


                trunkshorterSide = (diameter1 + diameter2) / 2;


            } else {
                //only one Contour!
                let trunkRotatedRect = cv.minAreaRect(contours.get(0));
                let trunkVertices = cv.RotatedRect.points(trunkRotatedRect);

                //DRAW MIN AREA RECT OF CONTOURS
                console.log("card begin draw rectangle");
                for (let i = 0; i < 4; i++) {
                    cv.line(trunkRects, trunkVertices[i], trunkVertices[(i + 1) % 4], trunkRectangleColor, 2, cv.LINE_AA, 0);
                }

                // 5.) GET SIZE OF TRUNK_RECTANGLE IN PIXELS
                //check range to each point from point [0], second most far away is point to longer side
                range0_to_1 = getRange(trunkVertices[0]["x"], trunkVertices[0]["y"], trunkVertices[1]["x"], trunkVertices[1]["y"]);
                range0_to_2 = getRange(trunkVertices[0]["x"], trunkVertices[0]["y"], trunkVertices[2]["x"], trunkVertices[2]["y"]);
                range0_to_3 = getRange(trunkVertices[0]["x"], trunkVertices[0]["y"], trunkVertices[3]["x"], trunkVertices[3]["y"]);
                //console.log(range0_to_1);
                //console.log(range0_to_2);
                //console.log(range0_to_3);
                //get second biggest
                distances = [range0_to_1, range0_to_2, range0_to_3];
                distances.sort(function (a, b) {
                    return a - b
                });
                trunklongerSide = distances[1];
                trunkshorterSide = distances [0];
            }


            //console.log("trunklong: ", trunklongerSide);
            console.log("trunkshort: ", trunkshorterSide);
            console.log("cardlong: ", cardlongerSide);
            console.log("cardshort: ", cardshorterSide);

            //COMPARE SIZES TO ESTIMATE DIAMETER
            let cardLength = 856; //mm
            let pixelSize = cardLength / cardlongerSide;
            let trunkDiameter = (trunkshorterSide * pixelSize) / 100;  //cm
            //adjust mean error
            trunkDiameter = trunkDiameter * meaErrorAdjustmentFactor;

            if (trunkDiameter > 0) {


                let currentaccuracy = 100 - Math.abs((realDiameter - trunkDiameter) / realDiameter * 100);
                console.log(currentaccuracy);
                document.getElementById("currentaccuracy").innerHTML = "Real Diameter: " + realDiameter + "<br/>" + "Estimated: " + trunkDiameter + "<br/>" + 'accuracy: ' + currentaccuracy;
                if (currentImageId >= highestImageId) {
                    allAccuracies.push(currentaccuracy);
                    realDiameters.push(realDiameter);
                    absolutAbweichungen.push(trunkDiameter - realDiameter);
                    var abweichungsRatio = trunkDiameter / realDiameter;
                    if (abweichungsRatio === 1) {
                        prozentualeAbweichungen.push(0);
                    } else if (abweichungsRatio > 1) {
                        prozentualeAbweichungen.push((abweichungsRatio - 1) * 100);
                    } else {
                        prozentualeAbweichungen.push((1 - abweichungsRatio) * (-100));
                    }
                    //update mean procentual Error:
                    var sumProcentualErrors = 0;
                    for (var i = 0; i < prozentualeAbweichungen.length; i++) {
                        sumProcentualErrors += prozentualeAbweichungen[i];
                    }
                    var meanProcentualError = sumProcentualErrors / absolutAbweichungen.length;
                    document.getElementById("meanprocentaulerror").innerHTML = "Mean Procentaul Error:" + meanProcentualError + "%" +"<br/>";

                    //update mean absolute Error:
                    var sumAbsoluteErrors = 0;
                    for (var i = 0; i < absolutAbweichungen.length; i++) {
                        sumAbsoluteErrors += absolutAbweichungen[i];
                    }
                    var meanAbsoluteError = sumAbsoluteErrors / absolutAbweichungen.length;
                    document.getElementById("meanabsoluteerror").innerHTML = "Mean Absolute Error:" + meanAbsoluteError + "<br/>";

                    //update mean Accuracy
                    var sumAccuracies = 0;
                    for (var i = 0; i < allAccuracies.length; i++) {
                        sumAccuracies += allAccuracies[i];
                    }
                    var meanAccuracy = sumAccuracies / allAccuracies.length;
                    document.getElementById("meanaccuracy").innerHTML = "Mean Accuracy:" + meanAccuracy + "<br/>" + 'Insgesamt fehlgeschlagen:' + goneWrong + "<br/>" + "Bilder Verarbeitet:" + (highestImageId + 1);
                }


                console.log("EstimatedDiameter: ", trunkDiameter);
                console.log("RealDiameter: ", realDiameter);
            } else {
                goneWrong++;
                console.log("goneWrong:" + goneWrong)
            }
            cv.imshow('output-cardrectangle', cardRect);
            cv.imshow('output-trunkrectangle', trunkRects);
            cv.imshow('output-trunkclosed', unpaddedTrunk);

            resolve("inference_success");
        }
        //triggers onload
        img.src = img_path;

    });

}

async function startInferenceLoop() {
    disableButton('predict-button');
    disableButton('next-button');
    disableButton('previous-button');
    enableButton('pause-button');

    $("#predict-button").innerHTML = "Test is running...";
    pause = 0;
    while (pause == 0 && currentImageId < lines.length) {
        await Inference(currentImageId);
        currentImageId++;
        if (highestImageId < currentImageId) {
            highestImageId = currentImageId;
        }
    }
    enableButton('predict-button');
    enableButton('next-button');
    enableButton('previous-button');
    disableButton('pause-button');
}


async function stopInferenceLoop() {
    disableButton('pause-button');
    pause = 1;
    document.getElementById("predict-button").innerHTML = "Resume";


}

async function nextInference() {
    disableButton('predict-button');
    disableButton('next-button');
    disableButton('previous-button');
    document.getElementById("predict-button").innerHTML = "Test is running...";
    await Inference(currentImageId);
    currentImageId++;
    if (highestImageId < currentImageId) {
        highestImageId = currentImageId;
    }
    document.getElementById("predict-button").innerHTML = "Resume";
    enableButton('predict-button');
    enableButton('next-button');
    enableButton('previous-button');
}

async function previousInference() {
    disableButton('predict-button');
    disableButton('next-button');
    disableButton('previous-button');
    document.getElementById("predict-button").innerHTML = "Test is running...";
    currentImageId = currentImageId - 2;
    if (currentImageId < 0) {
        currentImageId = 0
    }
    await Inference(currentImageId);
    currentImageId++;
    if (highestImageId < currentImageId) {
        highestImageId = currentImageId;
    }
    document.getElementById("predict-button").innerHTML = "Resume";
    enableButton('predict-button');
    enableButton('next-button');
    enableButton('previous-button');
}

$(document).ready(function () {
    console.log("document ready!");

    document.getElementById("predict-box").style.display = "table-cell";
    document.getElementById("image-box").style.display = "table-cell";
    document.getElementById("test-image").crossorigin = "anonymous";
    document.getElementById("rectangle-box").style.display = "table-cell";
    document.getElementById("accuracy-box").style.display = "table-cell";
    document.getElementById("openclose-box").style.display = "table-cell";

    //open textFile
    txtFile.open("GET", testdatatxt, true);
    txtFile.onreadystatechange = function () {
        if (txtFile.readyState === 4) {  // Makes sure the document is ready to parse.
            if (txtFile.status === 200) {  // Makes sure it's found the file.
                let allText = txtFile.responseText;
                lines = allText.split("\n"); // Will separate each line into an array
            }
        }
    };
    //trigger load txt
    txtFile.send(null);

    $("#load-button").click(loadModel);
    $("#predict-button").click(startInferenceLoop);
    $("#pause-button").click(stopInferenceLoop);
    $("#next-button").click(nextInference);
    $("#previous-button").click(previousInference);

    disableButton('pause-button');

});

function disableButton(id) {
    var currentButton = $("#" + id);
    currentButton.prop("disabled", true);
    currentButton.addClass("disabledButton");
}

function enableButton(id) {
    var currentButton = $("#" + id);
    currentButton.prop("disabled", false);
    currentButton.removeClass("disabledButton");
}
