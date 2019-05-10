//Class to Calculate diameter of trunk from one dimensional predictions Array with length 224*224 = 50176 of deepLabV3+
//, trained with treeo dataset.


class MasksToDiameterService {

    constructor() {
        this.width = 224;
        this.height = 224;
        this.closingFactor = 20;
        this.openingFactor = 20;
        this.buffer = new Uint8ClampedArray(this.width * this.height * 4);
        //save all Occurrences
        this.classifications = [];
    }


// Input: - predictions from deepLabV3+
// Outputs: - 0, if no card or no trunk detected or if predictions has not the size of 50176
//          - Diameter in cm, if card and trunk are detected and if predictions has size of 50176
//REQUIRES OPENCV to work. Can be downloaded from here: https://docs.opencv.org/master/opencv.js
    predict(predictions, cv) {
        //check if predictions has correct size
        if (predictions.length !== this.width * this.height) {
            console.log("predictions has wrong size");
            return 0;
        }
        //CREATE PICTURE FROM PREDICTIONS
        // row after row
        for (let y = 0; y < this.height; y++) {
            //index after index
            for (let x = 0; x < this.width; x++) {
                let classification = predictions[y * this.width + x];
                if (!this.classifications.includes(classification)) {
                    this.classifications.push(classification);
                }
                let pos = (y * this.width + x) * 4; // position in buffer based on x and y
                //background
                if (classification === 0) {
                    this.buffer[pos] = 0;           // some R value [0, 255]
                    this.buffer[pos + 1] = 0;                     // some G value
                    this.buffer[pos + 2] = 0;                     // some B value
                    this.buffer[pos + 3] = 255;                 // set alpha channel

                }
                //trunk
                else if (classification === 1) {
                    this.buffer[pos] = 127;           // some R value [0, 255]
                    this.buffer[pos + 1] = 127;                     // some G value
                    this.buffer[pos + 2] = 127;                     // some B value
                    this.buffer[pos + 3] = 255;                 // set alpha channel

                }
                //card
                else if (classification === 2) {
                    this.buffer[pos] = 255;           // some R value [0, 255]
                    this.buffer[pos + 1] = 255;                     // some G value
                    this.buffer[pos + 2] = 255;                     // some B value
                    this.buffer[pos + 3] = 255;                 // set alpha channel

                }

            }
        }
        //Check if no card or no trunk Pixel was detected
        if (!this.classifications.includes(1)) {
            //console.log("no trunk pixel detected");
            return 0;
        }
        if (!this.classifications.includes(2)) {
            //console.log("no card pixel detected");
            return 0;
        }

        // create off-screen canvas element
        let canvas = document.createElement('canvas');
        let ctx = canvas.getContext('2d');

        canvas.width = this.width;
        canvas.height = this.height;

        // create imageData object
        let idata = ctx.createImageData(this.width, this.height);

        // set our buffer as source
        idata.data.set(this.buffer);
        // update canvas with new data
        ctx.putImageData(idata, 0, 0);


        //orig greyscale masks
        let mat = cv.imread(canvas, 0);
        //binary card mask
        let card = new cv.Mat();
        //binary trunk mask
        let trunk = mat;

        //seperate card mask
        //means every value below 254, will be set to 0, and above 254 to the value of 255
        cv.threshold(mat, card, 254, 255, cv.THRESH_BINARY);

        //seperate trunk mask
        //for trunk its in range from 127 to 127 only
        for (let i = 0; i < trunk.rows; i++) {
            for (let j = 0; j < trunk.cols; j++) {
                let editValue = trunk.ucharPtr(i, j);
                if (editValue[0] !== 127) //check whether value is within range.
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
        //console.log("masks seperated");


        ///get size of CARD
        //1.) RGBA to ONE CHANNEL
        cv.cvtColor(card, card, cv.COLOR_RGBA2GRAY, 0);
        /*
        console.log('card width: ' + card.cols + '\n' +
            'card height: ' + card.rows + '\n' +
            'card size: ' + card.size().width + '*' + card.size().height + '\n' +
            'card depth: ' + card.depth() + '\n' +
            'card channels ' + card.channels() + '\n' +
            'card type: ' + card.type() + '\n');
        */
        //2.) CLOSE OPERATION TO KILL NOISE
        //close and open to kill noise
        let cardCloseFilter = cv.Mat.ones(this.closingFactor, this.closingFactor, cv.CV_8U);
        let closedCard = new cv.Mat();
        cv.morphologyEx(card, closedCard, cv.MORPH_CLOSE, cardCloseFilter);
        let cardOpenFilter = cv.Mat.ones(this.openingFactor, this.openingFactor, cv.CV_8U);
        let openedCard = new cv.Mat();
        cv.morphologyEx(closedCard, openedCard, cv.MORPH_OPEN, cardOpenFilter);


        //3.) FIND COUNTOURS
        let cardCountoursDrawn = cv.Mat.zeros(card.cols, card.rows, cv.CV_8UC3);
        let contours = new cv.MatVector();
        let hierarchy = new cv.Mat();
        cv.findContours(openedCard, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
        //console.log("card contours found");

        // DRAW COUNTOURS
        for (let i = 0; i < contours.size(); ++i) {
            let color = new cv.Scalar(Math.round(Math.random() * 255), Math.round(Math.random() * 255),
                Math.round(Math.random() * 255));
            cv.drawContours(cardCountoursDrawn, contours, i, color, 1, cv.LINE_8, hierarchy, 100);
        }
        //console.log("card contours drawn");

        //4.) FIND MIN AREA RECT OF CONTOURS
        let cardRect = cv.Mat.zeros(card.rows, card.cols, cv.CV_8UC3);
        let cardRotatedRect = cv.minAreaRect(contours.get(0));
        let cardVertices = cv.RotatedRect.points(cardRotatedRect);
        let cardRectangleColor = new cv.Scalar(255, 0, 0);
        //DRAW MIN AREA RECT OF CONTOURS
        //console.log("card begin draw rectangle");
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
        //get second biggest
        let distances = [range0_to_1, range0_to_2, range0_to_3];
        distances.sort(function (a, b) {
            return a - b
        });
        let cardlongerSide = distances[1];
        //console.log("cardlong: ", cardlongerSide);
        //console.log("cardshort: ", cardshorterSide);


        ///get size of TRUNK
        //1.) RGBA to ONE CHANNEL
        cv.cvtColor(trunk, trunk, cv.COLOR_RGBA2GRAY, 0);
        /*
        console.log('trunk width: ' + trunk.cols + '\n' +
            'trunk height: ' + trunk.rows + '\n' +
            'trunk size: ' + trunk.size().width + '*' + trunk.size().height + '\n' +
            'trunk depth: ' + trunk.depth() + '\n' +
            'trunk channels ' + trunk.channels() + '\n' +
            'trunk type: ' + trunk.type() + '\n');
        */
        //2.) CLOSE OPERATION TO KILL NOISE AND CONNECT MASKS
        let trunkCloseFilter = cv.Mat.ones(this.closingFactor, this.closingFactor, cv.CV_8U);
        let closedTrunk = new cv.Mat();
        cv.morphologyEx(trunk, closedTrunk, cv.MORPH_CLOSE, trunkCloseFilter);
        let trunkOpenFilter = cv.Mat.ones(this.openingFactor, this.openingFactor, cv.CV_8U);
        let openedTrunk = new cv.Mat();
        cv.morphologyEx(closedTrunk, openedTrunk, cv.MORPH_OPEN, trunkOpenFilter);

        //3.) FIND COUNTOURS
        let trunkCountoursDrawn = cv.Mat.zeros(card.cols, card.rows, cv.CV_8UC3);
        contours = new cv.MatVector();
        hierarchy = new cv.Mat();
        cv.findContours(openedTrunk, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
        //console.log("trunk contours found");
        // DRAW COUNTOURS
        for (let i = 0; i < contours.size(); ++i) {
            let color = new cv.Scalar(Math.round(Math.random() * 255), Math.round(Math.random() * 255),
                Math.round(Math.random() * 255));
            cv.drawContours(trunkCountoursDrawn, contours, i, color, 1, cv.LINE_8, hierarchy, 100);
        }
        //console.log("trunk contours drawn");
        //4.) FIND MIN AREA RECT OF CONTOURS
        //TODO: MAYBE USE ONLY ONE CONTOUR:merge 2 biggest Area contours(if more than 1) to one remaining contour to feed to convexHull and get average diameter from it!
        //TODO: If 2 masks, check if both have realistic size
        let trunkRects = cv.Mat.zeros(trunk.rows, trunk.cols, cv.CV_8UC3);
        let trunkRectangleColor = new cv.Scalar(255, 0, 0);

        //let trunklongerSide;
        let trunkshorterSide;
        let diameter1;
        let diameter2;
        //console.log("number of contours: ", contours.size());


        if (contours.size() > 1) {
            //more than one contour
            //get 2 biggest area contours
            let biggestContoursIndexes = [];
            let biggestAreas = [];
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
            //console.log("card begin draw rectangle");
            for (let i = 0; i < 4; i++) {
                cv.line(trunkRects, trunkVertices1[i], trunkVertices1[(i + 1) % 4], trunkRectangleColor, 2, cv.LINE_AA, 0);
                cv.line(trunkRects, trunkVertices2[i], trunkVertices2[(i + 1) % 4], trunkRectangleColor, 2, cv.LINE_AA, 0);

            }
            //get centers of rectangles
            //1
            let xCoordinates1 = 0;
            let yCoordinates1 = 0;
            let xCoordinates2 = 0;
            let yCoordinates2 = 0;
            for (let i = 0; i < 4; i++) {
                xCoordinates1 += trunkVertices1[i]["x"];
                yCoordinates1 += trunkVertices1[i]["y"];
                xCoordinates2 += trunkVertices2[i]["x"];
                yCoordinates2 += trunkVertices2[i]["y"];
            }
            let center1 = [xCoordinates1 / 4, yCoordinates1 / 4];
            let center2 = [xCoordinates2 / 4, yCoordinates2 / 4];
            //console.log("center1", center1);
            //console.log("center2", center2);

            //check if parts of trunk are stacked above or next to each other
            if (Math.abs(center1[0] - center2[0]) < Math.abs(center1[1] - center2[1])) {
                console.log("above each other");
                //above each other
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
                //console.log("highestPoint1.1", highestPoint1);
                //console.log("highestPoint1.2", highestPoint2);


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
                //console.log("highestPoint2.1", highestPoint1);
                //console.log("highestPoint2.2", highestPoint2);

                //get side above trunk
                diameter2 = getRange(highestPoint1[0], highestPoint1[1], highestPoint2[0], highestPoint2[1]);
                //console.log("diamter1:", diameter1);
                //console.log("diamter2:", diameter2);

            } else {
                //next to each other
                //console.log("next to each other");

                //RECTANGLE 1
                //get 2 most right points
                let mostRightPoint1 = [];
                let mostRightPoint2 = [];


                for (let i = 0; i < 4; i++) {
                    if (mostRightPoint1.length < 2) {
                        mostRightPoint1[0] = trunkVertices1[i]["x"];
                        mostRightPoint1[1] = trunkVertices1[i]["y"];
                        mostRightPoint2[0] = trunkVertices1[i]["x"];
                        mostRightPoint2[1] = trunkVertices1[i]["y"];

                    } else {

                        if (trunkVertices1[i]["x"] > mostRightPoint1[0]) {
                            mostRightPoint1[0] = trunkVertices1[i]["x"];
                            mostRightPoint1[1] = trunkVertices1[i]["y"];
                        } else if (trunkVertices1[i]["x"] > mostRightPoint2[0]) {
                            mostRightPoint2[0] = trunkVertices1[i]["x"];
                            mostRightPoint2[1] = trunkVertices1[i]["y"];
                        }
                    }
                }

                //console.log("mostRightPoint1.1", mostRightPoint1);
                //console.log("mostRightPoint1.2", mostRightPoint2);

                //get side above trunk
                diameter1 = getRange(mostRightPoint1[0], mostRightPoint1[1], mostRightPoint2[0], mostRightPoint2[1]);

                //RECTANGLE 2
                //get 2 highest points
                mostRightPoint1 = [trunkVertices2[0]["x"], trunkVertices2[0]["y"]];
                mostRightPoint2 = [trunkVertices2[1]["x"], trunkVertices2[1]["y"]];

                for (let i = 0; i < 2; i++) {
                    if (trunkVertices2[i]["y"] > mostRightPoint1[1]) {
                        mostRightPoint1[0] = trunkVertices2[i]["x"];
                        mostRightPoint1[1] = trunkVertices2[i]["y"];
                    } else if (trunkVertices2[i]["y"] > mostRightPoint2[1]) {
                        mostRightPoint2[0] = trunkVertices2[i]["x"];
                        mostRightPoint2[1] = trunkVertices2[i]["y"];
                    }
                }

                console.log("mostRightPoint2.1", mostRightPoint1);
                console.log("mostRightPoint2.2", mostRightPoint2);
                //get side above trunk
                diameter2 = getRange(mostRightPoint1[0], mostRightPoint1[1], mostRightPoint2[0], mostRightPoint2[1]);
                console.log("diamter1:", diameter1);
                console.log("diamter2:", diameter2);
            }
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

            //get second biggest
            distances = [range0_to_1, range0_to_2, range0_to_3];
            distances.sort(function (a, b) {
                return a - b
            });
            //trunklongerSide = distances[1];
            trunkshorterSide = distances [0];
        }

        //COMPARE SIZES TO ESTIMATE DIAMETER
        let cardLength = 856; //mm
        let pixelSize = cardLength / cardlongerSide;

        //return trunkDiameter
        //console.log("EstimatedDiameter: ", (trunkshorterSide * pixelSize) / 100);
        return(trunkshorterSide * pixelSize) / 100;  //cm
    }
}

export default MasksToDiameterService