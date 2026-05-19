// ======================================================================
// MONEYVISION AI PRO
// ANTI FLICKER + STABLE DETECTION SYSTEM
// ======================================================================


// ======================================================================
// CONFIG
// ======================================================================

const CONFIG = {

    modelPath: "./best.onnx",

    labels: [
        "Uang_Kertas",
        "Uang_Koin"
    ],

    threshold: 0.60,

    iouThreshold: 0.45,

    inputSize: 640,

    maxFPS: 30,

    boxColor: "#00FF99",

    // =========================
    // ANTI FLICKER SETTINGS
    // =========================

    // Berapa frame harus muncul
    // sebelum dianggap valid
    stableFrames: 5,

    // Berapa frame hilang
    // sebelum dihapus
    maxMissedFrames: 10,

    // Minimal confidence stabil
    stableConfidence: 0.65

};


// ======================================================================
// ELEMENTS
// ======================================================================

const video =
    document.getElementById("webcam");

const overlay =
    document.getElementById("overlay");

const ctx =
    overlay.getContext("2d");

const processor =
    document.getElementById("processor");

const processorCtx =
    processor.getContext("2d", {
        willReadFrequently: true
    });

const statusText =
    document.getElementById("status");

const initBtn =
    document.getElementById("btn-init");


// ======================================================================
// VARIABLES
// ======================================================================

let session = null;

let running = false;

let trackedObjects = [];

let objectIdCounter = 0;


// ======================================================================
// INITIALIZE
// ======================================================================

initBtn.addEventListener("click", async () => {

    try {

        initBtn.disabled = true;

        statusText.innerText =
            "Loading AI...";

        ort.env.wasm.wasmPaths =
            "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

        session =
            await ort.InferenceSession.create(
                CONFIG.modelPath,
                {
                    executionProviders: [
                        "webgl",
                        "wasm"
                    ]
                }
            );

        await startCamera();

    } catch (error) {

        console.error(error);

        statusText.innerText =
            "Model gagal dimuat";

    }

});


// ======================================================================
// CAMERA
// ======================================================================

async function startCamera() {

    const stream =
        await navigator.mediaDevices.getUserMedia({

            video: {
                facingMode: "environment",
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },

            audio: false

        });

    video.srcObject = stream;

    video.onloadedmetadata = () => {

        video.play();

        overlay.width =
            video.videoWidth;

        overlay.height =
            video.videoHeight;

        running = true;

        statusText.innerText =
            "AI AKTIF";

        initBtn.style.display = "none";

        requestAnimationFrame(loop);

    };

}


// ======================================================================
// MAIN LOOP
// ======================================================================

async function loop() {

    if (!running) return;

    const tensor =
        preprocess();

    const results =
        await session.run({
            [session.inputNames[0]]: tensor
        });

    const output =
        results[session.outputNames[0]].data;

    const detections =
        processOutput(output);

    updateTracking(detections);

    drawTrackedObjects();

    requestAnimationFrame(loop);

}


// ======================================================================
// PREPROCESS
// ======================================================================

function preprocess() {

    const size =
        CONFIG.inputSize;

    processorCtx.drawImage(
        video,
        0,
        0,
        size,
        size
    );

    const imageData =
        processorCtx.getImageData(
            0,
            0,
            size,
            size
        ).data;

    const input =
        new Float32Array(
            3 * size * size
        );

    for (let i = 0; i < size * size; i++) {

        input[i] =
            imageData[i * 4] / 255;

        input[i + size * size] =
            imageData[i * 4 + 1] / 255;

        input[i + 2 * size * size] =
            imageData[i * 4 + 2] / 255;

    }

    return new ort.Tensor(
        "float32",
        input,
        [1, 3, size, size]
    );

}


// ======================================================================
// PROCESS OUTPUT
// ======================================================================

function processOutput(output) {

    const boxes = [];

    const numClasses =
        CONFIG.labels.length;

    const elements = 8400;

    for (let i = 0; i < elements; i++) {

        let bestScore = 0;
        let classId = 0;

        for (let c = 0; c < numClasses; c++) {

            const score =
                output[i + (4 + c) * elements];

            if (score > bestScore) {

                bestScore = score;
                classId = c;

            }

        }

        if (
            bestScore >
            CONFIG.threshold
        ) {

            let x = output[i];
            let y = output[i + elements];
            let w = output[i + 2 * elements];
            let h = output[i + 3 * elements];

            if (w <= 1.5) {

                x *= CONFIG.inputSize;
                y *= CONFIG.inputSize;
                w *= CONFIG.inputSize;
                h *= CONFIG.inputSize;

            }

            boxes.push({

                x: x - w / 2,
                y: y - h / 2,
                w,
                h,

                score: bestScore,

                classId

            });

        }

    }

    return nonMaxSuppression(
        boxes,
        CONFIG.iouThreshold
    );

}


// ======================================================================
// TRACKING SYSTEM
// ======================================================================

function updateTracking(detections) {

    // tandai semua object hilang
    trackedObjects.forEach(obj => {
        obj.missed++;
    });

    detections.forEach(det => {

        let matched = false;

        for (const obj of trackedObjects) {

            const iou =
                calculateIoU(obj, det);

            // kalau object sama
            if (iou > 0.5) {

                obj.x = det.x;
                obj.y = det.y;
                obj.w = det.w;
                obj.h = det.h;

                obj.score = det.score;
                obj.classId = det.classId;

                obj.missed = 0;

                obj.stable++;

                matched = true;

                break;

            }

        }

        // object baru
        if (!matched) {

            trackedObjects.push({

                id: objectIdCounter++,

                x: det.x,
                y: det.y,
                w: det.w,
                h: det.h,

                score: det.score,

                classId: det.classId,

                stable: 1,

                missed: 0

            });

        }

    });

    // hapus object hilang
    trackedObjects =
        trackedObjects.filter(obj =>
            obj.missed <
            CONFIG.maxMissedFrames
        );

}


// ======================================================================
// DRAW OBJECTS
// ======================================================================

function drawTrackedObjects() {

    ctx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    const scaleX =
        overlay.width /
        CONFIG.inputSize;

    const scaleY =
        overlay.height /
        CONFIG.inputSize;

    trackedObjects.forEach(obj => {

        // anti flicker
        if (
            obj.stable <
            CONFIG.stableFrames
        ) return;

        // confidence stabilizer
        if (
            obj.score <
            CONFIG.stableConfidence
        ) return;

        const x = obj.x * scaleX;
        const y = obj.y * scaleY;
        const w = obj.w * scaleX;
        const h = obj.h * scaleY;

        // glow
        ctx.shadowColor =
            CONFIG.boxColor;

        ctx.shadowBlur = 20;

        // box
        ctx.strokeStyle =
            CONFIG.boxColor;

        ctx.lineWidth = 4;

        ctx.strokeRect(x, y, w, h);

        // label
        const label =
            `${CONFIG.labels[obj.classId]} ${(obj.score * 100).toFixed(1)}%`;

        ctx.font =
            "bold 18px Arial";

        const textWidth =
            ctx.measureText(label).width;

        ctx.fillStyle =
            CONFIG.boxColor;

        ctx.fillRect(
            x,
            y - 35,
            textWidth + 20,
            30
        );

        ctx.fillStyle = "#001b14";

        ctx.fillText(
            label,
            x + 10,
            y - 12
        );

        // futuristic corners
        drawCorners(x, y, w, h);

    });

}


// ======================================================================
// CORNERS
// ======================================================================

function drawCorners(x, y, w, h) {

    const size = 25;

    ctx.strokeStyle =
        CONFIG.boxColor;

    ctx.lineWidth = 5;

    // kiri atas
    ctx.beginPath();
    ctx.moveTo(x, y + size);
    ctx.lineTo(x, y);
    ctx.lineTo(x + size, y);
    ctx.stroke();

    // kanan atas
    ctx.beginPath();
    ctx.moveTo(x + w - size, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + size);
    ctx.stroke();

    // kiri bawah
    ctx.beginPath();
    ctx.moveTo(x, y + h - size);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x + size, y + h);
    ctx.stroke();

    // kanan bawah
    ctx.beginPath();
    ctx.moveTo(x + w - size, y + h);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x + w, y + h - size);
    ctx.stroke();

}


// ======================================================================
// IOU
// ======================================================================

function calculateIoU(a, b) {

    const xA =
        Math.max(a.x, b.x);

    const yA =
        Math.max(a.y, b.y);

    const xB =
        Math.min(
            a.x + a.w,
            b.x + b.w
        );

    const yB =
        Math.min(
            a.y + a.h,
            b.y + b.h
        );

    const inter =
        Math.max(0, xB - xA) *
        Math.max(0, yB - yA);

    const union =
        (a.w * a.h) +
        (b.w * b.h) -
        inter;

    return inter / union;

}


// ======================================================================
// NMS
// ======================================================================

function nonMaxSuppression(
    boxes,
    threshold
) {

    boxes.sort((a, b) =>
        b.score - a.score
    );

    const result = [];

    while (boxes.length > 0) {

        const current =
            boxes.shift();

        result.push(current);

        boxes =
            boxes.filter(box =>
                calculateIoU(
                    current,
                    box
                ) < threshold
            );

    }

    return result;

}
