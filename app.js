// ======================================================================
// 1. PENGATURAN PROYEK (KALIAN HANYA PERLU MENGUBAH BAGIAN INI)
// ======================================================================
const CONFIG = {
    // Nama file model AI yang sudah kalian download dari Colab
    modelPath: './best.onnx', 
    
    // GANTI INI dengan nama kelas kalian. 
    // PERHATIAN: Urutannya HARUS SAMA PERSIS dengan urutan di Roboflow!
    labels: ["Kelas_Satu", "Kelas_Dua"], 
    
    // Batas keyakinan AI (0.45 = 45%). 
    // Jika AI terlalu sering salah tebak, naikkan angkanya (misal 0.60).
    threshold: 0.45,
    
    // Batas untuk menghapus kotak deteksi yang menumpuk (Biarkan saja 0.4)
    iouThreshold: 0.4
};

// ======================================================================
// 2. MESIN INTI AI (JANGAN MENGUBAH KODE DI BAWAH INI!)
// ======================================================================

// Menangkap elemen-elemen dari halaman HTML agar bisa dikendalikan oleh Javascript[cite: 1]
const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const ctxOverlay = overlay.getContext('2d');
const processor = document.getElementById('processor');
const ctxProcessor = processor.getContext('2d', { willReadFrequently: true });
const status = document.getElementById('status');
const initBtn = document.getElementById('btn-init');

let session;
const TARGET_SIZE = 640; // Ukuran gambar standar yang diminta oleh YOLO11n

// Langkah 1: Memuat Model AI saat tombol ditekan[cite: 1]
initBtn.addEventListener('click', async () => {
    initBtn.disabled = true;
    initBtn.innerText = "MEMUAT MODEL AI...";
    try {
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        session = await ort.InferenceSession.create(CONFIG.modelPath, { 
            executionProviders: ['webgl', 'wasm'] // Meminta browser menggunakan GPU/VGA jika tersedia
        });
        startCamera();
    } catch (e) {
        status.innerText = "GAGAL: FILE MODEL TIDAK DITEMUKAN";
        console.error(e);
    }
});

// Langkah 2: Menyalakan Kamera Web[cite: 1]
async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
        video.play();
        status.innerText = "SISTEM AKTIF: MENUNGGU OBJEK";
        initBtn.style.display = "none";
        requestAnimationFrame(processFrame);
    };
}

// Langkah 3: Proses Deteksi Berulang (Looping)[cite: 1]
async function processFrame() {
    if (!session) return;

    // A. Mengambil satu gambar dari video kamera dan menyesuaikan ukurannya ke 640x640[cite: 1]
    ctxProcessor.drawImage(video, 0, 0, TARGET_SIZE, TARGET_SIZE);
    const imageData = ctxProcessor.getImageData(0, 0, TARGET_SIZE, TARGET_SIZE).data;
    const float32Data = new Float32Array(3 * TARGET_SIZE * TARGET_SIZE);

    // B. Mengubah format warna piksel agar bisa dibaca oleh matriks AI[cite: 1]
    for (let i = 0; i < TARGET_SIZE * TARGET_SIZE; i++) {
        float32Data[i] = imageData[i * 4] / 255.0; // Warna Merah (R)
        float32Data[i + TARGET_SIZE * TARGET_SIZE] = imageData[i * 4 + 1] / 255.0; // Warna Hijau (G)
        float32Data[i + 2 * TARGET_SIZE * TARGET_SIZE] = imageData[i * 4 + 2] / 255.0; // Warna Biru (B)
    }

    // C. Mengirim gambar ke otak AI (Model ONNX)[cite: 1]
    const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, TARGET_SIZE, TARGET_SIZE]);
    const results = await session.run({ [session.inputNames[0]]: inputTensor });
    const output = results[session.outputNames[0]].data; 
    
    // D. Membaca hasil tebakan AI[cite: 1]
    const numClasses = CONFIG.labels.length;
    const elements = 8400; 
    let rawBoxes = [];

    for (let i = 0; i < elements; i++) {
        let maxScore = 0;
        let classId = -1;
        
        // Mencari nilai persentase tertinggi di antara semua tebakan kelas
        for (let c = 0; c < numClasses; c++) {
            const score = output[i + (4 + c) * elements];
            if (score > maxScore) {
                maxScore = score;
                classId = c;
            }
        }

        // Jika tebakan AI melebihi batas threshold yang kalian atur
        if (maxScore > CONFIG.threshold) {
            let x = output[i];
            let y = output[i + elements];
            let w = output[i + 2 * elements];
            let h = output[i + 3 * elements];
            
            // Menyesuaikan ukuran kotak hasil deteksi[cite: 1]
            if (w <= 1.5) { x *= TARGET_SIZE; y *= TARGET_SIZE; w *= TARGET_SIZE; h *= TARGET_SIZE; }

            rawBoxes.push({
                x: x - w / 2, y: y - h / 2, w: w, h: h,
                score: maxScore,
                classId: classId
            });
        }
    }

    // E. Membersihkan kotak-kotak yang menumpuk pada objek yang sama[cite: 1]
    const finalBoxes = nonMaxSuppression(rawBoxes, CONFIG.iouThreshold);
    drawBoxes(finalBoxes);
    requestAnimationFrame(processFrame);
}

// ======================================================================
// FUNGSI MATEMATIKA TAMBAHAN (Intersection over Union & NMS)
// ======================================================================
function calculateIoU(box1, box2) {
    const xA = Math.max(box1.x, box2.x);
    const yA = Math.max(box1.y, box2.y);
    const xB = Math.min(box1.x + box1.w, box2.x + box2.w);
    const yB = Math.min(box1.y + box1.h, box2.y + box2.h);
    const intersectionArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    return intersectionArea / ((box1.w * box1.h) + (box2.w * box2.h) - intersectionArea); //[cite: 1]
}

function nonMaxSuppression(boxes, iouThreshold) {
    boxes.sort((a, b) => b.score - a.score);
    const result = [];
    while (boxes.length > 0) {
        const current = boxes.shift();
        result.push(current);
        boxes = boxes.filter(box => calculateIoU(current, box) < iouThreshold); //[cite: 1]
    }
    return result; //[cite: 1]
}

// Fungsi untuk menggambar kotak hijau beserta teks label di atas video
function drawBoxes(boxes) {
    ctxOverlay.clearRect(0, 0, overlay.width, overlay.height);
    boxes.forEach(box => {
        const scaleX = overlay.width / TARGET_SIZE;
        const scaleY = overlay.height / TARGET_SIZE;
        
        ctxOverlay.strokeStyle = "#34C759";
        ctxOverlay.lineWidth = 3;
        ctxOverlay.strokeRect(box.x * scaleX, box.y * scaleY, box.w * scaleX, box.h * scaleY);
        
        ctxOverlay.fillStyle = "#34C759";
        ctxOverlay.font = "bold 16px Arial";
        ctxOverlay.fillText(`${CONFIG.labels[box.classId]} ${(box.score * 100).toFixed(0)}%`, box.x * scaleX, box.y * scaleY - 5);
    });
}
