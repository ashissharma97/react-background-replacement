import * as tf from '@tensorflow/tfjs';

let cam;
let bgim;
let model;

export async function init(ref, bg) {
    model = await loadModel();
    bgim = loadBackground(bg);
    cam = await tf.data.webcam(ref.current);
    const screenShot = await cam.capture();
    const pred = model.predict(tf.zeros([1, 128, 128, 3]).toFloat());

    const readable_output = pred.dataSync();
    console.log(readable_output);
    console.log(model.summary());

    pred.dispose();
    screenShot.dispose();
}

function refine(mask) {
    const refine_out = tf.tidy(() => {
     const newmask = mask.reshape([1, 128, 128, 1]);

     const kernel = tf.tensor4d([0.00092991, 0.00223073, 0.00416755, 0.00606375, 0.00687113, 0.00606375,
      0.00416755, 0.00223073, 0.00535124, 0.00999743, 0.01454618, 0.01648298,
      0.01454618, 0.00999743, 0.00416755, 0.00999743, 0.01867766, 0.02717584,
      0.03079426, 0.02717584, 0.01867766, 0.00606375, 0.01454618, 0.02717584,
      0.03954061, 0.04480539, 0.03954061, 0.02717584, 0.00687113, 0.01648298,
      0.03079426, 0.04480539, 0.05077116, 0.04480539, 0.03079426, 0.00606375,
      0.01454618, 0.02717584, 0.03954061, 0.04480539, 0.03954061, 0.02717584,
      0.00416755, 0.00999743, 0.01867766, 0.02717584, 0.03079426, 0.02717584,
      0.01867766
     ], [7, 7, 1, 1]);
    
     const blurred = tf.conv2d(newmask, kernel, [1, 1],'same');
     const fb = blurred.squeeze(0) 
     const norm_msk =   fb.sub(fb.min()).div(fb.max().sub(fb.min()))
    
     return smoothstep(norm_msk);
    
    });
    
    return refine_out;
    }

export async function getImage() {
    const img = await cam.capture();
    const processedImg =
     tf.tidy(() => img.div(tf.scalar(255.0)));
    img.dispose();
    return processedImg;
}

function smoothstep(x) {

    const smooth_out = tf.tidy(() => {
    
     // Define the left and right edges 
     const edge0 = tf.scalar(0.3);
     const edge1 = tf.scalar(0.5);
    
     // Scale, bias and saturate x to 0..1 range
     const z = tf.clipByValue(x.sub(edge0).div(edge1.sub(edge0)), 0.0, 1.0);
     
     //Evaluate polynomial  z * z * (3 - 2 * x)
     return tf.square(z).mul(tf.scalar(3).sub(z.mul(tf.scalar(2))));
    
    });
    return smooth_out ;
}

export function process(image, mask) {

    const blend_out = tf.tidy(() => {
    
     const img = image.resizeBilinear([300, 300]);
     const msk = refine(mask).resizeNearestNeighbor([300, 300]);;
     const img_crop = img.mul(msk);
     const bgd_crop = bgim.mul(tf.scalar(1.0).sub(msk));
     const result = tf.add(img_crop, bgd_crop);
    
     return result;
    });
    
    return blend_out;
    
}

export default function drawDefault(ref) {
    const ctx = ref.getContext('2d');
    const defImg = new Image();
    defImg.crossOrigin = "anonymous";
    defImg.src = "bg.jpg";

    defImg.onload = function () {
        ref.style.width = defImg.width;
        ref.style.height = defImg.height;
        ctx.drawImage(defImg, 0, 0,defImg.width,defImg.height,0,0,300,300);
    }
}

function loadBackground(bg) {   
    const bim = tf.browser.fromPixels(bg.current);
    const img = tf.image.resizeBilinear(bim, [300, 300]).div(tf.scalar(255.0));
    bim.dispose();
    return img;
   
   }

async function loadModel() {
const model = await tf.loadLayersModel("http://localhost:5000/model.json"); 
return model;
}

export async function predict(isPredicting) {
    while (isPredicting) {
        const img =  await getImage();
        const resize = img.resizeBilinear([128, 128]);
        const expdim = resize.expandDims(0);
       
        const out = await model.predict(expdim);
      
        const thresh = tf.scalar(0.5);
        const msk = out.greater(thresh);
        const cst = tf.cast(msk,'float32');
      
        const blend = process(img, cst);
        
        await tf.browser.toPixels();
       
        blend.dispose();
        resize.dispose();
        msk.dispose();
        expdim.dispose();
        cst.dispose();
        thresh.dispose();
        out.dispose();
        img.dispose();
      
        await tf.nextFrame();
    }
}