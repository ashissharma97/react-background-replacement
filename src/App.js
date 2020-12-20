import React, { useRef, useEffect,  } from 'react';
import { init, predict } from './processStream';

function App() {
  const video = useRef();
  const output = useRef();
  const bg = useRef();

  useEffect(() => {
    navigator.getUserMedia({video: true, audio:true}, stream =>{
      let v =  video.current;
      v.srcObject = stream;
      v.play();
    },
    e => console.log(e)); 
  },[video]);

  useEffect(() => {
    init(video , bg);
    // eslint-disable-next-line
  },[])

  return (
    <div className="container">
      <h1>Image Segementation</h1>
      <video muted id="video" ref={video} autoPlay></video>
      <button onClick={()=> predict(true) }>Add Background</button>
      <canvas ref={output}/>
      <img ref={bg} src="../public/bg.jpg"></img>
    </div>
  );
}

export default App;
