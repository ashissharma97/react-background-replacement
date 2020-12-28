import React, { useRef, useEffect,  } from 'react';
import { init, predict } from './processStream';
import bgimg from './bg.jpg';

function App() {
  const video = useRef();

  useEffect(() => {
    navigator.getUserMedia({video: true, audio:true}, stream =>{
      let v =  video.current;
      v.srcObject = stream;
      v.play();
    },
    e => console.log(e));
  },[video]);

  useEffect(() => {
    init();
    // eslint-disable-next-line
  },[])

  return (
    <div className="container">
      <h1>Image Segementation</h1>
      <video className="video" ref={video} muted id="video" autoPlay></video>
      <canvas className="hidden" id="canvas" />
      <img className="hidden" alt="Img" id="image" src={bgimg}></img>
      <button onClick={()=> {predict(true); video.current.className = 'hidden'; } }>Add Background</button>
      <video id="output" autoPlay />
    </div>
  );
}

export default App;
