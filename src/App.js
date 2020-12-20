import React, { useRef, useEffect,  } from 'react';
import { init, predict } from './processStream';
import bgimg from './bg.jpg';

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
      <div className="col">
        <video className="video" muted ref={video} autoPlay></video>
      </div>
      <div className="col">
        <canvas ref={output}/>
      </div>
      <img className="hidden" alt="Img" ref={bg} src={bgimg}></img>
      <button onClick={()=> {predict(true,output); video.current.className = 'hidden'; } }>Add Background</button>
    </div>
  );
}

export default App;
