import "./index.css";
import React from "react";
import videojs from "video.js";
import VideoJS from "./Video";

function App() {
  const playerRef = React.useRef(null);
  const videoJsOptions = {
    autoplay: true,
    controls: false,
    responsive: true,
    fluid: true,
    sources: [
      {
        src: "file:///Users/connornelson/Github/magic-touch/content/40secondclipballtracking.mp4",
        type: "video/mp4",
      },
    ],
  };

  const handlePlayerReady = (player) => {
    playerRef.current = player;

    player.on("play", () => {
      console.log("play");
    });

    player.on("timeupdate", () => {
      let time = player.currentTime();
      console.log(player.currentTime());
      if (time > 10) {
        player.currentTime(0);
      }
    });

    // You can handle player events here, for example:
    player.on("waiting", () => {
      videojs.log("player is waiting");
    });

    player.on("dispose", () => {
      videojs.log("player will dispose");
    });

    player.play();
  };

  return (
    <>
      <div>Rest of app here</div>
      <div className="w-1/2">
        <VideoJS options={videoJsOptions} onReady={handlePlayerReady} />
      </div>
      <button onClick={() => playerRef.current.play()}>Play</button>
      <button onClick={() => playerRef.current.pause()}>Pause</button>
      <button onClick={() => playerRef.current.currentTime(100)}>100</button>
      <div>Rest of app here</div>
    </>
  );
}

export default App;
