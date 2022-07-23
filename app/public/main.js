const { app, BrowserWindow } = require("electron");
const { protocol } = require("electron");

function fileHandler(req, callback) {
  let requestedPath = req.url;
  // Write some code to resolve path, calculate absolute path etc
  let check = true;

  if (!check) {
    callback({
      // -6 is FILE_NOT_FOUND
      // https://source.chromium.org/chromium/chromium/src/+/master:net/base/net_error_list.h
      error: -6,
    });
    return;
  }

  callback({
    path: requestedPath,
  });
}

function createWindow() {
  // Create the browser window.
  const win = new BrowserWindow({
    width: 1920,
    height: 1080,
    webPreferences: {
      nodeIntegration: true,
      enableRemoteModule: true,
      webSecurity: false,
    },
    title: "Magic Touch",
  });

  //load the index.html from a url
  win.loadURL("http://localhost:3000");

  // Open the DevTools.
  win.webContents.openDevTools();
}

// protocol.registerFileProtocol("f", fileHandler);

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(createWindow);

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  // On macOS it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.

  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
