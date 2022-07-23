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
