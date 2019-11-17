// Tweak to use ES6. Inspired by https://timonweb.com/posts/how-to-enable-es6-imports-in-nodejs/
require('babel-register')({
    presets: [ 'env' ]
});
require("babel-polyfill");

// Import the app
module.exports = require('./server.js');
