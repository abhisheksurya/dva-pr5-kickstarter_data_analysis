// Tested using Node.js v10.13.0, Python v3.7.1
import express from 'express'; // The server
import { exec } from 'child-process-promise'; // Used to spawn Python processes in the background
import fileUpload from 'express-fileupload'; // Middleware to assist in responding to user file uploads
import fs from 'fs'; // Used to perform filesystem operations
import tmp from 'tmp-promise'; // Used to safely create input/output files from Python processes
import Promise from 'promise'; // Avoid callback hell or synchronous requests
import { promisify } from 'util'; // Used to wrap functions to return promises
import path from 'path';

const app = express();
const port = 80;
/* 1) Serve initial page via '/' route
 * 1a) Use Handlebars to set this up
 * 2) Serve static content under static dir (referenced by initial page)
 * 3) Handle AJAX requests from front-end
 * 4) Generate temp files from user input to pass in to Python scripts
 */

// Add middleware to help process file uploads
app.use(fileUpload());

/***** File upload route *****/
app.post('/upload', uploadHandler);

/***** Static file route - includes index.html *****/
app.use(express.static('static'));

/***** Listen for clients! *****/
app.listen(port, function() {
    console.log('Listening on port ' + port);
});


// Calls python scripts with proper arguments, returning a promise
async function runModel(inputFilePaths, outputFilePaths) {
    const command = "python";
    const modelDir = path.join('.', 'model');
    const scriptFile = "kickstarter.py";
    const args = [path.join(modelDir, scriptFile)];
    for(let i = 3; i <= 5; ++i) {
        // These are dynamically generated
        args.push(outputFilePaths["encodeOut" + i + ".out"]);
    }
    for(let i = 1; i <= 10; ++i) {
        // These are statically generated in advance
        args.push(path.join(modelDir,"modelOut" + i + ".out"));
    }
    args.push(outputFilePaths["predictionOut.csv"]);
    args.push(outputFilePaths["visualization2Out.csv"]);
    args.push(outputFilePaths["visualization3Out.png"]);
    args.push(outputFilePaths["error.txt"]);
    // Run the encode first
    await runEncode(inputFilePaths, outputFilePaths);

    console.log("Calling:");
    let command_line = command;
    for(let i in args) {
        command_line += " " + args[i];
    }
    console.log(command_line);
    await exec(command_line).then(result => {
        console.log(result);
    }).catch(err => {
        console.error(err);
    });
    /* Now that all the python has run, we can process output */
    return await processOutput(outputFilePaths);
}

async function runEncode(inputFilePaths, outputFilePaths) {
    const command = "python";
    const modelDir = path.join('.', 'model');
    const scriptFile = "ksDatasetEncode.py";
    const dataset = "ksDataset.csv";
    const args = [path.join(modelDir, scriptFile),
                  path.join(modelDir, dataset),
                  inputFilePaths['inputFile.csv']];
    for(let i = 1; i <= 5; ++i) {
        args.push(outputFilePaths["encodeOut" + i + ".out"]);
    }
    console.log("Calling:");
    let command_line = command;
    for(let i in args) {
        command_line += " " + args[i];
    }
    console.log(command_line);
    await exec(command_line).then(result => {
        console.log(result);
    }).catch(err => {
        console.log(err);
    });
}

/* Function to open output files, and read into JSON */
async function processOutput(outputFilePaths) {
    const readFile = promisify(fs.readFile);
    const vis2Filename = outputFilePaths["visualization2Out.csv"];
    const vis3Filename = outputFilePaths["visualization3Out.png"];
    let result2 = await readFile(vis2Filename);
    console.log(result2);
    let result3 = await readFile(vis3Filename);
    console.log(result3);
    const pngPrefix = "data:image/png;base64,";
    return {vis3: pngPrefix + result3.toString('base64'), vis2: result2.toString('utf-8')};
}

// Returns a pair of keys
async function generateTempFiles(inputFilesContents, outputFileKeys) {
    let filePromises = [];
    let inputFileKeys = []; // Need this to guarantee an ordering, since Promise.all() returns an array of promises
    console.log("Creating input file promises");
    for(let fileKey in inputFilesContents) {
        console.log("Creating promise for " + fileKey);
        inputFileKeys.push(fileKey);
        filePromises.push(tmp.file({prefix: fileKey, keep: true}).then(o => { // Create tmp file promise
            return new Promise((resolve, reject) => { // Once temp file is created, asynchronously write to the file
                console.log("Writing to " + o.path);
                let wStream = fs.createWriteStream(null, { fd: o.fd }); // Create a writestream to write to the opened file descriptor
                wStream.write(inputFilesContents[fileKey]); // Write the file contents
                wStream.on('error', err => {
                    console.log("Error writing to " + o.path);
                    reject(err)
                });
                wStream.on('finish', () => {
                    console.log("Finished writing to " + o.path);
                    resolve(o); // Finally, resolve to the path, so we can pass it as input
                });
                wStream.end(); // This also closes the file descriptor
            });
        }));
    }
    console.log("Created input file promises");
    // Set aside empty files for output - we need to do this instead of filenames since we want to make sure 
    for(let fileKey of outputFileKeys) {
        console.log("Creating promise for " + fileKey);
        filePromises.push(tmp.file({prefix: fileKey, keep: true}).then(o => {
            // Close files so they can be written to later.
            return new Promise((resolve, reject) => {
                console.log("Closing " + o.path);
                fs.close(o.fd, err => {
                    if(err) {
                        reject(err);
                    } else {
                        resolve(o);
                    }
                });
                resolve(o);
            });
        }));
    }
    console.log("Created output file promises");
    // Finally, gather all of the paths
    console.log("Awaiting all promises");
    let filePaths = await Promise.all(filePromises).catch(err => { console.log(err.message); reject(err); });
    console.log("Promises done!");
    console.log(filePaths);
    let cleanupFuncs = [];

    // inputFilePaths is an object with keys matching inputFilesContents above, and corresponding paths of files generated
    let inputFilePaths = {};
    for(let i = 0; i < inputFileKeys.length; ++i) {
        // The first promises in filePromises resolve to input file paths
        inputFilePaths[inputFileKeys[i]] = filePaths[i].path;
        cleanupFuncs.push(filePaths[i].cleanup);
    }
    // outputFilePaths is an object with keys matching outputFileKeys above, and corresponding paths of files generated
    let outputFilePaths = {};
    for(let i = 0; i < outputFileKeys.length; ++i) {
        // The later promises in filePromises resolve to output file paths
        outputFilePaths[outputFileKeys[i]] = filePaths[i + inputFileKeys.length].path;
        cleanupFuncs.push(filePaths[i].cleanup);
    }
    return {inputFilePaths: inputFilePaths,
            outputFilePaths: outputFilePaths,
            cleanupFuncs: cleanupFuncs};
}

// This route is for file uploads made by the client only, and responds with a Content-Type of 'multipart/mixed' with all data desired.
// It's strongly linked with the client-side code; not generic.
async function uploadHandler(req, res) {
    console.log("Received request!\n");
    // Validate we received a file
    if(Object.keys(req.files).length == 0) {
        console.log('No file uploaded.');
        return res.status(400).send('No file uploaded.');
    }
    if(!('inputFile' in req.files)) {
        console.log('inputFile field not provided.');
        return res.status(400).send('inputFile field not provided.');
    }
    // Use the 'inputFile' field to upload file
    let file = req.files.inputFile;
    if(typeof file == 'undefined') {
        console.log('inputFile field undefined.');
        return res.status(400).send('inputFile field undefined.');
    }
    /*** Generate temporary files ***/
    // We've successfully received the file data
    // Parse HTTP request data into input file(s) contents
    let inputFilesContents = {'inputFile.csv': file.data}; // Returns keys which identify files and values which are contents to put in the files.
    console.log(inputFilesContents);
    const outputFileKeys = ['predictionOut.csv', 'visualization2Out.csv', 'visualization3Out.png', 'error.txt']; // List of keys used to reference output temp files we need
    for(let i = 1; i <= 5; ++i) {
        outputFileKeys.push('encodeOut' + i + '.out');
    }
    // Create temporary input files with contents, and output filenames generated.
    console.log("Calling generateTempFiles...");
    let {inputFilePaths, outputFilePaths, cleanupFuncs} = await generateTempFiles(inputFilesContents, outputFileKeys);
    console.log("generated temp files");
    let outputData = await runModel(inputFilePaths, outputFilePaths);
    /* Send response */
    console.log("Cleaning up temporary files");
    /* Cleanup files */
    for(let i in cleanupFuncs) {
        await cleanupFuncs[i]();
    }
    console.log("Sending response!");
    console.log(outputData);
    return res.send(outputData);
}
