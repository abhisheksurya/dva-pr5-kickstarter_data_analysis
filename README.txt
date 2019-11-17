DESCRIPTION
This is a web application that uses Express / Node.js as its backend, and Tableau and D3 to generate visualizations in the front-end.
Users may upload a CSV file containing data on their own Kickstarter project (a template file is provided), and receive dynamically generated visualizations as output.
This is all accomplished asynchronously via spawning a Python process on the server side, and Ajax from the clientside.
Included code:
CODE/
    datacleanup - Jupyter notebooks used to clean up data, reducing the original webrobots dataset into the ksDataset.csv.
    server/ - Code to run the model.
        model/ - Python code that does data analysis.
            Makefile - Makefile used to simplify server preparation (train the model before we start the server).
            ksDatasetEncode.py - Python script to encode and filter data to be processed by trainModel.py and kickstarter.py.
            trainModel.py - Python script which uses scikit-learn to train models, and exports them as files.
            kickstarter.py - Python script which uses trained models and encoded data to analyze user input. Outputs files that get used in client response.
            ksDataset.csv - Cleaned dataset used to train models
        static/ - Public content
            index.html - Home page
            js/ - Client-side javascript
                d3_code_slopegraph.js - D3 code for slopegraph - borrowed from eesur.com
                render_d3.js - D3 code for slopegraph - borrowed from eesur.com
                jq.js - code to handle file upload via jQuery ajax method
            img/ - images used by index.html
                ...
            csv/
                userInputTemplate.csv - Template file users can download to provide input
        package-lock.json - node config file
        package.json - node config file
        server.js - Main server code, uses Express to serve files, and promises, temporary files, child processes, etc. to handle file upload and serves as an intermediary between python
        start.js - Script used to provide ES6 compatibility to node via Babel

INSTALLATION
0) Ensure gmake, node v10+, Python 3.7+, scikit-learn, pandas are installed. (for other missing Python files, look at .py files in 'CORE/server/model' to see dependencies).
1) Change to the 'CODE/server' directory ('cd CODE/server')
2) Run 'npm install' to install node packages, and train the model, generating files in CODE/server/model.
    - Training the models using the ksDataset.csv in 'CODE/server/model' will take ~5 minutes.

EXECUTION
1) Make sure you have privileges to listen on port 80 (may require 'sudo' on Mac), and that nothing else is listening on that port.
2) Run 'npm start' to start the web server, listening on port 80. Note that a lot of console output will appear as a part of normal use to give you a look at what's happening under the hood ;)
3) Open a web browser to http://localhost/ or http://127.0.0.1/ and interact with the application.
Some interactions:
- You can download a template CSV file, modify it to match your prospective project, and upload it to the server. This will process the data and produce dynamically generated visualizations.
- To view/interact with the first visualization, you must have a Tableau account. Note that this visualization is not dynamically generated, but provides exploratory information about the dataset we used.
- The second and third visualizations are automatically generated after you receive a server response from your file upload.
4) Ctrl-C to stop the server. In some environments, node may still be running, which requires you to seek and destroy (kill) the runaway process.



# dva-fall-2018-teamproject

1. Credits to Dr. Polo and DVA team
2. Link to project doc - https://docs.google.com/document/d/e/2PACX-1vRj6A3u1okyJEClnAwUU_9zlPNCJsKKlrzLp0eOnadtwnWRy47qSehKcFFn1rhYzpu5RNQQe3AkI-1J/pub
3. About contributors
