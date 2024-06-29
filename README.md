**UFSCChatBot**


**Make sure you have the following dependencies installed**:


pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip3 install transformers

pip3 install flask, datasets, spaces, sentence_transformers, threading



**Steps to Run**
**1. Run Frontend**
First, establish an SSH tunnel to the remote server for the frontend:

ssh -L 3000:localhost:3000 username@serverIP

cd frontend/src/pages/ChatUfscPage

npm start

**2. Run Backend**
Next, set up the backend on the remote server. Establish an SSH tunnel and activate the virtual environment:


ssh -L 8080:localhost:8080 username@serverIP

source <.venvname>/bin/activate

cd backend

gunicorn -c gunicorn_config.py llama3:app


**3. Open Browser**

Once both frontend and backend are running, open your browser and navigate to http://localhost:3000 to use the application.
