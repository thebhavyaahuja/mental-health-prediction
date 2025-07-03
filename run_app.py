import os
import subprocess
import sys

# Set environment variables before any imports
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Run the Streamlit app
if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"] + sys.argv[1:])
