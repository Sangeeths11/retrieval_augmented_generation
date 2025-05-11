import os
import sys
import subprocess

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["STREAMLIT_WATCHDOG_IGNORE_TORCH"] = "True"
os.environ["PYTHONPATH"] = os.getcwd()

print("Starting Streamlit with PyTorch compatibility fixes...")


streamlit_cmd = ["streamlit", "run", "streamlit_app.py"] + sys.argv[1:]
subprocess.run(streamlit_cmd) 