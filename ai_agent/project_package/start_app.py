import os
import subprocess

# Set the environment variables if needed
os.environ['API_HOST'] = '0.0.0.0'
os.environ['API_PORT'] = '5000'

# Command to run the FastAPI application
subprocess.run(['uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '5000'])
