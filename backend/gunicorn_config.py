# gunicorn_config.py

bind = '0.0.0.0:8080'  # Replace with your desired host and port
workers = 1  # Number of worker processes, adjust as needed
timeout = 120  # Timeout in seconds
