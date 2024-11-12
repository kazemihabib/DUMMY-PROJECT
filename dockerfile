# Use the official MiniZinc image as a base
FROM minizinc/minizinc

# Install Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv

# Decrease the image size
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment in /app/venv
RUN python3 -m venv /app/venv

# Set the working directory in the container
WORKDIR /app

# Copy local files to the container
COPY . /app

# Activate the virtual environment and install dependencies
RUN . /app/venv/bin/activate && pip install -r /app/requirements.txt

# Default command: Open a bash shell (can be changed as needed)
CMD ["bash"]
