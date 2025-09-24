# Using the PyTorch image as the base
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app/

# Set a non-interactive frontend to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary libraries, tools, and Python development headers
RUN apt update && \
    apt install -y git openslide-tools libopenslide-dev build-essential python3-dev tmux && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt ./

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install Python packages from requirements
RUN pip install --no-cache-dir -r requirements.txt

# For the Streamlit
EXPOSE 8051

# Final requirement to use openslide
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
# Reset the frontend variable to its default value
ENV DEBIAN_FRONTEND=

# Keep the container running by tailing /dev/null
#CMD ["tail", "-f", "/dev/null"]

