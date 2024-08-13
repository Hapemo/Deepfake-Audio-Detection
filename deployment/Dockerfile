# Use the PyTorch base image
FROM pytorch/pytorch:latest

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    # Install any additional dependencies if needed
    # For example, if you need NVIDIA drivers or CUDA libraries, include them here
    # For now, this is just a placeholder
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Set the working directory inside the container
WORKDIR /app

# Copy the application files and dependencies
COPY . /app

# Install Python dependencies
RUN pip install -r /app/requirements.txt

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV JUPYTER_PORT=8888
ENV TENSORBOARD_PORT=6006
ENV GRADIO_SERVER_PORT=7860

# Expose the necessary ports
EXPOSE 8888
EXPOSE 6006
EXPOSE 7860

# Command to run the application
CMD ["jupyter lab"]