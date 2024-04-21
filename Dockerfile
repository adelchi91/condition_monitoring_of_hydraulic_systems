# Use the official Python 3.9 image as the base image
FROM python:3.9

# Set the working directory inside the container to /code
WORKDIR /code

# Copy the requirements.txt file from the host to the container's working directory
COPY requirements.txt ./code/requirements.txt

# Install the Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the app directory from the host to the container's working directory
COPY ./app ./code/app

# Copy the data directory from the host to the container's working directory
COPY ./data ./code/data

# Specify the default command to run when the container starts
# In this case, it uses uvicorn to run the FastAPI application (fast_api.py) and listens
# on all available network interfaces (0.0.0.0).
CMD ["uvicorn", "app.fast_api:app", "--host", "0.0.0.0"]
