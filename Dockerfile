# 1. Use an official Python runtime as a parent image
FROM python:3.9

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install the needed packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code
COPY app.py .

# 6. Expose the port that Hugging Face Spaces expects (7860)
EXPOSE 7860

# 7. Run the application
# We force Streamlit to use port 7860 to match Hugging Face's requirements
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
