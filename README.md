# AWS Custom Chatbot App
This Streamlit app allows users to upload files (CSV, PDF, DOCX) and interact with a chatbot that can answer questions about the files and utilize AWS services such as S3 and Bedrock LLMs for enhanced capabilities.

## Features
- Upload and query files such as CSV, PDF, and DOCX.
- The app uses AWS services, including Bedrock LLMs, for natural language processing and file analysis.
- Includes a memory feature for conversation continuity.

## Requirements
Before running the app, ensure you have installed all the required dependencies. These can be found in the requirements.txt file.

Installation
Clone the repository:

```
git clone https://github.com/your-username/your-repository.git
```
Navigate into the project directory:

```
cd your-repository
```
Install the required dependencies using pip:
```
pip install -r requirements.txt
```
Create an .env file in the root of the project. The .env file should contain your credentials for accessing AWS services:

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_aws_region
BUCKET_NAME=your_bucket_name
```
Once your .env file is set up, you can run the app.

Usage
To start the app, run the following command:

```
streamlit run main.py
```
Once the app is running, you can upload a file (CSV, PDF, DOCX) and interact with the chatbot to ask questions about the content.

The app also provides memory features for conversational continuity and leverages AWS Bedrock for enhanced language processing capabilities.

## Notes
- Make sure that your AWS credentials have sufficient permissions for services such as S3 and Bedrock LLMs.
- The .env file should not be shared as it contains sensitive credentials.

## License
This project is licensed under the MIT License.
