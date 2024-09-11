# AWS Custom Chatbot App
This Streamlit app allows users to upload files (CSV, PDF, DOCX) and interact with a chatbot that can answer questions about the files and utilize AWS services such as S3 and Bedrock LLMs for enhanced capabilities.

## Features
- Upload and query files such as CSV, PDF, and DOCX.
- The app uses AWS services, including Bedrock LLMs, for natural language processing and file analysis.
- Includes a memory feature for conversation continuity.

## Setting up an S3 Bucket on AWS
### Follow these steps to create an S3 bucket on AWS:

1. Log in to AWS Console:
Navigate to AWS Management Console.
Log in using your AWS credentials.

2. Go to S3:
In the AWS Management Console, search for "S3" in the services search bar and click on it.

3. Create a New Bucket:
Click the "Create Bucket" button.
Enter a unique bucket name. Bucket names must be globally unique across all AWS users.
Select your preferred AWS Region.
Configure any additional settings (e.g., public/private access, versioning, etc.), or leave the defaults for a simple setup.
Click "Create Bucket" at the bottom of the page.

4. Get Your Bucket Name:
Once created, you can see the bucket name in the S3 dashboard.
Copy the name of the bucket and add it to your .env file as BUCKET_NAME (elaborated on below).

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
Create an .env file in the root of the project. The .env file should contain your credentials in single quotes for accessing AWS services:

```
REGION_NAME=your_region_name
ACCESS_KEY=your_access_key
SECRET_ACCESS_KEY=your_secret_access_key
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
