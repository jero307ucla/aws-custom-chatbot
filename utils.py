from dotenv import load_dotenv
import os
import boto3
from langchain.llms.bedrock import Bedrock


load_dotenv()

# AWS credentials
aws_access_key_id = os.getenv('ACCESS_KEY')
aws_secret_access_key = os.getenv('SECRET_ACCESS_KEY')
aws_region = os.getenv('REGION_NAME')
BUCKET_NAME = os.getenv("BUCKET_NAME")
folder_path="/tmp/"

bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )

def create_llm(temperature=0):
    # Initialize the custom LLM with the Bedrock client
    model_id = 'anthropic.claude-v2'
    # model_id = 'anthropic.claude-3-opus-20240229-v1:0'
    # model_id = 'amazon.titan-text-premier-v1:0'
    llm = Bedrock(
        model_id=model_id, 
        client=bedrock_client, 
        model_kwargs={"temperature": temperature}
    )
    # llm = BedrockChat(
    #     model_id=model_id,
    #     client=bedrock_client
    # )
    return llm