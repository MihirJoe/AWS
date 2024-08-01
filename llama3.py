import boto3
import json


prompt_data = """
    Act as Shakespeare and write a poem on machine learning
"""

bedrock = boto3.client(service_name="bedrock-runtime")

# goes into API call
payload = {
    "prompt":"[INST]" + prompt_data + "[/INST]",
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}

body = json.dumps(payload) # converts to json

model_id = "meta.llama3-8b-instruct-v1:0"

response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response.get("body").read()) # get the body of the response

response_text = response_body['generation'] # the text of the response
print(response_text)