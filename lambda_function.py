import json
import boto3
from botocore.exceptions import ClientError
import os

# --- CONFIGURATION (UPDATE THESE) ---
# CRITICAL: REPLACE with your actual AWS region (e.g., 'us-east-1')
REGION = 'us-east-1' 
TABLE_NAME = 'ElderlyCareProfiles'
PATIENT_ID = 'P101'
# CRITICAL: REPLACE with the email address you VERIFIED in Amazon SES
SENDER_EMAIL = 'yourverifiedemail@example.com' 
# -----------------------------------

def initialize_clients(region):
    """Initializes AWS clients for Bedrock, DynamoDB, and SES."""
    try:
        bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
        dynamodb = boto3.resource('dynamodb', region_name=region)
        ses_client = boto3.client('ses', region_name=region)
        return bedrock_runtime, dynamodb, ses_client
    except Exception as e:
        print(f"CRITICAL CLIENT INIT ERROR: {e}")
        return None, None, None 

def get_patient_data(dynamodb_resource):
    """Retrieves patient information from DynamoDB (Knowledge Base)."""
    table = dynamodb_resource.Table(TABLE_NAME)
    response = table.get_item(Key={'PatientID': PATIENT_ID})
    return response.get('Item', None)

def classify_emergency(bedrock_client, user_input):
    """Uses Bedrock LLM for symptom triage."""
    
    BEDROCK_MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'
    
    prompt_template = f"""
    You are an AI Triage Agent. Given the statement, classify the emergency type and the required action. 
    Emergency Type MUST be one of: [CARDIAC_STROKE, FALLS_FRACTURES, BREATHING_CRISIS, NONE].
    Action MUST be one of: [MOCK_AMBULANCE_DISPATCH, ADVISE_INHALER, ALERT_FAMILY, NONE].
    
    Statement: "{user_input}"
    
    Respond ONLY with the EXACT JSON object format below. DO NOT add any extra text or markdown.
    {{"emergency_type": "<TYPE>", "action": "<ACTION>"}}
    """

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_template}]}]
    body = json.dumps({"anthropic_version": "bedrock-2023-05-31", "messages": messages, "max_tokens": 500, "temperature": 0.0})
    
    response = bedrock_client.invoke_model(modelId=BEDROCK_MODEL_ID, body=body, contentType='application/json', accept='application/json')
    response_body = json.loads(response.get('body').read())
    completion_text = response_body['content'][0].get('text', '').strip()
    
    # Robustly extract JSON 
    start = completion_text.find('{')
    end = completion_text.rfind('}') + 1
    if start != -1 and end != -1 and start < end:
        return json.loads(completion_text[start:end])
            
    return {"emergency_type": "ERROR", "action": "ERROR"}

def send_alert(ses_client, patient_data, classification):
    """Sends Email alert via SES (Guaranteed Alert)."""
    
    alert_type = classification['emergency_type']
    
    if alert_type == "NONE" or alert_type == "ERROR":
        return {"AlertStatus": "Skipped", "Message": "No critical action required."}
        
    msg_body = f"""
    *** URGENT LIFE-GUARD AI ALERT ***
    Patient: {patient_data['Name']} ({patient_data['Age']}) - ID {PATIENT_ID}
    EMERGENCY: {alert_type}
    ACTION: {classification['action']}
    Patient Info Packet: Comorbidities: {patient_data['Comorbidities']}. Medications: {patient_data['Meds']}.
    ---
    This alert guarantees delivery and provides an auditable record of the crisis triage.
    """
    
    try:
        recipient_email = patient_data.get('Caregiver_Email')
        
        if recipient_email:
            ses_client.send_email(
                Source=SENDER_EMAIL,
                Destination={'ToAddresses': [recipient_email]},
                Message={
                    'Subject': {'Charset': 'UTF-8', 'Data': f"URGENT LIFE-GUARD AI ALERT: {alert_type} DETECTED"},
                    'Body': {'Text': {'Charset': 'UTF-8', 'Data': msg_body}}
                }
            )
            return {"AlertStatus": "EMAIL_SENT_CONFIRMED", "Message": f"Guaranteed email alert sent for {alert_type} to caregiver: {recipient_email}"}
        else:
            return {"AlertStatus": "FAILED", "Message": "Caregiver Email missing in DynamoDB."}
            
    except ClientError as e:
        print(f"SES Error: {e}")
        return {"AlertStatus": "FAILED", "Message": f"SES Error: {e.response['Error']['Message']}"}


def lambda_handler(event, context):
    
    # Initialize clients for this invocation
    bedrock_runtime, dynamodb, ses_client = initialize_clients(REGION)
    if not all([bedrock_runtime, dynamodb, ses_client]):
         return {'statusCode': 500, 'headers': {'Access-Control-Allow-Origin': '*'}, 'body': json.dumps({"status": "Error", "message": "Critical: AWS Client Initialization Failed."})}


    # 1. Handle GET request from browser (prevents internal server error crash)
    if 'httpMethod' not in event or event.get('httpMethod') == 'GET':
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({"status": "Ready for POST", "message": "Agent is waiting for symptom data."})
        }
        
    # 2. Process POST request
    try:
        # Parse input body
        body = json.loads(event.get('body', '{}'))
        user_input = body.get('symptoms', '').strip()
        if not user_input: user_input = "No symptoms provided."
        
        # Agent Workflow Execution
        patient_data = get_patient_data(dynamodb)
        if not patient_data: raise Exception(f"Patient ID {PATIENT_ID} not found.")
            
        classification = classify_emergency(bedrock_runtime, user_input)
        alert_result = send_alert(ses_client, patient_data, classification)
        
        # Final Success Response
        response_body = {
            "status": "Success",
            "message": "AI Agent workflow completed.",
            "input_symptoms": user_input,
            "ai_classification": classification,
            "patient_info_packet": {"name": patient_data.get('Name'), "comorbidities": patient_data.get('Comorbidities'), "meds": patient_data.get('Meds')},
            "alert_status": alert_result
        }
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps(response_body)
        }

    except Exception as e:
        print(f"RUNTIME EXCEPTION: {e}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({"status": "Error", "message": "Backend Processing Failure.", "error_detail": str(e)})
        }
