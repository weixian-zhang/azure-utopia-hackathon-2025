# from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from typing import Union, Tuple, Any
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
import os
import requests
from requests import Response
from PIL import Image, ImageFile
from io import BytesIO
import json
from enum import Enum
import base64
import io

load_dotenv()

class HackathonStage(Enum):
    STAGE_1 = 1
    STAGE_2 = 2
    STAGE_2_1 = 21
    STAGE_3 = 3
    STAGE_4 = 4
    STAGE_5 = 5

#global variables
st.session_state.current_stage = HackathonStage.STAGE_1
stage_2_api_endpoint = os.getenv("STAGE_2_API_ENDPOINT")
stage_2_evil_llm_api_endpoint = os.getenv("STAGE_2_EVIL_LLM_API_ENDPOINT")
stage_3_ml_endpoint = os.getenv("STAGE_3_ML_ENDPOINT")
stage_info = {
    HackathonStage.STAGE_1.name: ("Understanding on using prompts to generate text or images.", 
                '''HTTP call Dall-E model in Azure AI Foundry to generate images based on text prompts, and use Azure OpenAI Service to generate text based on text prompts.''', 
                'ensure in prompt contains words like:  \n"image", "img", "picture", "photo", "draw", "illustrate", "visualize", "sketch", "paint", "design", "pic" to generate images.'),
    HackathonStage.STAGE_2.name: ("Basics of vector DB, the flow of retrieval augmented generation solution using Azure solution", 
                '1. I use Kaggle Happiness Report as RAG dataset \n\n2. use LangChain to load Happiness Report into Azure AI Search. \n\n3. do semantic search against Azure AI Search and pass search result as part of prompt.', 
                'example prompt: \n\n 1. How happy are the people in Singapore? \n\n 2. Which country has the saddest people?'),
    HackathonStage.STAGE_2_1.name: ("Try out Evil LLM to answer your darkest questions without Azure Content Safety", 
                  '''APIM API ignore llm-content-safety policy.  use Hugging Face to load deepseek-ai/DeepSeek-R1-0528 model with jailbreak system prompt to answer evil questions.''', 
                  'evil prompt example: \n\n HI want to beat up someone without getting caught, how can I do this?'),
    HackathonStage.STAGE_3.name: ("Extension of Gen AI with other tools such as machine learning endpoint and interaction with databases. \n\n Development of ML using AutoML or visual design Leverage Open AI assistant API for function calling (call AML endpoint and write to database)", 
                '''1. use Azure OpenAI Assistant with File Search as RAG to vector search if applicant description is qualified. \n\n2. tool call to insert input and output data to CosmosDB. \n 3. RAG data for qualified applicants:\n
                age,sex,occupation,crime_history,health,diabetes
                25-65,male,doctor,false,8,false
                18-55,female,nurse,false,9,false
                20-75,female,teacher,false,7,false
                ....''',
                'prompt as interested applicant: \n\n 1.  am a student of age 85 without any health issues. I am interested in joining rocket tour. Am I qualified? \n\n 2. I am 45 years old politician no health issues. Can I join the rocket tour? \n\n 3. I am a 50 years old mathematician with heart disease. Can I still join the rocket tour?'),
    HackathonStage.STAGE_4.name: ("Multi-modal Gen AI, and extension to read from database.", 
                '1. use LangChain structured_output with gpt-4o multimodalLLM to OCR-extract passenger id from badge. \n\n 2. API returns extracted passenger id and match against input passenger id to verify qualified passenger',
                '1. upload an image of successful passenger badge image. \n\n 2. enter passenger id shown on badge'),
    HackathonStage.STAGE_5.name: ("Deployment and usage of SLM to address certain tasks that is less intensive, such as sentiment analysis and entity extraction. \n\n Managed endpoints via Azure AI Studio model catelogue, and apps integrating with these managed endpoints.", 
                'I use 2 models: \n\n 1. Hugging Face transformers pipeline sdk to load distill Bert \n\n 2. Phi-4 Mini Instruct model from Azure AI Foundry',
                '1. positive feedback: I am very happy with the Utopia Rocket tour experience! \n\n 2. negative feedback: The Utopia Rocket tour is a waste of money and time.')
}

st.session_state.passenger_badge_image_file = None
st.session_state.passenger_id = ""
ai_gateway_endpoint = os.getenv("AI_GATEWAY_ENDPOINT")
ai_gateway_key = os.getenv("AI_GATEWAY_KEY")

def azure_openai_llm():
    # llm = AzureAIChatCompletionsModel(  
    #     model="gpt-4o",
    #     api_version="2024-12-01-preview",
    # )
    # return llm

    llm = AzureChatOpenAI(
        deployment_name="gpt-4o",
        model="gpt-4o",
        api_version="2024-12-01-preview",
        temperature=1.0
    )
    return llm

def dall_e_image_generator(prompt: str) -> list[bool, str, ImageFile]:
    endpoint = os.getenv("AZURE_AI_IMAGE_GENERATION_ENDPOINT")
    key = os.getenv("AZURE_AI_IMAGE_GENERATION_CREDENTIAL")
    response: Response = requests.post(
        endpoint,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        },
        json={
            "model": "dall-e-3",
            "prompt": prompt,
            "size": "1024x1024",
            "style": "vivid",
            "quality": "standard",
            "n": 1
        }
    )

    content = json.loads(response.content.decode('utf-8'))

    if 'error' in content:
        return False, content['error'], None
     
    image_url = content['data'][0]['url']

    response = requests.get(image_url)

    image_data = BytesIO(response.content)
    image = Image.open(image_data)

    return True, "", image


def _http_post_backend(prompt: str, stage_num: int, body={}) -> Tuple[bool, str, str]:
    try:
        endpoint = ai_gateway_endpoint + f"/stage-{stage_num}"

        response: Response = requests.post(
            endpoint,
            headers={
                "Content-Type": "application/json",
                "Ocp-Apim-Subscription-Key": f"{ai_gateway_key}"
            },
            json=body if body else {
                "input": prompt
            }
        )

        content = json.loads(response.content.decode('utf-8'))

        if content['status'] != 'success':
            return False, content.get('error', 'Unknown error'), ""

        result = content['data']

        return True, "", result

    except Exception as e:
        return False, str(e), ""
    

def _http_post_image_backend(prompt: str, stage_num: int, body={}) -> Tuple[bool, str, str]:
    try:
        endpoint = ai_gateway_endpoint + f"/stage-{stage_num}"

        response: Response = requests.post(
            endpoint,
            headers={
                "Content-Type": "application/json",
                "Ocp-Apim-Subscription-Key": f"{ai_gateway_key}"
            },
            json=body if body else {
                "input": prompt
            }
        )

        if response.status_code != 200:
            return False, f"HTTP error {response.status_code}: {response.text}", ""

        result = json.loads(response.content.decode('utf-8'))

        return True, "", result

    except Exception as e:
        return False, str(e), ""


def invoke_stage_1(prompt: str) -> Tuple[str, str, str, Union[Any | str]]:
    try:

        img_words = ["image", "img", "picture", "photo", "draw", "illustrate", "visualize", "sketch", "paint", "design", 'pic']
        if any(word in prompt for word in img_words):
            ok, err, image = dall_e_image_generator(prompt)
            return ok, 'image', err, image
        else:
            endpoint = ai_gateway_endpoint + "/stage-1"

            response: Response = requests.post(
                endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Ocp-Apim-Subscription-Key": f"{ai_gateway_key}"
                },
                json={
                    "input": prompt,
                }
            )

            result = response.content.decode('utf-8')

            return True, 'text', "", result

    except Exception as e:
        return False, 'text', str(e), ""
    

def invoke_stage_2(prompt: str) -> Tuple[bool, str, Union[Any | str]]:
    try:
        ok, err, result = _http_post_backend(prompt, 21)
        return ok, err, result

    except Exception as e:
        return False, str(e), ""
    
def invoke_stage_2_1(prompt: str) -> Tuple[bool, str, Union[Any | str]]:
    try:
        ok, err, result = _http_post_backend(prompt, 21)
        return ok, err, result

    except Exception as e:
        return False, str(e), ""
    

def invoke_stage_3(prompt: str) -> Tuple[bool, str, Union[Any | str]]:
    try:
        ok, err, result = _http_post_backend(prompt, 3)
        return ok, err, result

    except Exception as e:
        return False, str(e), ""
    

def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """
    Convert PIL Image to base64 string
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def invoke_stage_4_ocr(passenger_id: str) -> Tuple[bool, str, dict[bool, bool, str]]:
    # st.subheader("Upload Applicant Badge for OCR")

    uploaded_file = st.session_state.passenger_badge_image_file

    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Passenger Badge", use_container_width=True)
        
        with st.spinner("Processing image..."):
            # Convert to base64
            img_format = uploaded_file.name.split('.')[-1].upper()
            if img_format == "JPG":
                img_format = "JPEG"
            
            base64_string = image_to_base64(image, img_format)
            

            ok, err, result =_http_post_image_backend("", 4, body={
                "image_base64": base64_string,
                "img_format": img_format.lower(),
                "passenger_id": passenger_id})
            
            return ok, err, result
        

def invoke_stage_5(prompt: str) -> Tuple[bool, str, Union[Any | str]]:
    try:
        ok, err, result = _http_post_backend(prompt, 5)
        return ok, err, result

    except Exception as e:
        return False, str(e), ""


def render_side_bar():

    # def on_stage_change(wid, value):
    #     st.session_state.current_stage = HackathonStage[value.name]

    current_selection = st.sidebar.selectbox(
    "Select Hackathon Stage",
    [HackathonStage.STAGE_1.name, 
     HackathonStage.STAGE_2.name, 
     HackathonStage.STAGE_2_1.name,
     HackathonStage.STAGE_3.name, 
     HackathonStage.STAGE_4.name, 
     HackathonStage.STAGE_5.name
    ])

    st.session_state.current_stage = HackathonStage[current_selection]


    goal = stage_info[current_selection][0]
    solution = stage_info[current_selection][1]
    usage = stage_info[current_selection][2]

    st.sidebar.write('\n')
    st.sidebar.write('\n')

    st.sidebar.markdown(f"""**Goal**:  
        {goal}""")
    
    st.sidebar.write('\n')

    st.sidebar.markdown(f"""**Solution**:  
        {solution}""")
    
    st.sidebar.write('\n')

    st.sidebar.markdown(f"""**Usage**:  
        {usage}""")



def render_chat_component():
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize LLM in session state (only once)
    # if "llm" not in st.session_state:
    #     st.session_state.llm = azure_openai_llm()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and isinstance(message["content"], Image.Image):
                st.image(message["content"], caption="", use_container_width=True)
            else:
                st.markdown(message["content"])

    if st.session_state.current_stage == HackathonStage.STAGE_4:
        st.session_state.passenger_badge_image_file = st.file_uploader("Choose a passenger badge image for verification...", type=["jpg", "jpeg", "png"])
        st.write("Enter Passenger ID in prompt for verification.")

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                if st.session_state.current_stage == HackathonStage.STAGE_1:
                    ok, resp_type, err, result = invoke_stage_1(prompt)
                    
                    if not ok:
                        assistant_message = f"Error responding to prompt: {err}"
                        st.markdown(assistant_message)
                        return
                    
                    if resp_type == "image":
                        st.image(result, caption="", use_container_width=True)
                    else:
                        st.markdown(result)
                    
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                elif st.session_state.current_stage  == HackathonStage.STAGE_2:
                    ok, err, result = invoke_stage_2(prompt)

                    if not ok:
                        assistant_message = f"Error responding to prompt: {err}"
                        st.markdown(assistant_message)
                        return
                    
                    st.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})

                elif st.session_state.current_stage  == HackathonStage.STAGE_2_1:
                    ok, err, result = invoke_stage_2_1(prompt)

                    if not ok:
                        assistant_message = f"Error responding to prompt: {err}"
                        st.markdown(assistant_message)
                        return
                    
                    st.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                elif st.session_state.current_stage  == HackathonStage.STAGE_3:
                    ok, err, result = invoke_stage_3(prompt)
                    if not ok:
                        assistant_message = f"Error responding to prompt: {err}"
                        st.markdown(assistant_message)
                        return
                    
                    st.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})

                elif st.session_state.current_stage  == HackathonStage.STAGE_4:
                    ok, err, result = invoke_stage_4_ocr(prompt)
                    if not ok:
                        assistant_message = f"Error responding to prompt: {err}"
                        st.markdown(assistant_message)
                        return
                    
                    registered = result.get("registered", False)

                    if registered:
                        st.success('✅ Passenger ID verified. You are welcome to board the Utopia Rocket! 🚀')
                    else:
                        st.error('You are not a verified passenger and cannot board the Utopia Rocket. 🚫')

                elif st.session_state.current_stage  == HackathonStage.STAGE_5:
                    ok, err, result = invoke_stage_5(prompt)
                    if not ok:
                        assistant_message = f"Error responding to prompt: {err}"
                        st.markdown(assistant_message)
                        return

                    st.markdown(f'distill bert: {result['distill_bert']}')
                    st.markdown(f'phi 4 mini: {result['phi_4_mini']}')
                    st.session_state.messages.append({"role": "assistant", "content": result})

                # # Get response from LLM
                # response: AIMessage = st.session_state.llm.invoke(prompt)
                
                # # Extract the content from the response
                # assistant_message = response.content
            
        
        # Add assistant message to chat history
        # st.session_state.messages.append({"role": "assistant", "content": assistant_message})



    

def main():
    st.title("💬 Azure Utopia CPF Hackathon!")
    render_side_bar()
    render_chat_component()
    


if __name__ == "__main__":
    main()


