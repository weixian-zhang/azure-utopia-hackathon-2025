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
    HackathonStage.STAGE_3.name: ("Extension of Gen AI with other tools such as machine learning endpoint and interaction with databases.", 
                '1. use Azure OpenAI Assistant with File Search as RAG to vector search if applicant description is qualified. \n\n2. tool call to insert input and output data to CosmosDB',
                'prompt as interested applicant: \n\n I am a teacher who teaches math, I am of age 31 without any health issues. I am interested in joining the Utopia Rocjet tour. Am I qualified? '),
    HackathonStage.STAGE_4.name: ("Multi-modal Gen AI, and extension to read from database.", 
                '1. use LangChain multimodal LLM to OCR-extract applicant id from image badge. \n\n 2. query Sqlite where image.applicant.id == Sqlite.applicant.id. \n\n 3. Use Azure Content Safety API to analyze image for content safety.',
                '1. upload an image of successful applicant badge. \n\n 2. upload malicious image for content safety check.'),
    HackathonStage.STAGE_5.name: ("Deployment and usage of SLM to address certain tasks that is less intensive, such as sentiment analysis and entity extraction.", 
                'use Azure OpenAI SDK to call Azure AI Foundry Llama model to perform: \n\n 1. sentiment analysis \n\n 2. entity extraction',
                '1. prompt feedback with positive sentiment: I am very happy with the Utopia Rocket tour experience! \n\n 2. prompt feedback with negative sentiment: The Utopia Rocket tour is a waste of money and time.')
}

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
                    pass
                elif st.session_state.current_stage  == HackathonStage.STAGE_4:
                    pass
                elif st.session_state.current_stage  == HackathonStage.STAGE_5:
                    pass

                # # Get response from LLM
                # response: AIMessage = st.session_state.llm.invoke(prompt)
                
                # # Extract the content from the response
                # assistant_message = response.content
            
        
        # Add assistant message to chat history
        # st.session_state.messages.append({"role": "assistant", "content": assistant_message})


def _http_post_backend(prompt: str, stage_num: int) -> Tuple[bool, str, str]:
    try:
        endpoint = ai_gateway_endpoint + f"/stage-{stage_num}"

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

        content = json.loads(response.content.decode('utf-8'))

        if content['status'] != 'success':
            return False, content.get('error', 'Unknown error'), ""

        result = content['data']

        return True, "", result

    except Exception as e:
        return False, str(e), ""
    

def main():
    st.title("💬 Azure Utopia CPF Hackathon!")
    render_side_bar()
    render_chat_component()
    


if __name__ == "__main__":
    main()


