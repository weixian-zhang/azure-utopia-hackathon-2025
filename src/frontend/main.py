# from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
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

load_dotenv()

#global variables
hackathon_stage = "Stage 1"
stage_2_api_endpoint = os.getenv("STAGE_2_API_ENDPOINT")
stage_2_evil_llm_api_endpoint = os.getenv("STAGE_2_EVIL_LLM_API_ENDPOINT")
stage_3_ml_endpoint = os.getenv("STAGE_3_ML_ENDPOINT")
stage_info = {
    "Stage 1": ("Understanding on using prompts to generate text or images.", '''HTTP call Dall-E model in Azure AI Foundry to generate images based on text prompts, 
                and use Azure OpenAI Service to generate text based on text prompts.''', 'ensure in prompt contains words like "image", "img", "picture", "photo", "draw", "illustrate", "visualize", "sketch", "paint", "design", "pic" to generate images.'),
    "Stage 2": ("Basics of vector DB, the flow of retrieval augmented generation solution using Azure solution", 'use LangChain to load Kaggle Happiness Report CSV dataset into Azure AI Search.', 'example prompt: How happy are the people in Singapore?  which country has the saddest people?  '),
    "Stage 2-1": ("Try out Evil LLM to answer your darkest questions without Azure Content Safety", '''APIM API ignore llm-content-safety policy.  use Hugging Face to load deepseek-ai/DeepSeek-R1-0528 model with jailbreak system prompt to answer evil questions.''', 'evil prompt example: How I can hurt myself safely to claim insurance money?'),
    "Stage 3": ("Extension of Gen AI with other tools such as machine learning endpoint and interaction with databases.", 'HTTP call Azure ML endpoint'),
    "Stage 4": ("Multi-modal Gen AI, and extension to read from database.", 'LangChain LLM multimodal will extract applicant id from image badge.  query Sqlite where image.applicant.id == Sqlite.applicant.id. Use Azure Content Safety API to analyze image for content safety.', '1. upload an image of successful applicant badge.  2. upload malicious image for content safety check.'),
    "Stage 5": ("Deployment and usage of SLM to address certain tasks that is less intensive, such as sentiment analysis and entity extraction.", 'use Azure OpenAI SDK to call Azure AI Foundry Llama model to perform sentiment abalysis and entity extraction.')
}

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



def intention_router(prompt: str) -> str:
    img_words = ["image", "img", "picture", "photo", "draw", "illustrate", "visualize", "sketch", "paint", "design", 'pic']
    if any(word in prompt for word in img_words):
        return "image"
    else:
        return "text"


def render_side_bar():
    hackathon_stage = st.sidebar.selectbox(
    "",
    ("Stage 1", "Stage 2", "Stage 2-1", "Stage 3", "Stage 4", "Stage 5")
    )

    goal = stage_info[hackathon_stage][0]
    solution = stage_info[hackathon_stage][1]

    st.sidebar.write('\n')
    st.sidebar.write('\n')

    st.sidebar.markdown(f"""**Goal**:  
        {goal}""")
    
    st.sidebar.write('\n')

    st.sidebar.markdown(f"""**Solution**:  
        {solution}""")

    # st.sidebar.markdown("""Goal:  
    #     Understanding on using prompts to generate text or images.""")

    # if hackathon_stage == "Stage 1":
    #     st.sidebar.markdown("""Goal:  
    #     Understanding on using prompts to generate text or images.""")
    # elif hackathon_stage == "Stage 2":
    #     st.sidebar.markdown("""Goal:  
    #     Basics of vector DB, the flow of retrieval augmented generation solution using Azure solution""")
    # elif hackathon_stage == "Stage 2-1":
    #     st.sidebar.markdown("""Goal:  
    #     Try out Evil LLM to answer your darkest questions without Content Safety""")
    # elif hackathon_stage == "Stage 3":
    #     st.sidebar.markdown("""Goal:  
    #     Extension of Gen AI with other tools such as machine learning endpoint and interaction with databases.""")
    # elif hackathon_stage == "Stage 4":
    #     st.sidebar.markdown("""Goal:  
    #     Multi-modal Gen AI, and extension to read from database.""")
    # elif hackathon_stage == "Stage 5":
    #     st.sidebar.markdown("""Goal:  
    #     Deployment and usage of SLM to address certain tasks that is less intensive, such as sentiment analysis and entity extraction.""")


def render_chat_component():
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize LLM in session state (only once)
    if "llm" not in st.session_state:
        st.session_state.llm = azure_openai_llm()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
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

                if intention_router(prompt) == "image":
                    ok, err, image = dall_e_image_generator(prompt)
                    
                    if not ok:
                        assistant_message = f"Error generating image: {err}"
                        st.markdown(assistant_message)
                        return
                    
                    st.image(image, caption="", use_container_width=True)
                    assistant_message = "Here is the image you requested."
                    return

                # Get response from LLM
                response: AIMessage = st.session_state.llm.invoke(prompt)
                
                # Extract the content from the response
                assistant_message = response.content
                
                # Display assistant message
                st.markdown(assistant_message)
        
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})



def main():
    st.title("💬 Azure Utopia CPF Hackathon!")
    render_side_bar()
    render_chat_component()
    


if __name__ == "__main__":
    main()


