from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
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


def azure_ai_llm():
    llm = AzureAIChatCompletionsModel(  
        model="gpt-4o",
        api_version="2024-12-01-preview",
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

# image = dall_e_image_generator("generate image of cartoonish cuttlefish")
# image.show()

# response: AIMessage = azure_ai_llm().invoke("Hello from Azure AI LLM!")
# print(response.pretty_print())


def main():
    import streamlit as st
    
    st.title("💬 Azure Utopia CPF Hackathon!")
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize LLM in session state (only once)
    if "llm" not in st.session_state:
        st.session_state.llm = azure_ai_llm()
    
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


if __name__ == "__main__":
    main()


