

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
import base64
from pydantic import BaseModel, Field

class PassengerBadgeInfo(BaseModel):
    passenger_id: str = Field(description="The passenger ID number")


class LLMImageOCR:


    def __init__(self):
        self.llm = AzureChatOpenAI(
            deployment_name="gpt-4o",
            model="gpt-4o",
            api_version="2024-12-01-preview",
            temperature=0.0
        ).with_structured_output(PassengerBadgeInfo)


    def extract_passenger_badge_id_from_base64_image(self, base64_string: str, image_format: str = "png") -> str:
        """
        Perform OCR on base64 encoded image
        
        Args:
            base64_string: Base64 encoded image string (without data URI prefix)
            image_format: Image format (jpeg, png, etc.)
        """
        
        
        # Create data URI from base64 string
        data_uri = f"data:image/{image_format};base64,{base64_string}"
        
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Extract all text from this image. Provide only the extracted text."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri}
                }
            ]
        )

        response = self.llm.invoke([message])

        return response.passenger_id