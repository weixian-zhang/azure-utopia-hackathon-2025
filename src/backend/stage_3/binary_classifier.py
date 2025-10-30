import os
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
from dotenv import load_dotenv
load_dotenv()



class AcceptedResponse(BaseModel):
    accepted: bool = Field(description="indicates whether applicant is accepted or not")
    reason: str = Field(description="Brief explanation for the decision")
    confidence: str = Field(description="Confidence level: high, medium, or low")


class BinaryClassifier:

    def __init__(self):

        self.accepted_csv, self.rejected_csv = self.load_data()
    

    def load_data(self) -> bool:

        data_path = os.path.join(os.path.dirname(__file__), 'stage-3-data.csv')

        df = pd.read_csv(data_path)
        df = df.drop('PassengerId', axis=1)
        df = df.drop('Pclass', axis=1)

        accepted = df[df['Selected'] == 1]

        rejected = df[df['Selected'] == 0]

        return accepted.to_csv(index=False), rejected.to_csv(index=False)
         

    def invoke(self, input: str) -> AcceptedResponse:

        #passenger_features_dict = self.extract_features(input)

        system_prompt = f"""You are a binary classifier for determining if passengers are qualified for a rocket tour.

        header fields of CSV data:

        Selected: 1 means accepted, 0 means rejected
        Name: name of passenger
        Sex: male or female
        Age: age
        SibSp: number of siblings/spouses aboard
        Parch: number of parents/children aboard
        Fare: amount of money paid for ticket
        Occupation: occupation of the passenger
        FavoriteColor: favourite color of the passenger
        Hobby: hobby of the passenger

        examples of accepted passengers:
        {self.accepted_csv}

        examples of rejected passengers:
        {self.rejected_csv}


        Study the patterns of these examples of accepted and rejected passengers, compare with input passenger features.
        Passenger features to compare:

        Selected: 1 means accepted, 0 means rejected
        Name: name of passenger
        Sex: male or female
        Age: age
        SibSp: number of siblings/spouses aboard
        Parch: number of parents/children aboard
        Fare: amount of money paid for ticket
        Occupation: occupation of the passenger
        FavoriteColor: favourite color of the passenger
        Hobby: hobby of the passenger

        Based on your study, classify the input passenger as accepted or rejected.
        """

                
        user_prompt = f"""Classify passenger to accepted or rejected based on features: {input}."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        llm = AzureChatOpenAI(
            deployment_name="gpt-4o",
            model="gpt-4o",
            api_version="2024-12-01-preview",
            temperature=0.0
        ).with_structured_output(AcceptedResponse)

        response = llm.invoke(messages)

        return response
    

if __name__ == "__main__":
    input_1 = "Excited to board Azure Utopia, Doe, Ms. Jane paid a fare of 34.5 and travels alone..."
    input_2 = 'With a heart set on Azure Utopia, Gale, Mr. Shadrach is a 34-year-old male second-class Artist who paid a fare of 21, treasures Gardening, prefers White, and travels with one sibling or spouse and no parents or children.'
    input_3 = 'Eager to explore Azure Utopia, Rosblom, Mr. Viktor Richard is an 18-year-old male third-class Grocer who paid a fare of 20.2125, enjoys Walking, prefers Brown, and is accompanied by one sibling or spouse and one parent or child.'
    input_4 = 'Radiant with anticipation for Azure Utopia, Phillips, Miss. Kate Florence ("Mrs Kate Louise Phillips Marshall") is a 19-year-old female second-class Tailor who paid a fare of 26, adores Knitting, favors Yellow, and travels alone with 0 siblings/spouse and 0 parents/children.'
    checker = BinaryClassifier()
    # print(checker.invoke(input_1).accepted)
    # print(checker.invoke(input_2).accepted)
    print(checker.invoke(input_4).accepted)
    #print(f"Is the passenger accepted? {accepted.accepted}, Reason: {accepted.reason}, Confidence: {accepted.confidence}")