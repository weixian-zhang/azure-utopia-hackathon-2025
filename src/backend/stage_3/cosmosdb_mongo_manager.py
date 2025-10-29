from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import os
from datetime import datetime

class CosmosMongoDBManager:
    def __init__(self):
        connection_string = os.getenv("COSMOS_MONGODB_CONNECTION_STRING")
        
        if not connection_string:
            raise ValueError("COSMOS_MONGODB_CONNECTION_STRING environment variable not set")

        self.client = MongoClient(connection_string)
        self.database = self.client["hackathon_db"]
        self.collection = self.database["passenger_qualifying_info"]
        
        # Test connection
        try:
            self.client.admin.command('ping')
            print("Successfully connected to Cosmos DB")
        except ConnectionFailure as e:
            print(f"Failed to connect to Cosmos DB: {e}")
            raise
        

    def insert_passenger_qualifying_info(self, prompt, assistant_response) -> str:
        """
        Insert a single applicant document
        """
        try:
            document = {
                "prompt": prompt,
                "assistant_response": assistant_response,
                "created_at": datetime.now()
            }

            result = self.collection.insert_one(document)
            print(f"Inserted passenger qualifying info with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        # except DuplicateKeyError:
        #     print(f"Passenger qualifying info with ID {applicant_data.get('id')} already exists")
        #     raise
        except Exception as e:
            print(f"Error inserting passenger qualifying info: {e}")
            raise