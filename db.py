import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
load_dotenv()

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv('NEO4J_USERNAME'), os.getenv("NEO4J_PASSWORD"))

driver = GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()
