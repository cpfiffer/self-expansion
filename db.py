import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

try:  # to connect to the graph db
    URI = os.environ["NEO4J_URI"]
    assert URI
    AUTH = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
except Exception as e:
    raise ValueError("Error fetching Neo4J credentials from environment") from e


driver = GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()
