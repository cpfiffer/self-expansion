from typing import List, Literal
from db import driver
import structured_gen as sg
from pydantic import BaseModel, Field
from rich import print


# Define the data structures
class Question(BaseModel):
    type: Literal["Question"]
    text: str


class Concept(BaseModel):
    type: Literal["Concept"]
    # Text must be lowercase
    text: str = Field(pattern=r"^[a-z ]+$")


class ConceptWithLinks(Concept):
    relationship_type: Literal["IS_A", "AFFECTS", "CONNECTS_TO"]


class Answer(BaseModel):
    type: Literal["Answer"]
    text: str


# Permitted response formats
class FromQuestion(BaseModel):
    """If at a question, may generate an answer."""

    answer: List[Answer]


class FromConcept(BaseModel):
    """If at a concept, may produce questions or relate to concepts"""

    questions: List[Question]
    concepts: List[ConceptWithLinks]


class FromAnswer(BaseModel):
    """If at an answer, may generate concepts or new questions"""

    concepts: List[Concept]
    questions: List[Question]


# Create a core node if it doesn't exist, or return the existing core node ID
def get_or_make_core(question: str):
    with driver.session() as session:
        # Check if node exists and get its ID
        result = session.run(
            """
            MATCH (n:Core {text: $question})
            RETURN n.id as id
        """,
            question=question,
        )

        data = result.data()
        if len(data) > 0:
            return data[0]["id"]

        # Create new node with UUID if it doesn't exist
        result = session.run(
            """
            MERGE (n:Core {text: $question})
            ON CREATE SET n.id = randomUUID()
            RETURN n.id as id
        """,
            question=question,
        )
        data = result.data()
        if len(data) > 0:
            return data[0]["id"]
        else:
            raise ValueError(f"Failed to create new core node for question: {question}")


def load_neighbors(node_id: str, distance: int = 1):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (node {id: $node_id})-[rel]-(neighbor)
            WHERE type(rel) <> "TRAVERSED"
            RETURN
                node.id as node_id,
                node.text as node_text,
                type(rel) as rel_type,
                neighbor.id as neighbor_id,
                neighbor.text as neighbor_text,
                labels(neighbor)[0] as neighbor_type,
                labels(node)[0] as node_type
        """,
            node_id=node_id,
        )
        return result.data()


def load_node(node_id: str):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (node {id: $node_id})
            RETURN node.id as node_id, node.text as node_text, labels(node)[0] as label
        """,
            node_id=node_id,
        )
        return result.single(strict=True)


# Node linking functions
# Relationship Types (all have curiosity score 0-1):
#   RAISES -> (Concept/Core to Question)
#   ANSWERS -> (Answer to Question)
#   SUGGESTS -> (Answer to Concept)
#   RELATES_TO -> (Concept to Concept)
def question_to_concept(question: str, concept: str):
    try:
        question_embedding = sg.embed(question)
        concept_embedding = sg.embed(concept)
    except Exception as e:
        print(f"Skipping connection due to embedding error: {e}")
        return

    with driver.session() as session:
        session.run(
            """
            MERGE (question:Question {text: $question})
            ON CREATE SET question.id = randomUUID(), question.embedding = $question_embedding
            MERGE (concept:Concept {text: $concept})
            ON CREATE SET concept.id = randomUUID(), concept.embedding = $concept_embedding
            MERGE (concept)-[:RAISES]->(question)
        """,
            question=question,
            concept=concept,
            question_embedding=question_embedding,
            concept_embedding=concept_embedding,
        )


def question_to_answer(question: str, answer: str):
    try:
        question_embedding = sg.embed(question)
        answer_embedding = sg.embed(answer)
    except Exception as e:
        print(f"Skipping connection due to embedding error: {e}")
        return

    with driver.session() as session:
        session.run(
            """
            MERGE (question:Question {text: $question})
            ON CREATE SET question.id = randomUUID(), question.embedding = $question_embedding
            MERGE (answer:Answer {text: $answer})
            ON CREATE SET answer.id = randomUUID(), answer.embedding = $answer_embedding
            MERGE (answer)-[:ANSWERS]->(question)
        """,
            question=question,
            answer=answer,
            question_embedding=question_embedding,
            answer_embedding=answer_embedding,
        )


def concept_to_concept(
    concept1: str,
    concept2: str,
    relationship_type: Literal["IS_A", "AFFECTS", "CONNECTS_TO"],
):
    # If the concepts are the same, don't create a relationship
    if concept1 == concept2:
        return

    try:
        concept1_embedding = sg.embed(concept1)
        concept2_embedding = sg.embed(concept2)
    except Exception as e:
        print(f"Skipping connection due to embedding error: {e}")
        return

    with driver.session() as session:
        query = f"""
            MERGE (concept1:Concept {{text: $concept1}})
            ON CREATE SET concept1.id = randomUUID(), concept1.embedding = $concept1_embedding
            MERGE (concept2:Concept {{text: $concept2}})
            ON CREATE SET concept2.id = randomUUID(), concept2.embedding = $concept2_embedding
            MERGE (concept1)-[:{relationship_type}]->(concept2)
        """
        session.run(
            query,
            concept1=concept1,
            concept2=concept2,
            concept1_embedding=concept1_embedding,
            concept2_embedding=concept2_embedding,
        )


def concept_to_question(concept: str, question: str):
    try:
        concept_embedding = sg.embed(concept)
        question_embedding = sg.embed(question)
    except Exception as e:
        print(f"Skipping connection due to embedding error: {e}")
        return

    with driver.session() as session:
        session.run(
            """
            MERGE (concept:Concept {text: $concept})
            ON CREATE SET concept.id = randomUUID(), concept.embedding = $concept_embedding
            MERGE (question:Question {text: $question})
            ON CREATE SET question.id = randomUUID(), question.embedding = $question_embedding
            MERGE (concept)-[:RAISES]->(question)
        """,
            concept=concept,
            question=question,
            concept_embedding=concept_embedding,
            question_embedding=question_embedding,
        )


# Core-specific functions
def core_to_question(core: str, question: str):
    try:
        core_embedding = sg.embed(core)
        question_embedding = sg.embed(question)
    except Exception as e:
        print(f"Skipping connection due to embedding error: {e}")
        return

    with driver.session() as session:
        session.run(
            """
            MERGE (core:Core {text: $core})
            ON CREATE SET core.id = randomUUID(), core.embedding = $core_embedding
            MERGE (question:Question {text: $question})
            ON CREATE SET question.id = randomUUID(), question.embedding = $question_embedding
            MERGE (core)-[:RAISES]->(question)
        """,
            core=core,
            question=question,
            core_embedding=core_embedding,
            question_embedding=question_embedding,
        )


def concept_to_core(concept: str, core: str):
    try:
        concept_embedding = sg.embed(concept)
        core_embedding = sg.embed(core)
    except Exception as e:
        print(f"Skipping connection due to embedding error: {e}")
        return

    with driver.session() as session:
        session.run(
            """
            MERGE (concept:Concept {text: $concept})
            ON CREATE SET concept.id = randomUUID(), concept.embedding = $concept_embedding
            MERGE (core:Core {text: $core})
            ON CREATE SET core.id = randomUUID(), core.embedding = $core_embedding
            MERGE (concept)-[:EXPLAINS]->(core)
        """,
            concept=concept,
            core=core,
            concept_embedding=concept_embedding,
            core_embedding=core_embedding,
        )


def answer_to_concept(answer: str, concept: str):
    try:
        answer_embedding = sg.embed(answer)
        concept_embedding = sg.embed(concept)
    except Exception as e:
        print(f"Skipping connection due to embedding error: {e}")
        return

    with driver.session() as session:
        session.run(
            """
            MERGE (answer:Answer {text: $answer})
            ON CREATE SET answer.id = randomUUID(), answer.embedding = $answer_embedding
            MERGE (concept:Concept {text: $concept})
            ON CREATE SET concept.id = randomUUID(), concept.embedding = $concept_embedding
            MERGE (answer)-[:SUGGESTS]->(concept)
        """,
            answer=answer,
            concept=concept,
            answer_embedding=answer_embedding,
            concept_embedding=concept_embedding,
        )


def answer_to_question(answer: str, question: str):
    try:
        answer_embedding = sg.embed(answer)
        question_embedding = sg.embed(question)
    except Exception as e:
        print(f"Skipping connection due to embedding error: {e}")
        return

    with driver.session() as session:
        session.run(
            """
            MERGE (answer:Answer {text: $answer})
            ON CREATE SET answer.id = randomUUID(), answer.embedding = $answer_embedding
            MERGE (question:Question {text: $question})
            ON CREATE SET question.id = randomUUID(), question.embedding = $question_embedding
            MERGE (answer)-[:ANSWERS]->(question)
        """,
            answer=answer,
            question=question,
            answer_embedding=answer_embedding,
            question_embedding=question_embedding,
        )


def record_traversal(
    from_node_id: str,
    to_node_id: str,
    traversal_type: Literal["random", "core", "neighbor"],
):
    with driver.session() as session:
        session.run(
            """
            MERGE (from_node {id: $from_node_id})
            MERGE (to_node {id: $to_node_id})
            MERGE (from_node)-[:TRAVERSED {timestamp: timestamp(), traversal_type: $traversal_type}]->(to_node)
        """,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            traversal_type=traversal_type,
        )


def clear_db():
    with driver.session() as session:
        session.run(
            """
            MATCH (n) DETACH DELETE n
        """
        )


def random_node_id():
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n) RETURN n.id as id LIMIT 1
        """
        )
        return result.single(strict=True)["id"]


def format_node_neighborhood(node_id, truncate: bool = True):
    # Create ID mapping using ASCII uppercase letters (AA, AB, AC, etc.)
    id_counter = 0
    uuid_to_simple_mapping = {}
    simple_to_uuid_mapping = {}

    def get_simple_id():
        nonlocal id_counter
        # Generate IDs like AA, AB, ..., ZZ
        first = chr(65 + (id_counter // 26))
        second = chr(65 + (id_counter % 26))
        id_counter += 1
        return f"NODE-{first}{second}"

    node = load_node(node_id)
    neighbors = load_neighbors(node_id)
    neighbors_string = f"{node['label'].upper()} {node['node_text']}\n"

    # Add direct neighbors
    if len(neighbors) > 0:
        neighbors_string += "\nDIRECT CONNECTIONS:\n"
        for neighbor in neighbors:
            text = neighbor["neighbor_text"]
            if truncate:
                text = text[:70] + "..." if len(text) > 70 else text
            simple_id = get_simple_id()
            simple_to_uuid_mapping[simple_id] = neighbor["neighbor_id"]
            uuid_to_simple_mapping[neighbor["neighbor_id"]] = simple_id
            neighbors_string += f"{simple_id:<8} {neighbor['rel_type']:<12} {neighbor['neighbor_type'].upper():<10} {text}\n"

    # Add semantically related nodes
    related = find_related_nodes(node_id)

    if len(related) > 0:
        neighbors_string += "\nSEMANTICALLY RELATED:\n"
        for node_type, nodes in related.items():
            if nodes:  # Only add section if there are related nodes
                neighbors_string += f"\n{node_type}s:\n"
                for n in nodes:
                    text = n["node_text"]
                    if truncate:
                        text = text[:70] + "..." if len(text) > 70 else text
                    simple_id = get_simple_id()
                    simple_to_uuid_mapping[simple_id] = n["node_id"]
                    uuid_to_simple_mapping[n["node_id"]] = simple_id
                    neighbors_string += f"{simple_id:<8} {n['score']:<12.2f} {node_type.upper():<10} {text}\n"

    return neighbors_string, uuid_to_simple_mapping, simple_to_uuid_mapping


def find_related_nodes(node_id: str):
    with driver.session() as session:
        result = {}
        for node_type in ["Question", "Concept", "Answer"]:
            result[node_type] = session.run(
                """
                MATCH (m {id: $node_id})
                WHERE m.embedding IS NOT NULL
                CALL db.index.vector.queryNodes(
                    $vector_index_name,
                    $limit,
                    m.embedding
                )
                YIELD node, score
                RETURN node.id as node_id, node.text as node_text, score
            """,
                node_id=node_id,
                vector_index_name=f"{node_type.lower()}_embedding",
                limit=10,
            ).data()
        return result


def remove_index(index_name: str):
    with driver.session() as session:
        session.run(
            f"""
            DROP INDEX {index_name} IF EXISTS
        """
        )


def main(do_clear_db=False, purpose="Support humanity"):
    # Clear the database if requested
    if do_clear_db:
        print("WARNING: Clearing the database")
        clear_db()

    # Create the core node and get its ID
    current_node_id = get_or_make_core(purpose)
    core_node_id = current_node_id

    # Get embedding dimensions
    embedding_dimensions = len(sg.embed(purpose))

    # Remove existing indices
    remove_index("core_id")
    remove_index("question_embedding")
    remove_index("concept_embedding")
    remove_index("answer_embedding")

    # Create indices
    with driver.session() as session:
        # Create regular indices
        index_queries = [
            "CREATE INDEX core_id IF NOT EXISTS FOR (n:Core) ON (n.id)",
            "CREATE INDEX question_id IF NOT EXISTS FOR (n:Question) ON (n.id)",
            "CREATE INDEX concept_id IF NOT EXISTS FOR (n:Concept) ON (n.id)",
            "CREATE INDEX answer_id IF NOT EXISTS FOR (n:Answer) ON (n.id)",
        ]

        # Create vector indices
        vector_index_queries = []
        for node_type in ["Question", "Concept", "Answer"]:
            vector_index_queries.append(
                f"""
                CREATE VECTOR INDEX {node_type.lower()}_embedding IF NOT EXISTS
                FOR (n:{node_type}) ON (n.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: $embedding_dimensions,
                        `vector.similarity_function`: 'COSINE'
                    }}
                }}
            """
            )

        # Execute all queries
        for query in index_queries + vector_index_queries:
            session.run(query, embedding_dimensions=embedding_dimensions)

    # Loop through the main code
    history = []
    while True:
        # Get current node
        current_node = load_node(current_node_id)
        current_node_text = current_node["node_text"]
        current_node_label = current_node["label"]

        # Get the user prompt. Shows previous nodes and actions, then
        # shows the current node.
        prompt = (
            "\n".join([f"{n['label'].upper()} {n['node_text']}" for n in history])
            + f"\nCurrent node: {current_node_label.upper()} {current_node_text}"
        )
        prompt = "Here is the traversal history:\n" + prompt
        # prompt += f"Here are nodes related to the current node:\n" +\
        #       format_node_neighborhood(current_node_id, truncate=False)

        # Check current node type
        result_format = None
        if current_node_label == "Question":
            # May only extract concepts and observations from questions
            result_format = FromQuestion
        elif current_node_label == "Concept" or current_node_label == "Core":
            # May generate other concepts, new questions, or new observations
            result_format = FromConcept
        elif current_node_label == "Answer":
            # May generate new concepts or new questions
            result_format = FromAnswer
        else:
            raise ValueError(f"Unknown node type: {current_node_label}")

        # Get the system prompt
        system_prompt = f"""
        You are a superintelligent AI building a self-expanding knowledge graph.
        Your goal is to achieve the core directive "{purpose}".

        Generate an expansion of the current node. An expansion may include:

        - A list of new questions.
            - Questions should be short and direct.
            - If you generate multiple questions, they should be distinct and not similar.
        - A list of new concepts.
            - Concepts are words or short combinations of words that
              are related to the current node.
        - Concepts may connect to each other.
            - Concepts may be related by IS_A, AFFECTS, or CONNECTS_TO.
            - IS_A: A concept is a type of another concept.
            - AFFECTS: A concept is related to another concept because it affects it.
            - CONNECTS_TO: A concept is generally related to another concept.
        - A list of new answers.
            - Answers should be concise and to the point.

        When concepts can be generated, try to do so. They're important.

        Your role is to understand the core directive "{purpose}".

        Respond in the following JSON format:
        {result_format.model_json_schema()}
        """

        # Generate an expansion
        try:
            # print("generating expansion")
            # result = sg.generate(
            #     sg.messages(user=prompt, system=system_prompt),
            #     response_format=result_format
            # )

            print(prompt)

            result = sg.generate_by_schema(
                sg.messages(user=prompt, system=system_prompt),
                result_format.model_json_schema(),
            )
            expansion = result_format.model_validate_json(
                result.choices[0].message.content
            )
        except Exception as e:
            print(f"Error generating expansion: {e}")
            # Return to the core node
            current_node_id = core_node_id
            continue

        # Link the new nodes to the current node.
        # If we are at a question, we can only provide answers.
        if current_node_label == "Question":
            for answer in expansion.answer:
                answer_to_question(answer.text, current_node_text)

        # If we are at a concept, we can link to the questions and concepts.
        elif current_node_label == "Concept":
            for purpose in expansion.questions:
                concept_to_question(current_node_text, purpose.text)
            for concept in expansion.concepts:
                concept_to_concept(
                    current_node_text, concept.text, concept.relationship_type
                )

        # If we are at an answer, we can link to the concepts and questions.
        elif current_node_label == "Answer":
            for concept in expansion.concepts:
                answer_to_concept(current_node_text, concept.text)
            for purpose in expansion.questions:
                answer_to_question(current_node_text, purpose.text)

        # If we are at a core, we can link to the questions and concepts.
        elif current_node_label == "Core":
            for purpose in expansion.questions:
                core_to_question(current_node_text, purpose.text)
            for concept in expansion.concepts:
                concept_to_core(concept.text, current_node_text)

        # Grab the current node's neighbors and format them for display
        neighbors = load_neighbors(current_node_id)

        # Formatting the neighbor table
        (
            neighbors_string,
            uuid_to_simple_mapping,
            simple_to_uuid_mapping,
        ) = format_node_neighborhood(current_node_id)

        # Choose a new node if there are any neighbors
        if len(neighbors) > 0:
            old_node_id = current_node_id

            print(
                "----------------------------------------------------------------------------------"
            )
            print(neighbors_string)

            # Construct selectable nodes
            selectable_nodes = set()
            for neighbor in neighbors:
                # Add the neighbor's simple ID
                selectable_nodes.add(uuid_to_simple_mapping[neighbor["neighbor_id"]])

            # Add all the keys in the uuid_to_simple_mapping
            selectable_nodes.update(simple_to_uuid_mapping.keys())

            selectable_nodes.add("random")
            # selectable_nodes.add('core')

            # Remove the current node from the selectable nodes if it's in there
            # This prevents the AI from choosing the current node again.
            if current_node_id in selectable_nodes:
                selectable_nodes.remove(current_node_id)

            choice_prompt = (
                prompt
                + "Select a node to traverse to. Respond with the node ID."
                + "You will generate a new expansion of the node you traverse to."
                + "You will not be able to choose the current node."
                + "You may choose 'random' to choose a random node."
            )
            # "You may also choose 'core' to return to the core node, " + \
            # "or 'random' to choose a random node."

            node_selection = sg.choose(
                sg.messages(user=choice_prompt, system=system_prompt),
                choices=list(selectable_nodes),
            )

            is_random = node_selection == "random"
            is_core = node_selection == "core"

            if is_random:
                current_node_id = random_node_id()
            elif is_core:
                current_node_id = core_node_id
            else:
                current_node_id = simple_to_uuid_mapping[node_selection]

            # Print the node label + text
            print(f"SELECTED {node_selection} {current_node_id}")
            node = load_node(current_node_id)
            print(f"SELECTED {node['label'].upper()} {node['node_text']}\n")

            history.append(current_node)

            traversal_type = (
                "random" if is_random else "core" if is_core else "neighbor"
            )
            record_traversal(old_node_id, current_node_id, traversal_type)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a self-expanding knowledge graph around a core purpose."
    )

    parser.add_argument(
        "--do-clear-db",
        "--do_clear_db",
        action="store_true",
        help="If set, clear the database before proceeding.",
    )
    parser.add_argument(
        "--purpose",
        type=str,
        default="Support humanity",
        help='Set the purpose (default: "Support humanity").',
    )

    args = parser.parse_args()

    main(args.do_clear_db, args.purpose)
