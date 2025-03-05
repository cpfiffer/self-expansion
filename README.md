# Self-Expanding Knowledge Graph

The self-expanding knowledge graph is a proof-of-concept built for [an event](https://lu.ma/2jacrv79?tk=yPsIgu_) held jointly by [.txt](https://dottxt.co/), [Modal](https://modal.com/), [Neo4j](https://neo4j.com/), and [Neural Magic](https://neuralmagic.com/). 

This repo demonstrates the use of structured generation via Outlines + vLLM in AI systems engineering. Hopefully the code here inspires you to work on something similar. 

Running expander is as simple as 

```
python expand.py --purpose "Do dogs know that their dreams aren't real?"
```

but please see the setup section for installation and infrastructure.

## Overview

For more information, check out the `slides.qmd` file for the Quarto version of the slides presented. `slides.pdf` contains the rendered PDF slides. A video was recorded at some point but it is not currently available.

### Core directives

The project works by generating units of information (nodes) organized around a core directive. A core directive is anything you want the model to think about or accomplish. Core directives can be basically anything you might imagine, so try playing around with them.

Some examples include:

- "understand humans"
- "Do dogs know that their dreams aren't real?"

### The prompt

The model generally follows the following system prompt:

```
╭───────────────────────────────────────────────────────────────────────────╮
│                                                                           │
│ You are a superintelligent AI building a self-expanding knowledge graph.  │
│ Your goal is to achieve the core directive "Understand humans".           │
│                                                                           │
│ Generate an expansion of the current node. An expansion may include:      │
│                                                                           │
│ - A list of new questions.                                                │
│ - A list of new concepts.                                                 │
│ - Concepts may connect to each other.                                     │
│ - A list of new answers.                                                  │
│                                                                           │
│ Respond in the following JSON format:                                     │
│ {result_format.model_json_schema()}                                       │
│                                                                           │
╰───────────────────────────────────────────────────────────────────────────╯
```

### The graph structure

The model is allowed to generate one of four node types. This can include questions, concepts, or answers. Nodes are connected to one another using the following edges:

- `RAISES` (core/concept/answer generates question)
- `ANSWERS` (answer to question)
- `EXPLAINS` (concept to core)
- `SUGGESTS` (answer proposes new concepts)
- `IS_A` (hierarchical concept relationship)
- `AFFECTS` (causal concept relationship)
- `CONNECTS_TO` (general concept relationship)
- `TRAVERSED` (tracks navigation history)

###  Structured generation

The model uses structured generation with Outlines to generate reliably structured output from language models. The nodes a model is allowed to generate depend on its current location in the graph.

For example, if the model is on a `Question` node, it must only generate a list of questions. If the model is on a `Concept` or `Answer` node, it may generate concepts or questions.

```python
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
```

### Algorithm overview

1. Start at a node (initialized at core directive)
2. Perform an __expansion__ to generate new nodes
    - If at `Question`: answers 
    - If at `Concept`: questions + concepts
    - If at `Answer`: questions + concepts
3. Choose a related node to `TRAVERSE` to
4. Repeat forever

### The model's context

The model is shown relevant context of nodes linked to the current node, as well as semantically related nodes. Aura DB supports vector search, and this code will embed all nodes as they enter the graph database.

When a model is generating an expansion, it's prompt includes the following information:

```
ANSWER Humans have been able to benefit from AI in terms of efficiency and accuracy, but there are also concerns about job displacement and loss of personal touch.

DIRECT CONNECTIONS:
NODE-AA  SUGGESTS     CONCEPT    artificial intelligence
NODE-AE  ANSWERS      QUESTION   Do humans benefit from AI?
NODE-AJ  ANSWERS      QUESTION   What are the benefits of AI?

SEMANTICALLY RELATED:

NODE-AK  0.89         QUESTION   How does AI affect job displacement?
NODE-AL  0.88         QUESTION   How does AI maintain personal touch?

NODE-AU  0.85         CONCEPT    human ai trust
NODE-BC  0.84         CONCEPT    artificial intelligence self awareness

NODE-BG  0.89         ANSWER     Self-awareness in humans and  AI...
NODE-BN  0.89         ANSWER     Self-awareness in AI can enable ...
```

### Traversals

After a model generates an expansion, it chooses a node from it's context to traverse to by choosing from the simplified node IDs `NODE-AA`, `NODE-BB`, etc. This is a simple regular expression constraint -- structured generation ensures that the model output is exactly one of the valid nodes to traverse to.

## Set up

### Create a `.env` file

You'll need a `.env` file to store various environment variables to make sure the expander can run. There's an environment variable template in `.env.template`.

Copy it to `.env` using

```bash
cp .env.template .env
```

You'll need to set up two cloud services: the Neo4j Aura database, and Modal for LLM inference. 

### Set up Neo4j Aura

1. Go to [Neo4j's AuraDB site](https://neo4j.com/product/auradb/?ref=nav-get-started-cta).
2. Click "Start Free".
3. Select the free instance.
4. Copy the password shown to you into `NEO4J_PASSWORD` in your `.env` file.
5. Wait for your Aura DB instance to initialize.
6. Copy the ID displayed in your new instance, usually on the top left. It looks something like `db12345b`.
7. Set your `NEO4J_URI` in `.env`. Typically, URI's look like `neo4j+s://db12345b.databases.neo4j.io`. Replace `db12345b` with your instance ID.

### Set up Modal

Language model inference in this demo is cloud-native, following best practices of separating inference from the logic of your program. The inference is provided by [Modal](https://modal.com/), though any vLLM server will work.

To use Modal:

```bash
# set environment variables in .env as in .env.example
pip install -r requirements.txt
modal setup
modal run download_llama.py
modal run modal_embeddings.py  # test run of embedding service
modal deploy modal_embeddings.py  # deploy embedding service
modal run modal_vllm_container.py  # test run of llm
modal deploy modal_vllm_container.py  # deploy llm service
```

After you deploy the LLM service, you'll typically get a printout like:

```
✓ Created objects.
├── 🔨 Created mount template_llama3.jinja
├── 🔨 Created mount /blah/blah/blah/self-expansion/modal_vllm_container.py
├── 🔨 Created mount PythonPackage:download_llama
├── 🔨 Created web function serve => https://your-organization-name--self-expansion-vllm-serve.modal.run
└── 🔨 Created function infer.
✓ App deployed in 0.925s! 🎉
```

Set your `VLLM_BASE_URL` to the web function endpoint, and add `/v1` to the end of it:

```
VLLM_BASE_URL=https://your-organization-name--self-expansion-vllm-serve.modal.run/v1
```

### Running expander

```
python expand.py --purpose "Do dogs know that their dreams aren't real?"
```

### Looking at the knowledge graph

I recommend visiting your instance's query dashboard, which you can usually find here:

https://console-preview.neo4j.io/tools/query

To get a visualization of your current knowledge graph, enter this query:

```cypher
MATCH (a)-[b]-(c) 
WHERE type(b) <> 'TRAVERSED' 
RETURN *
```
