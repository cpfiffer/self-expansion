# Self-Expanding Knowledge Graph


## Create a `.env` file

You'll need a `.env` file to store various environment variables to make sure the expander can run. There's an environment variable template in `.env.template`.

Copy it to `.env` using

```bash
cp .env.template .env
```

You'll need to set up two cloud services: the Neo4j Aura database, and Modal for LLM inference. 

## Set up Neo4j Aura

1. Go to [Neo4j's AuraDB site](https://neo4j.com/product/auradb/?ref=nav-get-started-cta).
2. Click "Start Free".
3. Select the free instance.
4. Copy the password shown to you into `NEO4J_PASSWORD` in your `.env` file.
5. Wait for your Aura DB instance to initialize.
6. Copy the ID displayed in your new instance, usually on the top left. It looks something like `db12345b`.
7. Set your `NEO4J_URI` in `.env`. Typically, URI's look like `neo4j+s://db12345b.databases.neo4j.io`. Replace `db12345b` with your instance ID.

## Set up Modal

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

## Running expander

```
# now, look for a URL in the terminal that includes -serve.modal.run
# set that PLUS "/v1" at the end as VLLM_BASE_URL in .env
python expand.py --purpose "Do dogs know that their dreams aren't real?"  # start expanding
```
