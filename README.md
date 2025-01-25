 # Self-Expanding Knowledge Graph


```bash
# set environment variables in .env as in .env.example
pip install -r requirements.txt
modal setup
modal run modal_embeddings.py  # test run of embedding service
modal deploy modal_embeddings.py  # deploy embedding service
modal run modal_vllm_container.py  # test run of llm
modal deploy modal_vllm_container.py  # deploy llm service
# now, look for a URL in the terminal that includes -serve.modal.run
# set that PLUS "/v1" at the end as VLLM_BASE_URL in .env
python expand.py --purpose "Do dogs know that their dreams aren't real?"  # start expanding
```
