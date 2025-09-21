Current state of this input guardrail is as FastAPI application.

### Steps to run this app -

1. You need to have ollama model of llama3:8b installed in your local, which is getting used here and ollama desktop should be running. By default, ollama python client connects to 11434 port.
To use openai client or any other model, make changes in the function of call_model_safely which is in file app/input_guardrails_main.py 

2. Run the docker file after building it -
     - docker build -t input-guardrails .  (Name I gave is input_guardrails)
     - docker run --rm -p 8000:8000 input-guardrails (8000 is the container port in the dockerfile)


### To do next -
-- Connect with langfuse prompt management
-- Right now, only words or single sentence matches are taken into consideration. Expand this to include scoring across multiple sentence and words tagging.
