Current state of this input guardrail is as FastAPI application.

### Steps to run this app -

1. You need to have ollama model of llama3:8b installed in your local, which is getting used here.
To use openai client or any other model, make changes in the function of call_model_safely which is in file app/input_guardrails_main.py 

2. Create a virtual environment and run - pip install -r requirements

3. Then Run - python call_input_guardrails
This will start the fastapi app using uvicorn in port 8000. Log file will be generated in the same directory to store the result.


### To do next -
Create docker for this application