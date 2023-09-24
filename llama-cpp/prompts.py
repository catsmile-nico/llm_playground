import re, json
from dataclasses import dataclass

@dataclass
class MessageDict():
    system_prompt: str
    sample_user_1: str
    sample_response_1: str
    user_msg_2: str

    def add_msg(self, msg: str):
        self.user_msg_2 = msg

@dataclass
class PromptConfig():
    msg: str
    system_prompt = prompt = template = response = duration = log_path = ""

    def __post_init__(self):
        msg_items = MessageDict(
            self.system_prompt,
            "I subscribe to this monthly but just got an email stating that it's changing from 17 oz. to 16.9 oz. - ",
            '{ "CATEGORY": "Complaint", "SUB-CATEGORY": ["Pricing"]}',
            self.msg
        )
        self.prompt = re.sub("{(.*?)}",lambda m:str(getattr(msg_items, m.group(1))), self.template)

    def set_response(self, response):
        self.response = response

    def set_duration(self, duration):
        self.duration = duration
    
    def set_log_path(self, log_path):
        self.log_path = log_path

CLASSIFICATION_SYSTEM_PROMPT = """You are an expert in going through customer messages and categorize them for an ecommerce website.
Your responsibility is to follow the steps provided without any preamble or further questions and provide the best categories you can come up with.
You must only output in JSON format with the keys CATEGORY and SUB-CATEGORY and nothing more.
DO NOT include CLUES and REASONING in your response.
Steps to follow:
1. Read the Message delimited with ```
2. List CLUES that will help you understand the sentiment of the INPUT message (i.e., keywords, phrases, contextual information, semantic relations, semantic meaning, tones, references) that support the intent of the INPUT.
3. Deduce the diagnostic REASONING process from premises (i.e., CLUES, INPUTS) to determine what the user is actually asking.
4. Decide which CATEGORY best fit the message from the following list [Review,Inquiry,Feedback,Cancellation,Complaint,Exchange,Return,Request,Notification].
5. Come up with a generic set of SUB-CATEGORY that best fit the INPUT message.
"""
# 4. Come up with generic categories, a main category and set of sub-categories that best fit the INPUT message (i.e., feedback, review, complain, inquiry, etc.). 

class LlamaClassificationConfig(PromptConfig):
# LLAMA2 CHAT GGUF

    system_prompt = CLASSIFICATION_SYSTEM_PROMPT
    cutoff = "[/INST]"
    stop = ["```","</s>","<s>","[INST]","[/INST]"]
    template = f"""
<s>[INST] <<SYS>>
{{system_prompt}}
<</SYS>>

```
{{sample_user_1}}
``` [/INST] 
{{sample_response_1}} </s>

<s>[INST] 
```
{{user_msg_2}}
``` [/INST]"""

class WizClassificationConfig(PromptConfig):
# Wizard Mega GGUF

    system_prompt = CLASSIFICATION_SYSTEM_PROMPT
    cutoff = "### Assistant:"
    stop = ["```","###"]
    template = f"""
### Instruction: 
{{system_prompt}}

```
{{sample_user_1}}
```

### Assistant:
{{sample_response_1}}

```
{{user_msg_2}}
``` """

def parseClassificationResponse(prompt_config):
    with open(prompt_config.log_path, "a") as file: 
        json.dump(prompt_config.response, file, indent=4)

    # Parse response
    response_args = prompt_config.response["choices"][0]["text"]
    print("RAW RESPONSE")
    print(response_args)
    # print(response_args.encode().decode('unicode-escape'))
    print("="*50)
    
    # Strip the prompt and get only llm response
    first_index = response_args.find(prompt_config.cutoff) # find the first instance of cutoff
    second_index = response_args.find(prompt_config.cutoff, first_index+len(prompt_config.cutoff)) # then find thne next instance
    response_args = response_args[second_index + len(prompt_config.cutoff):]
    response_args = response_args.replace("</s>","")
    response_args = response_args.strip()
    
    print("FILTERED RESPONSE")
    print(response_args)
    print("="*50)

    # Parse response into a dict
    response_items = {
        "text":prompt_config.msg
        , "category":json.loads(response_args).get("CATEGORY")
        , "subcategory":json.loads(response_args).get("SUB-CATEGORY")
        # json.loads(response_args).get("CLUES")
        # , json.loads(response_args).get("REASONING")
        , "prompt_cost":prompt_config.response["usage"]["prompt_tokens"]
        , "completion_cost":prompt_config.response["usage"]["completion_tokens"]
        , "duration":round(prompt_config.duration, 1)
    }
    return response_items