import re, json
from dataclasses import dataclass

from utils import count_tokens, limit_tokens

@dataclass
class MessageDict():
    system_prompt: str
    user_msg: str

@dataclass
class PromptConfig():
    msg: str
    prompt = response = duration = log_path = ""

    def __post_init__(self):
        self.prompt = self.msg

    def set_response(self, response):
        self.response = response

    def set_duration(self, duration):
        self.duration = duration
    
    def set_log_path(self, log_path):
        self.log_path = log_path

class LlamaConfig(PromptConfig):
# LLAMA2 CHAT GGUF

    system_prompt = "You are a very helpful assistant. Answer the message delimited with ```"
    cutoff = "[/INST]"
    stop = ["```","</s>","<s>","[INST]","[/INST]"]
    template = f"""
<s>[INST] <<SYS>>
{{system_prompt}}
<</SYS>>

```
{{user_msg}}
``` [/INST]"""

    def __post_init__(self):
        self.init_prompt()

    def init_prompt(self):
        msg_items = MessageDict(
            system_prompt=self.system_prompt,
            user_msg=self.msg,
        )
        self.prompt = re.sub("{(.*?)}",lambda m:str(getattr(msg_items, m.group(1))), self.template)

    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        self.init_prompt()

# region CLASSIFICATION
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

@dataclass
class OneshotMessageDict(MessageDict):
    sample_user_1: str
    sample_response_1: str

class ClassificationConfig(PromptConfig):

    token_limit: int
    system_prompt = CLASSIFICATION_SYSTEM_PROMPT
    template = ""

    def __post_init__(self):
        adjusted_msg = limit_tokens(self.msg, self.token_limit) if count_tokens(self.msg) > self.token_limit else self.msg
        print("Original Token count:", count_tokens(self.msg),"Limited Token count:", count_tokens(adjusted_msg))
        self.msg = adjusted_msg

        msg_items = OneshotMessageDict(
            system_prompt=self.system_prompt,
            user_msg=self.msg,
            sample_user_1="I subscribe to this monthly but just got an email stating that it's changing from 17 oz. to 16.9 oz. - ",
            sample_response_1='{ "CATEGORY": "Complaint", "SUB-CATEGORY": ["Pricing"]}',
        )
        self.prompt = re.sub("{(.*?)}",lambda m:str(getattr(msg_items, m.group(1))), self.template)

class LlamaClassificationConfig(ClassificationConfig):
# LLAMA2 CHAT GGUF

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
{{user_msg}}
``` [/INST]"""

class WizClassificationConfig(ClassificationConfig):
# Wizard Mega GGUF

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
{{user_msg}}
``` """

def parseCategoryResponse(prompt_config):
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
# endregion

PURCHASE_REASON_SYSTEM_PROMPT = """
You are an expert in Consumer Behavior with expertise in understanding why the customer bought or return the product.
Your responsibility is to follow the steps provided without any preamble or further questions and provide the best categories you can come up with.
You must only output in JSON format with the keys CATEGORY and SUB-CATEGORY and nothing more.
DO NOT include CLUES and REASONING in your response.
Steps to follow:
1. Read the Message delimited with ```
2. List CLUES that will help you understand the sentiment of the INPUT message (i.e., keywords, phrases, contextual information, semantic relations, semantic meaning, tones, references) that support the intent of the INPUT.
3. Deduce the diagnostic REASONING process from premises (i.e., CLUES, INPUTS) to determine the user's intended message clearly identifying GOOD or BAD imppression.
4. Come up with generic categories, a main category and set of sub-categories that best describe the customer's impression of the product.
"""

class PurchaseReasonConfig(PromptConfig):
    token_limit: int
    system_prompt = PURCHASE_REASON_SYSTEM_PROMPT
    template = ""

    def __post_init__(self):
        adjusted_msg = limit_tokens(self.msg, self.token_limit) if count_tokens(self.msg) > self.token_limit else self.msg
        print("Original Token count:", count_tokens(self.msg),"Limited Token count:", count_tokens(adjusted_msg))
        self.msg = adjusted_msg

        msg_items = OneshotMessageDict(
            system_prompt=self.system_prompt,
            user_msg=self.msg,
            sample_user_1="a great product and convenient shipment",
            sample_response_1='{ "CATEGORY": "Quality", "SUB-CATEGORY": ["Convenience"]}',
        )
        self.prompt = re.sub("{(.*?)}",lambda m:str(getattr(msg_items, m.group(1))), self.template)

class LlamaPurchaseReasonConfig(PurchaseReasonConfig):
# LLAMA2 CHAT GGUF
    
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
{{user_msg}}
``` [/INST]"""

class WizardLMPurchaseReasonConfig(PurchaseReasonConfig):
# WIZARDLM GGUF

    system_prompt = PURCHASE_REASON_SYSTEM_PROMPT
    cutoff = "ASSISTANT:"
    stop = ["```","USER:", "ASSISTANT:"]
    template = f"""
{{system_prompt}}

USER:
```
{{sample_user_1}}
```
ASSISTANT:
{{sample_response_1}} 

USER:
```
{{user_msg}}
``` 

ASSISTANT: [/INST]"""