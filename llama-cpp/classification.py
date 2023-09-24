import pandas as pd
import os, pytz, time

from datetime import datetime
from llama_cpp import Llama

from prompts import LlamaClassificationConfig as LLamaConfig, parseClassificationResponse
from utils import write_csv_line

# ===============
# EDIT HERE
# ===============
MODELS = {
    "LLAMA13B_Q4":"llama-2-13b-chat.Q4_K_M",
    "LLAMA13B_Q5":"llama-2-13b-chat.Q5_K_M.gguf",
    "LLAMA7B_Q4":"llama-2-7b-chat.Q4_K_M.gguf",
    "LLAMA7B_Q4":"llama-2-7b-chat.Q5_K_M.gguf",
}

DATA_FILE = "../data/fine_food_reviews_1k.csv"
OUT_FILE = "./outdata2.csv"
TEXT_COLUMN = "Text"
# MAX_TOKENS = 200
SAMPLE_SIZE = 100
PROJECT_NAME = "amazonfoodreview"

MODEL_PATH = "/home/catsmile/models/{model}.gguf".format(model=MODELS["LLAMA13B_Q4"])

# ===============
# READ DATA
# ===============
# DF = pd.read_csv(DATA_FILE, index_col=False, encoding="shift_jis", usecols=["INQUIRY_ID", "ITEM_NAME", "MSG"])
DF = pd.read_csv(DATA_FILE, index_col=False)
print(DF.head(3))
print("="*50)

# ===============
# INIT VARIOUS
# ===============
LOG_DT = str(datetime.now().astimezone(pytz.timezone('Asia/Tokyo')).strftime('%y%m%d_%H%M%S_'))
if not os.path.exists("./logs/"): os.mkdir("./logs/")

# n_gpu_layers: 0 for no GPU, -1 to offload everything to GPU
llm = Llama(model_path=MODEL_PATH, verbose=False, n_ctx=700, n_threads=1, n_gpu_layers=-1)

# ===============
# CHAT FUNCTION
# ===============
def chat_request(prompt_config, outfile=OUT_FILE):

    # Call API for initial thought prompting
    start_time = time.time()
    response = llm(
        prompt_config.prompt,
        temperature=0,
        max_tokens=100, 
        stop=prompt_config.stop, 
        echo=True,
    )
    end_time = time.time()
    duration = end_time - start_time

    prompt_config.set_response(response)
    prompt_config.set_duration(duration)
    prompt_config.set_log_path("./logs/"+LOG_DT+"_"+PROJECT_NAME+"_raw.md")

    response_parsed = parseClassificationResponse(prompt_config)
    write_csv_line(response_parsed)

    return response_parsed["category"], response_parsed["subcategory"]


# ===============
# TEST on dataframe
# ===============
# test_df = DF.sample(SAMPLE_SIZE) if SAMPLE_SIZE > 0 else DF.copy()
# test_df[['llama-cat','llama-subcat']] = test_df.apply(lambda x: chat_request(LLamaConfig(x[TEXT_COLUMN])), axis=1, result_type='expand')

# left_part, file_ext = os.path.splitext(OUT_FILE)
# outfile_clean = left_part+"-clean"+file_ext
# test_df.to_csv(outfile_clean, index=False)

# ===============
# TEST 1 line
# ===============
print(chat_request(LLamaConfig("a great product, and convenient shipment")))