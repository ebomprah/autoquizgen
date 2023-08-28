# # # install the following 
# # # pip install --quiet git+https://github.com/huggingface/transformers.git@5c00918681d6b4027701eb46cea8f795da0d4064
# # # pip install --quiet git+https://github.com/huggingface/transformers/commit/5c00918681d6b4027701eb46cea8f795da0d4064
# # # pip install --quiet sentencepiece 
# # # pip install --quiet ipython-autotime
# # # %load_ext autotime

from transformers import T5ForConditionalGeneration, T5Tokenizer

pretrained_model_name = 't5-base'

question_model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name)
question_tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name)

def get_question(sentence, answer, mdl, tknizer):
    text= "context: {} answer: {}".format(sentence, answer)
    print(text)
    max_len= 256
    encoding= tknizer.encode_plus(text, max_length= max_len, pad_to_max_length= False, 
                                  truncation= True, return_tensors= "pt" )
    
    input_ids, attention_mask= encoding["input_ids"], encoding["attention_mask"]

    outs= mdl.generate(input_ids= input_ids,
                       attention_mask=attention_mask,
                       early_stopping= True,
                       num_beams= 5,
                       num_return_sequences= 1,
                       no_repeat_ngram_size= 2,
                       max_length= 300 )
    
    dec= [tknizer.decode(ids, skip_special_tokens= True, ) for ids in outs ]
    
    Question= dec[0].replace("question:", "")
    Question= Question.strip()
    return Question

context= "Emmanuel likes to play basketball during his free time"
answer= "Emmanuel"

ques= get_question(context,answer, question_model, question_tokenizer)
print("Question: ", ques)
