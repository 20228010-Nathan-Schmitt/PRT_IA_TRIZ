from parrot import Parrot
import torch
import warnings
warnings.filterwarnings("ignore")

''' 
uncomment to get reproducable paraphrase generations
def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)
'''

#Init models (make sure you init ONLY once if you integrate this to your code)
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

"""phrases = ["Big wheels are required but it increases mass and space",
          "Can you recommend some upscale restaurants in Newyork?",
           "What are the famous places we should not miss in Russia?"
]

for phrase in phrases:
  print("-"*100)
  print("Input_phrase: ", phrase)
  print("-"*100)
  para_phrases = parrot.augment(input_phrase=phrase,
                               diversity_ranker="levenshtein",
                               do_diverse=False, 
                               max_return_phrases = 10, 
                               max_length=32, 
                               adequacy_threshold = 0.99, 
                               fluency_threshold = 0.90)
  for para_phrase in para_phrases:
   print(para_phrase)"""

def rephrase(sentence):
    para_phrases = parrot.augment(input_phrase=sentence)
    print("base : ",sentence)
    for i in range(min(len(para_phrases), 10)):
        para_phrase = para_phrases[i]
        print(i, "\t", para_phrase)
    i = int(input("Which sentence do we keep ? "))
    return para_phrases[i]
