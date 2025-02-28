from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn

def generate_inputs(input_texts, tokenizer): # not using right now
    # optimize the following three lines
    input_texts_with_mask = []
    for text in input_texts:
        input_texts_with_mask.append(text + tokenizer.mask_token) #?

    # chk the default max length given by the tokenizer
    inputs = tokenizer(input_texts_with_mask,
                       padding=True,
                       trucation=True,
                       return_tensors="pt")
    return inputs

# XLM-Roberta based Charater Prediction
class XLMRCP(nn.Module):
    def __init__(self, pretrained_model_name="xlm-roberta-base"):
        super(XLMRCP, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.pretrained = AutoModelForMaskedLM.from_pretrained(pretrained_model_name)

    def forward_one_x(self, x, max_length: int = 1, top_k: int = 10000) -> str: # not str later
        """
        Predict the next character based on X (input texts)
        """
        char_pred = ""
        # X = {key: value.to(device) for key, value in X.items()}
        # X = [x + self.tokenizer.mask_token for x in X]
        x = x + self.tokenizer.mask_token
        x = self.tokenizer(x, return_tensors="pt")

        with torch.no_grad():
            out = self.pretrained(**x)
            ut = torch.zeros(1,60000)

            # Get logits for the [MASK] token
            mask_token_index = (x["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            logits = out.logits[0, mask_token_index, :]

            top_k_indices, top_k_values = torch.topk(logits, top_k, dim=-1)
            ords = [ord(self.tokenizer.decode([int(ind.item())])[0]) for ind in top_k_indices[0]]
            for i in range(top_k):
                if ords[i] < 60000:
                    ut[0,ords[i]] += top_k_values[0][i].sqrt()

            # Block

            _, ut_top = torch.topk(ut, 3, dim=-1)
            char_pred = "".join([chr(int(i.item())) for i in ut_top[0]])

        return char_pred
    
    def forward(self, X):
        return [self.forward_one_x(x) for x in X]


# test if the model work
if __name__ == '__main__':
    input_texts = ["happ",
                   "great w",
                   "I am an a",
                   "我爱",
                   "私は持っ"]
    model = XLMRCP()
    outputs = model(input_texts)
    print(len(outputs))
    print(outputs)
