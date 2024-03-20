from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn

class RegionProbingClassifier(torch.nn.Module):
    def __init__(self, model_name, freeze_region, bool_freeze_region_or_rest, num_out):
        super(RegionProbingClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name, config = AutoConfig.from_pretrained(model_name, 
                                                                                                output_attentions = True, 
                                                                                                output_hidden_states = True
                                                                                              ) 
                                              )
                
        regions = {
            'R1': [0, 1, 2],
            'R2': [3, 4, 5],
            'R3': [6, 7, 8],
            'R4': [9, 10, 11],
        }

        # if bool_freeze_region_or_rest is True freeze only layers in region else freeze all layers except
        if bool_freeze_region_or_rest:
            # Freeze
            for i in range(12):
                if i in regions[freeze_region]:
                    for param in self.model.encoder.layer[i].parameters():
                        param.requires_grad = False
                else:
                    for param in self.model.encoder.layer[i].parameters():
                        param.requires_grad = True
        else:
            for i in range(12):
                if i in regions[freeze_region]:
                    for param in self.model.encoder.layer[i].parameters():
                        param.requires_grad = True
                else:
                    for param in self.model.encoder.layer[i].parameters():
                        param.requires_grad = False
            
        self.classifier = nn.Sequential(
            nn.Linear(768, num_out),
            nn.LogSoftmax(dim=1)
        )
        
        self.freeze_region = freeze_region
        self.bool_freeze_region_or_rest = bool_freeze_region_or_rest
        self.num_out = num_out
        self.model_name = model_name
    
    def forward(self, input_ids, attention_mask, label=None):
        # outputs  = self.bertweet_model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :].view(-1, 768)
        hidden_states = self.model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :].view(-1, 768)
        # https://stackoverflow.com/questions/61465103/how-to-get-intermediate-layers-output-of-pre-trained-bert-model-in-huggingface
        # We do +1 because the first element is the output of the embedding layer
        # print(len(bert_outputs_hidden_states))
        # print(bert_outputs_hidden_states[0].shape)
        # hidden_states = bert_outputs_hidden_states[self.freeze_except + 1][:, 0, :].view(-1, 768)
        output = self.classifier(hidden_states)
        
        # Compute Loss
        loss = None
        if label is not None:
            loss_fct = nn.CrossEntropyLoss()
            # print(output.shape)
            # print(label.shape)
            # print(label.view(-1).shape)
            # print(output.view(-1, 2).shape)
            loss = loss_fct(output.view(-1, self.num_out), label.view(-1))
            
            return TokenClassifierOutput(loss=loss, logits=output, hidden_states=None, attentions=None)