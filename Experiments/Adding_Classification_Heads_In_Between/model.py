from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn

class ClassifierOnMiddleLayer(torch.nn.Module):
    def __init__(self, model_name, freeze_except, num_out):
        super(ClassifierOnMiddleLayer, self).__init__()
        self.model = AutoModel.from_pretrained(model_name, config = AutoConfig.from_pretrained(model_name, 
                                                                                                output_attentions = True, 
                                                                                                output_hidden_states = True
                                                                                              ) 
                                              )
        
        # Freeze all layers 
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze the layers that we want to train
        for param in self.model.encoder.layer[freeze_except].parameters():
            param.requires_grad = True
            
        self.classifier = nn.Sequential(
            nn.Linear(768, num_out),
            nn.LogSoftmax(dim=1)
        )
        
        self.freeze_except = freeze_except
        self.num_out = num_out
        self.model_name = model_name
    
    def forward(self, input_ids, attention_mask, label=None):
        # print(input_ids, attention_mask, label)
        # outputs  = self.bertweet_model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :].view(-1, 768)
        bert_outputs_hidden_states = self.model(input_ids, attention_mask=attention_mask).hidden_states
        # https://stackoverflow.com/questions/61465103/how-to-get-intermediate-layers-output-of-pre-trained-bert-model-in-huggingface
        # We do +1 because the first element is the output of the embedding layer
        # print(len(bert_outputs_hidden_states))
        # print(bert_outputs_hidden_states[0].shape)
        hidden_states = bert_outputs_hidden_states[self.freeze_except + 1][:, 0, :].view(-1, 768)
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